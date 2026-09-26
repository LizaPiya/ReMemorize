from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from memory_manager import ReMemorizeMemoryManager


class ReMemorizeLLM(nn.Module):
    """
    LLM backbone with two-phase adaptive memory for clinical summarization.

    ── Phase 1: Source encoding ──────────────────────────────────────────
        source tokens → LLM hidden states → key_proj / val_proj
        → GRU memory updates → m_{T_src}

    ── Phase 2: Decoding (per generated token t) ────────────────────────
        h_t  ← LLM single forward step  (past_key_values cached)
        h_aug = h_t + σ(mem_gate(h_t)) · mem_proj(m_{t-1})   injection
        next token ← lm_head(h_aug)
        m_t  ← _gru_step( val_proj(h_t), γ_t )               update

    Memory evolves step-by-step throughout generation, modelling the
    retain-vs-write dynamic *during* decoding — the central thesis.

    ── Engineering safeguards ────────────────────────────────────────────
    · past_key_values caching preserved; no attention logic is reimplemented.
    · detach_memory_hidden=True (default) detaches hidden states before
      memory updates during forward_train, preventing runaway feedback.
    · γ bias initialised to -2.0 inside GatingPolicy (small initial writes).
    · Gradient clipping applied to memory_interface_parameters() in train.py.
    · forward_train returns gamma_mean + memory_norm for monitoring.
    """

    def __init__(
        self,
        model_name_or_path: str,
        memory_manager: ReMemorizeMemoryManager,
        max_source_length: int = 2048,
        max_target_length: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = True,
        use_flash_attention: bool = True,
        freeze_llm: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.memory_manager    = memory_manager
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.dtype             = dtype

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # ── Tokenizer ─────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── LLM ───────────────────────────────────────────────────────
        load_kw: Dict = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kw["torch_dtype"] = torch.bfloat16
        else:
            load_kw["torch_dtype"] = dtype
        if use_flash_attention:
            load_kw["attn_implementation"] = "flash_attention_2"

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kw
        )
        if gradient_checkpointing and not load_in_4bit:
            self.llm.gradient_checkpointing_enable()
        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad_(False)

        H = self.llm.config.hidden_size
        self._hidden_size = H
        mem_dev = memory_manager.device

        # ── Memory interface projections ──────────────────────────────
        # LLM hidden → memory key/value spaces
        self.key_proj = nn.Linear(H, memory_manager.d_key,   bias=False, dtype=dtype).to(mem_dev)
        self.val_proj = nn.Linear(H, memory_manager.d_value, bias=False, dtype=dtype).to(mem_dev)

        # Memory state → LLM hidden space  (gated injection)
        self.mem_proj = nn.Linear(memory_manager.d_memory, H, dtype=dtype).to(mem_dev)
        self.mem_gate = nn.Linear(H, 1,                       dtype=dtype).to(mem_dev)

        # Initialise mem_proj/gate near zero for training stability
        nn.init.normal_(self.key_proj.weight, std=0.02)
        nn.init.normal_(self.val_proj.weight, std=0.02)
        nn.init.zeros_(self.mem_proj.weight)
        nn.init.zeros_(self.mem_proj.bias)
        nn.init.zeros_(self.mem_gate.weight)
        nn.init.zeros_(self.mem_gate.bias)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _llm_device(self) -> torch.device:
        return next(self.llm.parameters()).device

    @property
    def _mem_device(self) -> torch.device:
        return self.memory_manager.device

    def _inject(self, h: torch.Tensor, skip: bool = False) -> torch.Tensor:
        """
        Inject current memory state into a hidden vector (or batch).
            h_aug = h + σ(mem_gate(h)) · mem_proj(m)

        h    : (..., H)  on any device/dtype
        skip : if True, return h unchanged (ablation: no_memory)
        Returns h_aug on the same device/dtype as h.
        """
        if skip:
            return h
        m = torch.tensor(
            self.memory_manager.read(), dtype=self.dtype, device=self._mem_device
        )                                                        # (d_mem,)
        proj = self.mem_proj(m)                                  # (H,)
        gate = torch.sigmoid(self.mem_gate(h.to(self._mem_device, self.dtype)))
        h_aug = h + (gate * proj).to(h.device, h.dtype)
        return h_aug

    def _update_memory(
        self,
        h: torch.Tensor,          # (H,) single-token hidden state
        detach: bool = True,
        fixed_gamma: Optional[float] = None,
    ) -> float:
        """
        Update memory from a single hidden-state vector.
        Returns the scalar γ used for logging.

        fixed_gamma : if set, bypass the gating policy and use this constant γ
                      (ablation: fixed_gate).
        """
        h_mem = h.to(self._mem_device, torch.float32)
        if detach:
            h_mem = h_mem.detach()
        k = self.key_proj(h_mem.to(self.dtype)).float()
        v = self.val_proj(h_mem.to(self.dtype)).float()
        if fixed_gamma is not None:
            gamma = float(max(0.0, min(1.0, fixed_gamma)))
        else:
            gamma = float(
                self.memory_manager.policy([k.tolist()], [self.memory_manager.read()])[0]
            )
        self.memory_manager._gru_step(v, gamma, detach=detach)
        return gamma

    # ------------------------------------------------------------------
    # Phase 1 — Source encoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_and_update_memory(
        self,
        source: str,
        ablation_mode: str = "full",
    ) -> Tuple[List[List[float]], Dict]:
        """
        Tokenise source, run LLM forward (no grad), project hidden states
        to key/value spaces, then run sequential GRU update over all tokens.

        ablation_mode="no_memory": still encodes source (needed for key_window),
        but skips memory update so Phase 1 state stays zeroed.

        ablation_mode="fixed_gate": uses γ=0.5 for every token in Phase 1.

        Returns:
            key_window : List[List[float]]  shape (T, d_key)  for trainer
            mem_stats  : Dict               gamma stats, MSE, memory_norm
        """
        inputs = self.tokenizer(
            source, return_tensors="pt",
            truncation=True, max_length=self.max_source_length, padding=False,
        ).to(self._llm_device)

        out = self.llm(**inputs, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[-1].squeeze(0).to(self._mem_device, self.dtype)  # (T, H)

        keys   = self.key_proj(hidden).float().cpu().tolist()   # (T, d_key)
        values = self.val_proj(hidden).float().cpu().tolist()   # (T, d_value)

        self.memory_manager.reset_state()

        if ablation_mode == "no_memory":
            # Skip Phase 1 update; memory stays zeroed
            mem_stats = {
                "gamma_mean": 0.0, "gamma_std": 0.0,
                "memory_norm": 0.0, "reconstruction_mse": 0.0,
            }
        elif ablation_mode == "fixed_gate":
            T = len(keys)
            fixed_gammas = [0.5] * T
            mem_stats = self.memory_manager.update(keys, values, gammas=fixed_gammas)
        else:
            mem_stats = self.memory_manager.update(keys, values)

        return keys, mem_stats

    # ------------------------------------------------------------------
    # Phase 2 — Generation  (manual autoregressive loop)
    # ------------------------------------------------------------------

    # Supported ablation modes for generate_summary:
    #   "full"             — default: inject + update with learned gating
    #   "no_memory"        — skip injection and memory updates entirely
    #   "fixed_gate"       — inject memory but fix γ=0.5 (no learned gating)
    #   "no_phase2_update" — inject memory but freeze updates during decoding
    ABLATION_MODES = {"full", "no_memory", "fixed_gate", "no_phase2_update"}

    # Mistral-7B-v0.1 is a base (completion) model — no chat template.
    # Role is assigned via plain text; "Brief Hospital Course:" triggers completion.
    MISTRAL_BASE_MIMIC_PROMPT = (
        "You are the best clinical physician assistant, highly skilled at writing "
        "accurate and concise hospital discharge summaries strictly based on clinical notes.\n\n"
        "Clinical Note:\n{source}\n\n"
        "Instructions: Write a Brief Hospital Course summary of around 150 to 200 words. "
        "Cover the reason for admission, key findings, treatments administered, and condition at discharge. "
        "Write in plain prose. "
        "Use ONLY information explicitly stated in the note above. "
        "Do NOT fabricate or infer anything not present in the note. "
        "Do NOT use markdown, bullet points, or special formatting.\n\n"
        "Brief Hospital Course:"
    )

    MISTRAL_BASE_SOAP_PROMPT = (
        "You are the best clinical physician assistant, highly skilled at writing "
        "accurate and concise clinical summaries based on patient-doctor conversations.\n\n"
        "Patient-Doctor Conversation:\n{source}\n\n"
        "Instructions: Write a clinical summary of around 150 to 200 words. "
        "Cover the patient's chief complaint, key findings, any diagnoses or assessments, "
        "and the plan or next steps discussed. "
        "Write in plain prose. "
        "Use ONLY information explicitly stated in the conversation above. "
        "Do NOT fabricate or infer anything not present in the conversation. "
        "Do NOT use markdown, bullet points, or special formatting.\n\n"
        "Clinical Summary:"
    )

    # LLaMA-3.1-8B-Instruct chat template prompts.
    # {source} is substituted at runtime; the assistant header is left open for continuation.

    # MIMIC: clinical note → discharge summary
    LLAMA_INSTRUCT_PROMPT = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert physician. "
        "Write a concise hospital discharge summary based on the clinical note provided. "
        "The summary should be around 150 to 200 words and cover: "
        "the reason for admission, key findings, treatments administered, and condition at discharge. "
        "Write in plain prose. "
        "Use ONLY information explicitly stated in the note. "
        "Do NOT fabricate or infer anything not present in the note. "
        "Do NOT use markdown, bullet points, or special formatting. "
        "Begin your summary immediately without any introductory sentence."
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Clinical note:\n\n"
        "{source}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    # SOAP / MTS-Dialog: patient-doctor conversation → clinical summary
    SOAP_INSTRUCT_PROMPT = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are an expert physician. "
        "Write a concise clinical summary based on the patient-doctor conversation provided. "
        "The summary should be around 150 to 200 words and cover: "
        "the patient's chief complaint, key findings from the conversation, "
        "any diagnoses or assessments mentioned, and the plan or next steps discussed. "
        "Write in plain prose. "
        "Use ONLY information explicitly stated in the conversation. "
        "Do NOT fabricate or infer anything not present in the conversation. "
        "Do NOT use markdown, bullet points, or special formatting. "
        "Begin your summary immediately without any introductory sentence."
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Patient-doctor conversation:\n\n"
        "{source}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    @torch.no_grad()
    def generate_summary(
        self,
        source: str,
        prompt_template: str = None,
        dataset_type: str = "mimic",
        max_new_tokens: int = 512,
        min_new_tokens: int = 200,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 4,
        do_sample: bool = True,
        detach_memory_hidden: bool = True,
        ablation_mode: str = "full",
        return_ids: bool = False,
    ):
        """
        Generate a summary with memory that continues evolving during decoding.

        Uses HuggingFace past_key_values caching — no attention logic is
        reimplemented. At each step:
            1. Single LLM forward → h_t  (cached KV, O(1) per step)
            2. Augment h_t with current m_{t-1}  → h_aug_t
            3. Sample next token from lm_head(h_aug_t)
            4. Update m_t from h_t  (detach controlled by detach_memory_hidden)

        ablation_mode controls which memory components are active:
            "full"             — full model (default)
            "no_memory"        — skip injection and updates (pure LLM baseline)
            "fixed_gate"       — inject memory but fix γ=0.5 (no learned gating)
            "no_phase2_update" — inject Phase 1 memory but freeze during decoding

        Assumes encode_and_update_memory() has already been called.

        return_ids: when True, returns (text, prompt_ids, generated_ids) instead
            of just the decoded text -- generated_ids is a (1, T) LongTensor of
            the raw sampled token ids, needed by callers that must recompute
            this generation's log-probability later (e.g. GRPO's importance
            ratio in train_dpo_baseline.py). Default False preserves the
            original str-only return for all existing callers.
        """
        if ablation_mode not in self.ABLATION_MODES:
            raise ValueError(
                f"Unknown ablation_mode {ablation_mode!r}. "
                f"Choose from {self.ABLATION_MODES}"
            )
        if prompt_template is None:
            is_mistral_base = "mistral" in self.model_name_or_path.lower() and "instruct" not in self.model_name_or_path.lower()
            if is_mistral_base:
                prompt_template = self.MISTRAL_BASE_SOAP_PROMPT if dataset_type == "mts_dialog" else self.MISTRAL_BASE_MIMIC_PROMPT
            elif dataset_type == "mts_dialog":
                prompt_template = self.SOAP_INSTRUCT_PROMPT
            else:
                prompt_template = self.LLAMA_INSTRUCT_PROMPT
        char_budget = self.max_source_length * 4
        prompt = prompt_template.format(source=source[:char_budget])

        # Mistral base model needs add_special_tokens=True for <s> BOS;
        # LLaMA instruct prompts already contain <|begin_of_text|> so False.
        is_mistral_base = "mistral" in self.model_name_or_path.lower() and "instruct" not in self.model_name_or_path.lower()
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=self.max_source_length,
            add_special_tokens=is_mistral_base,
        ).to(self._llm_device)

        prompt_ids   = inputs["input_ids"]           # (1, T_prompt), kept for return_ids
        past_kv      = None
        current_ids  = prompt_ids                    # (1, T_prompt) first step
        generated    : List[int] = []

        for _ in range(max_new_tokens):
            out = self.llm(
                input_ids=current_ids,
                past_key_values=past_kv,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values

            # Last-position hidden state  (1, H)
            h_t = out.hidden_states[-1][:, -1, :]

            # Inject current memory (skip entirely for no_memory ablation)
            skip_inject = (ablation_mode == "no_memory")
            h_aug = self._inject(h_t, skip=skip_inject)         # (1, H)

            # Project to logits and sample
            logits = self.llm.lm_head(h_aug.to(self._llm_device, self.dtype))  # (1, vocab)

            if do_sample:
                logits_f = logits[0].float()

                # Repetition penalty: downscale logits for already-generated tokens
                if repetition_penalty != 1.0 and generated:
                    for tok_id in set(generated):
                        if logits_f[tok_id] > 0:
                            logits_f[tok_id] /= repetition_penalty
                        else:
                            logits_f[tok_id] *= repetition_penalty

                # No-repeat n-gram: block any token that would extend a repeated n-gram
                if no_repeat_ngram_size > 0 and len(generated) >= no_repeat_ngram_size - 1:
                    ngram_prefix = tuple(generated[-(no_repeat_ngram_size - 1):])
                    for i in range(len(generated) - no_repeat_ngram_size + 1):
                        if tuple(generated[i:i + no_repeat_ngram_size - 1]) == ngram_prefix:
                            banned = generated[i + no_repeat_ngram_size - 1]
                            logits_f[banned] = float("-inf")

                logits_f = logits_f / max(temperature, 1e-6)
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits_f, descending=True)
                    cum_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    remove = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[remove] = float("-inf")
                    logits_f = torch.zeros_like(logits_f).scatter_(
                        0, sorted_idx, sorted_logits
                    )
                probs    = torch.softmax(logits_f, dim=-1)
                next_tok = int(torch.multinomial(probs, 1).item())
            else:
                next_tok = int(logits[0].argmax(-1).item())

            generated.append(next_tok)
            if next_tok == self.tokenizer.eos_token_id and len(generated) >= min_new_tokens:
                break

            # Update memory from this decoding step
            # no_memory and no_phase2_update: skip Phase 2 updates
            if ablation_mode in ("full", "fixed_gate"):
                fixed_g = 0.5 if ablation_mode == "fixed_gate" else None
                self._update_memory(
                    h_t.squeeze(0),
                    detach=detach_memory_hidden,
                    fixed_gamma=fixed_g,
                )

            # Advance: feed only the new token (past_kv handles context)
            current_ids = torch.tensor([[next_tok]], device=self._llm_device)

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if return_ids:
            generated_ids = torch.tensor([generated], device=prompt_ids.device, dtype=torch.long)
            return text, prompt_ids, generated_ids
        return text

    # ------------------------------------------------------------------
    # Phase 2 — Training forward  (teacher-forced + sequential memory)
    # ------------------------------------------------------------------

    def forward_train(
        self,
        source: str,
        target: str,
        detach_memory_hidden: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Teacher-forced training with sequential memory augmentation.

        Steps:
          1. Source encoding (no grad) → m_{T_src}.
          2. Full LLM forward over [source | target] → hidden states (T_total, H).
          3. Sequential loop over target positions:
               a. Inject m_{t-1} into h_t  → h_aug_t.
               b. Update m_t from h_t  (detach controlled by detach_memory_hidden).
          4. CE loss on target tokens from augmented logits.
          5. Auxiliary reconstruction loss to train write_proj
             (since the GRU update is no_grad, write_proj needs a separate path):
               aux = MSE( tanh(write_proj(val_proj(h_tgt))), m_T.detach() )

        Returns (loss, stats) where stats includes gamma_mean, memory_norm, mse.
        """
        # ── Step 1: source encoding ────────────────────────────────────
        _, mem_stats = self.encode_and_update_memory(source)

        # ── Step 2: build teacher-forcing input ───────────────────────
        src_ids = self.tokenizer(
            source, truncation=True, max_length=self.max_source_length
        )["input_ids"]
        tgt_ids = (
            self.tokenizer(target, truncation=True, max_length=self.max_target_length
        )["input_ids"] + [self.tokenizer.eos_token_id])

        max_total = self.max_source_length + self.max_target_length
        full_ids  = (src_ids + tgt_ids)[:max_total]
        labels    = ([-100] * len(src_ids) + tgt_ids)[:len(full_ids)]

        input_ids = torch.tensor([full_ids], device=self._llm_device)
        label_ids = torch.tensor([labels],   device=self._llm_device)

        # ── Step 3: LLM forward — teacher-forced ──────────────────────
        out    = self.llm(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        hidden = out.hidden_states[-1].squeeze(0)               # (T_total, H)

        tgt_start  = len(src_ids)
        tgt_hidden = hidden[tgt_start:]                         # (T_tgt, H)
        T_tgt      = tgt_hidden.size(0)

        # ── Step 4: sequential memory augmentation over target tokens ──
        h_aug_list   : List[torch.Tensor] = []
        gamma_values : List[float]        = []

        for t in range(T_tgt):
            h_t = tgt_hidden[t]                                 # (H,)

            # Inject m_{t-1} (stop-gradient on memory state itself)
            h_t_aug = self._inject(h_t)                         # (H,)
            h_aug_list.append(h_t_aug)

            # Update memory from this step; detach hidden to avoid feedback
            g = self._update_memory(h_t, detach=detach_memory_hidden)
            gamma_values.append(g)

        # ── Step 5: CE loss on target tokens ──────────────────────────
        h_aug_tgt = torch.stack(h_aug_list)                     # (T_tgt, H)
        logits    = self.llm.lm_head(
            h_aug_tgt.to(self._llm_device, self.dtype)
        )                                                        # (T_tgt, vocab)

        shift_logits = logits[:-1].contiguous()
        shift_labels = label_ids[0, tgt_start + 1 : tgt_start + T_tgt].contiguous()
        ce_loss = nn.functional.cross_entropy(
            shift_logits.float(), shift_labels, ignore_index=-100
        )

        # ── Step 6: auxiliary reconstruction loss (trains write_proj) ─
        # write_proj is updated analytically in _gru_step (no_grad), so
        # it needs a separate differentiable path through the CE graph.
        h_mem  = tgt_hidden.to(self._mem_device, self.dtype)
        V_tgt  = self.val_proj(h_mem)                           # (T_tgt, d_value)
        C_tgt  = torch.tanh(self.memory_manager.write_proj(V_tgt.float()))  # (T_tgt, d_mem)
        m_final = torch.tensor(
            self.memory_manager.read(), dtype=C_tgt.dtype, device=C_tgt.device
        )                                                        # (d_mem,) detached
        aux_loss = nn.functional.mse_loss(
            C_tgt, m_final.unsqueeze(0).expand_as(C_tgt).detach()
        )

        loss = ce_loss + 0.1 * aux_loss

        return loss, {
            "ce_loss":     ce_loss.item(),
            "aux_loss":    aux_loss.item(),
            "mse":         mem_stats["reconstruction_mse"],
            "gamma_mean":  sum(gamma_values) / max(len(gamma_values), 1),
            "gamma_std":   float(torch.tensor(gamma_values).std()) if T_tgt > 1 else 0.0,
            "memory_norm": self.memory_manager.memory_norm(),
        }

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        save_dir: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        global_step: int = 0,
        best_rouge1: float = 0.0,
        memory_config: Optional[Dict] = None,
    ) -> None:
        """
        Save memory interface weights + optional training state.

        Always saved:
          memory_interface.pt  — weights for key_proj, val_proj, mem_proj,
                                  mem_gate, memory_manager (including GatingPolicy)
          config.json          — memory dimensions + model name, so load_checkpoint
                                  can reconstruct ReMemorizeMemoryManager without
                                  the caller needing to remember the original args.

        Optionally saved (when optimizer is provided):
          training_state.pt    — optimizer state, scheduler state, global_step,
                                  best_rouge1, and GatingPolicy's own optimizer.
                                  Required for proper training resumption.
        """
        import json
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        # ── Weights ──────────────────────────────────────────────────────────
        torch.save(
            {
                "key_proj":       self.key_proj.state_dict(),
                "val_proj":       self.val_proj.state_dict(),
                "mem_proj":       self.mem_proj.state_dict(),
                "mem_gate":       self.mem_gate.state_dict(),
                "memory_manager": self.memory_manager.state_dict(),
            },
            path / "memory_interface.pt",
        )

        # ── Config — needed to reconstruct memory_manager on load ────────────
        cfg = memory_config or {}
        cfg.setdefault("d_key",    self.memory_manager.d_key)
        cfg.setdefault("d_value",  self.memory_manager.d_value)
        cfg.setdefault("d_memory", self.memory_manager.d_memory)
        with open(path / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        # ── Tokenizer ────────────────────────────────────────────────────────
        self.tokenizer.save_pretrained(str(path / "tokenizer"))

        # ── Training state (optional) ────────────────────────────────────────
        if optimizer is not None:
            training_state: Dict = {
                "global_step":        global_step,
                "best_rouge1":        best_rouge1,
                "optimizer":          optimizer.state_dict(),
                "policy_optimizer":   self.memory_manager.policy.optimizer.state_dict(),
            }
            if scheduler is not None:
                training_state["scheduler"] = scheduler.state_dict()
            torch.save(training_state, path / "training_state.pt")

    @classmethod
    def load_checkpoint(
        cls,
        save_dir: str,
        model_name_or_path: str,
        memory_manager: Optional[ReMemorizeMemoryManager] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler=None,
        **init_kwargs,
    ) -> Tuple["ReMemorizeLLM", Dict]:
        """
        Load a checkpoint. Returns (model, resume_info).

        resume_info keys:
          global_step  — step to resume from (0 if no training_state.pt)
          best_rouge1  — best ROUGE-1 seen so far (0.0 if not saved)

        If memory_manager is None, reconstructs it automatically from config.json.
        If optimizer/scheduler are provided and training_state.pt exists,
        restores their states for proper training resumption.
        """
        import json
        path = Path(save_dir)

        # ── Reconstruct memory_manager from saved config if not provided ─────
        if memory_manager is None:
            cfg_path = path / "config.json"
            if not cfg_path.exists():
                raise FileNotFoundError(
                    f"config.json not found in {save_dir}. "
                    "Either provide memory_manager explicitly or use a checkpoint "
                    "saved with the updated save_checkpoint()."
                )
            with open(cfg_path) as f:
                cfg = json.load(f)
            memory_manager = ReMemorizeMemoryManager(
                d_key=cfg["d_key"],
                d_value=cfg["d_value"],
                d_memory=cfg["d_memory"],
            )

        model = cls(
            model_name_or_path=model_name_or_path,
            memory_manager=memory_manager,
            **init_kwargs,
        )

        # ── Weights ──────────────────────────────────────────────────────────
        ckpt = torch.load(path / "memory_interface.pt", map_location="cpu")
        model.key_proj.load_state_dict(ckpt["key_proj"])
        model.val_proj.load_state_dict(ckpt["val_proj"])
        model.mem_proj.load_state_dict(ckpt["mem_proj"])
        model.mem_gate.load_state_dict(ckpt["mem_gate"])
        model.memory_manager.load_state_dict(ckpt["memory_manager"])

        # ── Training state (optional) ────────────────────────────────────────
        resume_info: Dict = {"global_step": 0, "best_rouge1": 0.0}
        ts_path = path / "training_state.pt"
        if ts_path.exists():
            ts = torch.load(ts_path, map_location="cpu")
            resume_info["global_step"] = ts.get("global_step", 0)
            resume_info["best_rouge1"] = ts.get("best_rouge1", 0.0)
            if optimizer is not None and "optimizer" in ts:
                optimizer.load_state_dict(ts["optimizer"])
            if scheduler is not None and "scheduler" in ts:
                scheduler.load_state_dict(ts["scheduler"])
            if "policy_optimizer" in ts:
                model.memory_manager.policy.optimizer.load_state_dict(ts["policy_optimizer"])

        return model, resume_info

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------

    def memory_interface_parameters(self):
        """
        Parameters for the memory interface (updated by backbone optimiser).
        Gradient clipping should be applied to this group in train.py.
        """
        return (
            list(self.key_proj.parameters())
            + list(self.val_proj.parameters())
            + list(self.mem_proj.parameters())
            + list(self.mem_gate.parameters())
            + list(self.memory_manager.write_proj.parameters())
        )
