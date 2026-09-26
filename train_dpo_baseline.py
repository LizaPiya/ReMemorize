"""
DPO/GRPO-direct baseline: applies the same composite reward directly to the
LM via RL, with no memory module -- isolating whether ReMemorize's gains
come from adaptive memory or just from reward-tuned generation in general.

Algorithm: standard, textbook GRPO applied directly to the LLM's own
generation via LoRA -- importance-sampled clipped surrogate loss on
sequence-level log-probabilities, rather than the reward-weighted-CE-toward-
best-candidate approach the full ReMemorize model uses for its memory
interface. This is a deliberate choice: ReMemorize's GRPO only trains the
small gating policy, and reward reaches the memory/LLM weights via a
separate mechanism (reward-weighted regression toward the best-of-G
candidate). Since there's no gating policy here (no memory module at all),
that mechanism has nothing to act on, so this baseline uses real
policy-gradient RL on generation instead -- an explicit, disclosed
algorithmic difference from the full model, not just an architecture
ablation.

Pure RL fine-tuning with no supervised anchor and no penalty for drifting
from the original policy can converge to a degenerate reward-hacking
attractor (e.g. collapsing to a single repeated token) that is trivially
self-consistent step to step, so a naive step-to-step KL check alone does
not catch it. This script adds a KL-to-reference-policy penalty
(--kl_ref_coef), a hard KL early-stop, an independent repetition check, and
periodic sample logging (--log_sample_every) so training can be visually
inspected for degeneration rather than trusted from loss/KL numbers alone.

Reuses, unmodified, everything that must match Full model for a fair
comparison:
  - ReMemorizeLLM.generate_summary(ablation_mode="no_memory") for sampling
    (same prompt templates, same decoding loop, same repetition penalty --
    exactly what the existing "w/o Memory" ablation already uses).
  - NLIRewardScorer (same chunked DeBERTa factuality/hallucination, same
    sentence-embedding completeness -- scorers.py, unmodified).
  - RewardCalibrator(ema_alpha=0.1, clip_z=2.5, factual_floor=0.5,
    factual_floor_penalty=0.2) -- identical to train.py's calibrator, so
    reward normalization behaves the same way.
  - Same reward weights (--reward_alpha/beta/gamma/delta), same training
    data, same max_source_length/max_new_tokens conventions as train.py.

A throwaway ReMemorizeMemoryManager is constructed only because
ReMemorizeLLM's constructor requires one -- it is never trained, never
updated, and its parameters are excluded from the optimizer. Only the LoRA
adapter weights on the LLM are trainable.

Usage:
    python train_dpo_baseline.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset mimic \
        --train_file Datasets/mimic_short_5k_train.jsonl \
        --eval_file  Datasets/mimic_short_5k_val.jsonl \
        --max_train 1000 --max_source_length 2048 --max_new_tokens 280 \
        --reward_alpha 0.7 --reward_beta 0.5 --reward_gamma 0.3 --reward_delta 1.2 \
        --output_dir runs/dpo_baseline/mimic
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import signal
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from dataset import MIMICDataset, MTSDialogDataset, ReMemorizeDataset, SummarizationExample, build_dataloader
from llm_backbone import ReMemorizeLLM
from memory_manager import ReMemorizeMemoryManager
from reward_manager import RewardCalibrator, RewardWeights
from scorers import NLIRewardScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("dpo_baseline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dataset", type=str, default="mimic", choices=["mimic", "mts_dialog"])
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default=None)
    p.add_argument("--max_train", type=int, default=None)

    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--group_size", type=int, default=4, help="G candidates sampled per document for GRPO.")
    p.add_argument("--max_source_length", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=280)
    p.add_argument("--min_new_tokens", type=int, default=80)
    p.add_argument("--sample_temperature", type=float, default=0.7,
                    help="Temperature for candidate sampling. Higher than eval's 0.1 -- "
                         "GRPO needs genuine variation across the G candidates within a group.")
    p.add_argument("--no_flash_attn", action="store_true")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0,
                    help="Must stay 0 for this on-policy GRPO setup: nonzero dropout makes the "
                         "old/new log-prob comparison used for the importance ratio diverge once "
                         "LoRA's B matrix moves off its zero init, growing worse with more training "
                         "(dropout stochastically perturbs the LoRA forward pass differently each "
                         "call, once it is actually contributing something nonzero).")
    p.add_argument("--lora_target_modules", type=str,
                    default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                    help="Comma-separated module names to attach LoRA adapters to.")

    # Reward weights (must match the configuration used to train the full model)
    p.add_argument("--reward_alpha", type=float, default=1.0)
    p.add_argument("--reward_beta", type=float, default=0.3)
    p.add_argument("--reward_gamma", type=float, default=0.1)
    p.add_argument("--reward_delta", type=float, default=0.8)

    # GRPO (matches policy_manager.py's PPO/GRPO conventions)
    p.add_argument("--lr", type=float, default=1e-6,
                    help="Matches the GRPO learning rate reported in DeepSeekMath "
                         "(Shao et al., 2024), the paper this project's GRPO methodology is based "
                         "on. A substantially higher learning rate destabilizes training even with "
                         "a KL-to-reference penalty active.")
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--max_kl", type=float, default=0.02)
    p.add_argument("--group_eps", type=float, default=1e-8, help="Numerical stabilizer for group std.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--kl_ref_coef", type=float, default=0.04,
                    help="INITIAL coefficient for the KL-to-reference-policy penalty (reward -= "
                         "coef * (logp_current - logp_reference), reference = frozen base model via "
                         "peft's disable_adapter()). Matches the KL coefficient reported in "
                         "DeepSeekMath (Shao et al., 2024). The coefficient is adapted during "
                         "training rather than held fixed, see --kl_target/--kl_horizon below.")
    p.add_argument("--kl_target", type=float, default=8.0,
                    help="Target KL-to-reference (nats) for the adaptive controller (Ziegler et al. "
                         "2019 / InstructGPT-style). Published guidance suggests KL should stay in "
                         "roughly 5-10 nats; without adaptive control it can grow into a runaway "
                         "regime with degenerate generations.")
    p.add_argument("--kl_horizon", type=int, default=50,
                    help="Adaptation horizon (in documents) for the adaptive KL controller -- how fast "
                         "the coefficient responds to KL drifting from --kl_target. Set lower than the "
                         "original paper's larger-scale default so the controller reacts quickly at "
                         "this training scale. The hard KL/repetition stops below remain the primary "
                         "safety net; this just reduces how often they're needed.")
    p.add_argument("--kl_hard_stop", type=float, default=15.0,
                    help="Hard early-stop: if the EMA of kl_to_ref exceeds this (nats), stop training "
                         "immediately and save whatever checkpoint exists, rather than continuing on "
                         "an already-degenerating policy. Set with a safety margin below the KL range "
                         "associated with degenerate generations in the literature.")
    p.add_argument("--repetition_threshold", type=float, default=0.3,
                    help="Hard early-stop: if any single word makes up more than this fraction of a "
                         "generated candidate's words, treat it as degenerate repetition and stop. "
                         "Independent of the KL-based stop, since kl_to_ref can be near zero or "
                         "negative during single-word repetition collapse, so the KL metric alone "
                         "does not reliably catch this failure mode.")
    p.add_argument("--log_sample_every", type=int, default=10,
                    help="Print one full decoded candidate + its reward to the log every N steps, "
                         "so training can be visually checked for degeneration rather than trusted "
                         "from loss/KL numbers alone, which can remain nominal even under collapse.")

    p.add_argument("--save_every", type=int, default=250)
    p.add_argument("--eval_every", type=int, default=250,
                    help="NOT YET IMPLEMENTED -- periodic eval-based checkpoint selection isn't "
                         "wired up yet (best_ckpt currently just mirrors latest_ckpt at the end "
                         "of training). This flag is accepted for CLI parity with train.py but "
                         "is currently a no-op. Passing it will not error, but will not do anything.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume_from", type=str, default=None,
                    help="Path to a checkpoint directory (latest_ckpt/) to resume from. "
                         "Restores LoRA weights, optimizer state, and global_step.")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


class AdaptiveKLController:
    """
    Standard adaptive KL controller (Ziegler et al. 2019, "Fine-Tuning Language
    Models from Human Preferences" -- the same mechanism used in InstructGPT).
    Raises the KL coefficient when measured KL runs above target, relaxes it
    when comfortably below -- unlike a static coefficient, this doesn't let KL
    run away just because reward gains outweigh a fixed penalty.
    """
    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int = 1) -> None:
        proportional_error = max(-0.2, min(0.2, current_kl / self.target - 1))
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


def is_repetitive(text: str, threshold: float) -> bool:
    """
    True if any single word makes up more than `threshold` of all words --
    catches single-token/word collapse that kl_to_ref does not reliably
    flag, since KL-to-reference can be near zero or negative during this
    failure mode.
    """
    words = text.split()
    if len(words) < 10:
        return False
    counts: dict = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return max(counts.values()) / len(words) > threshold


# ── Sequence-level log-probability (standard PPO/GRPO machinery for an LM policy) ──

def sequence_logprob(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    device: str,
    requires_grad: bool,
) -> torch.Tensor:
    """
    Sum of log p(token_t | prefix) under `model` for the generated continuation,
    computed via one teacher-forced forward pass (not autoregressive replay).
    prompt_ids: (1, T_prompt). generated_ids: (1, T_gen).
    """
    full_ids = torch.cat([prompt_ids, generated_ids], dim=1).to(device)
    ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()
    with ctx:
        out = model(input_ids=full_ids)
        logits = out.logits[:, prompt_ids.shape[1] - 1: -1, :]  # predict each generated token
        logprobs = F.log_softmax(logits.float(), dim=-1)
        token_logprobs = logprobs.gather(-1, generated_ids.to(device).unsqueeze(-1)).squeeze(-1)
    return token_logprobs.sum()


def save_checkpoint(save_dir: str, backbone: ReMemorizeLLM, optimizer, global_step: int) -> None:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    backbone.llm.save_pretrained(str(path))  # peft: saves only the LoRA adapter weights
    torch.save(
        {"optimizer": optimizer.state_dict(), "global_step": global_step},
        path / "training_state.pt",
    )


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)
    log.warning(
        "--eval_every is accepted but NOT YET IMPLEMENTED: no periodic eval runs, "
        "best_ckpt will just mirror latest_ckpt at the end of training."
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Throwaway memory manager: ReMemorizeLLM's constructor requires one, but
    # it is never trained/updated -- ablation_mode="no_memory" everywhere below
    # means it's never even touched during generation. ────────────────────────
    inert_mm = ReMemorizeMemoryManager(d_key=64, d_value=64, d_memory=128, seed=args.seed, device=device)

    backbone = ReMemorizeLLM(
        args.model,
        inert_mm,
        max_source_length=args.max_source_length,
        max_target_length=args.max_new_tokens,
        use_flash_attention=not args.no_flash_attn,
        freeze_llm=False,  # will be superseded by LoRA wrapping below
    )

    # ── LoRA-wrap the LLM; freeze everything else (base LLM weights AND the
    # inert memory interface -- only LoRA adapter weights are trainable).
    #
    # Resume uses PeftModel.from_pretrained() directly on this still-unwrapped
    # base model, NOT a re-wrap via .get_base_model() on an already-PEFT-wrapped
    # instance -- the latter throws a "multiple adapters" warning from peft
    # and is not a safe pattern. Fresh base + direct from_pretrained is the
    # clean, warning-free path. ─────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model, PeftModel
    for p in backbone.llm.parameters():
        p.requires_grad_(False)

    global_step = 0
    if args.resume_from:
        log.info("Resuming from checkpoint: %s", args.resume_from)
        backbone.llm = PeftModel.from_pretrained(backbone.llm, args.resume_from, is_trainable=True).to(device)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            task_type="CAUSAL_LM",
            bias="none",
        )
        backbone.llm = get_peft_model(backbone.llm, lora_config).to(device)
    backbone.llm.print_trainable_parameters()

    trainable_params = [p for p in backbone.llm.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)

    if args.resume_from:
        state_path = Path(args.resume_from) / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(state["optimizer"])
            global_step = state["global_step"]
            log.info("Resumed at global_step=%d", global_step)

    # ── Reward pipeline: identical to Full model's (train.py:341, verified) ──
    scorer = NLIRewardScorer(device=device)
    calibrator = RewardCalibrator(ema_alpha=0.1, clip_z=2.5, factual_floor=0.5, factual_floor_penalty=0.2)
    weights = RewardWeights(
        alpha=args.reward_alpha, beta=args.reward_beta,
        gamma=args.reward_gamma, delta=args.reward_delta,
    )
    kl_controller = AdaptiveKLController(args.kl_ref_coef, args.kl_target, args.kl_horizon)
    kl_ema: Optional[float] = None

    # ── Dataset ────────────────────────────────────────────────────────────
    DatasetCls = MIMICDataset if args.dataset == "mimic" else MTSDialogDataset
    train_dataset: ReMemorizeDataset = DatasetCls.from_jsonl(args.train_file, max_examples=args.max_train)
    eval_dataset: Optional[ReMemorizeDataset] = DatasetCls.from_jsonl(args.eval_file) if args.eval_file else None
    log.info("Loaded %d train examples%s.", len(train_dataset),
             f" / {len(eval_dataset)} eval" if eval_dataset else "")
    dataloader = build_dataloader(train_dataset, batch_size=1, shuffle=True, seed=args.seed)

    # ── Emergency checkpoint on SLURM's SIGTERM (24h-cap safety net) ─────────
    def _sigterm_handler(signum, frame):
        log.warning("SIGTERM received -- saving emergency checkpoint before exit.")
        save_checkpoint(str(output_dir / "latest_ckpt"), backbone, optimizer, global_step)
        log.info("Emergency checkpoint saved to %s/latest_ckpt", output_dir)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Resume-by-step-count only correctly skips within epoch 1 -- fine for the
    # intended --num_epochs 1 usage (matching train.py's convention for this
    # baseline), but does NOT correctly resume into a later epoch if
    # --num_epochs > 1 is ever used. Unlike train.py, no epoch-aware resume_epoch
    # logic here.
    resume_skip = global_step

    # ── Training: standard GRPO on the LM's own generation, via LoRA ─────────
    for epoch in range(1, args.num_epochs + 1):
        log.info("=== Epoch %d / %d ===", epoch, args.num_epochs)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.num_epochs}", unit="doc")
        for batch_idx, batch in enumerate(pbar):
            if epoch == 1 and batch_idx < resume_skip:
                continue

            for ex in batch:
                assert isinstance(ex, SummarizationExample)
                source = ex.source
                reference = ex.reference or ex.target

                # 1. Sample G candidates from the CURRENT LoRA policy (no memory).
                backbone.llm.eval()
                candidates: List[str] = []
                gen_ids: List[torch.Tensor] = []
                prompt_ids_ref: Optional[torch.Tensor] = None
                with torch.no_grad():
                    for _ in range(args.group_size):
                        summary, prompt_ids, out_ids = backbone.generate_summary(
                            source,
                            dataset_type=args.dataset,
                            max_new_tokens=args.max_new_tokens,
                            min_new_tokens=args.min_new_tokens,
                            temperature=args.sample_temperature,
                            do_sample=True,
                            ablation_mode="no_memory",
                            return_ids=True,
                        )
                        candidates.append(summary)
                        gen_ids.append(out_ids)
                        prompt_ids_ref = prompt_ids

                # 1b. Repetition hard-stop, independent of the KL-based stop below --
                # kl_to_ref can be near zero or negative during single-word
                # repetition collapse, so KL alone does not reliably catch this
                # failure mode.
                if any(is_repetitive(c, args.repetition_threshold) for c in candidates):
                    log.warning(
                        "step=%d HARD STOP: repetition detected in a generated candidate "
                        "(a single word exceeds %.0f%% of its words). Saving checkpoint and exiting.",
                        global_step, args.repetition_threshold * 100,
                    )
                    save_checkpoint(str(output_dir / "latest_ckpt"), backbone, optimizer, global_step)
                    log.info("step=%d SAMPLE at stop: %r", global_step, candidates[0][:300])
                    return

                # 2. Score with the identical reward pipeline as Full model.
                base_rewards = []
                for cand in candidates:
                    comp = scorer.score(source=source, summary=cand, reference=reference)
                    stats = calibrator.calibrate(comp, w=weights)
                    base_rewards.append(stats.calibrated_reward)

                # 3. Old (pre-update) log-probs, frozen for the importance ratio.
                old_logprobs = [
                    sequence_logprob(backbone.llm, prompt_ids_ref, g_ids, device, requires_grad=False).detach()
                    for g_ids in gen_ids
                ]

                # 3b. KL-to-reference-policy penalty (see --kl_ref_coef help text):
                # without this, nothing stops the policy drifting into a degenerate
                # attractor over many steps, since a collapsed policy is trivially
                # self-consistent step to step and the step-to-step KL check above
                # does not catch it. Reference = frozen base model, LoRA adapter
                # disabled, no second model copy needed.
                with torch.no_grad(), backbone.llm.disable_adapter():
                    ref_logprobs = [
                        sequence_logprob(backbone.llm, prompt_ids_ref, g_ids, device, requires_grad=False).detach()
                        for g_ids in gen_ids
                    ]
                kl_to_ref = [float(old_lp - ref_lp) for old_lp, ref_lp in zip(old_logprobs, ref_logprobs)]
                rewards = [r - kl_controller.value * kl for r, kl in zip(base_rewards, kl_to_ref)]

                mean_r = sum(rewards) / len(rewards)
                var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
                std_r = max(var_r, args.group_eps) ** 0.5
                advantages = [(r - mean_r) / std_r for r in rewards]

                # Adaptive KL controller update (Ziegler et al. 2019 / InstructGPT-style) --
                # raises kl_controller.value when KL runs above target, relaxes it when
                # comfortably below. Uses the group's mean kl_to_ref for this step.
                step_mean_kl = sum(kl_to_ref) / len(kl_to_ref)
                kl_controller.update(step_mean_kl)
                kl_ema = step_mean_kl if kl_ema is None else 0.9 * kl_ema + 0.1 * step_mean_kl

                if kl_ema > args.kl_hard_stop:
                    log.warning(
                        "step=%d HARD STOP: KL-to-reference EMA=%.2f exceeds --kl_hard_stop=%.2f. "
                        "Saving checkpoint and exiting rather than continuing on a degenerating policy.",
                        global_step, kl_ema, args.kl_hard_stop,
                    )
                    save_checkpoint(str(output_dir / "latest_ckpt"), backbone, optimizer, global_step)
                    log.info("step=%d SAMPLE at stop: %r", global_step, candidates[0][:300])
                    return

                if global_step % args.log_sample_every == 0:
                    best_i = base_rewards.index(max(base_rewards))
                    log.info(
                        "step=%d SAMPLE (base_reward=%.3f, kl_to_ref=%.3f, kl_ema=%.3f, kl_coef=%.4f): %r",
                        global_step, base_rewards[best_i], kl_to_ref[best_i], kl_ema, kl_controller.value,
                        candidates[best_i][:300],
                    )

                # 4. GRPO clipped-surrogate update (Eq. rl_obj) into LoRA weights only.
                backbone.llm.train()
                optimizer.zero_grad()
                total_loss = torch.zeros((), device=device)
                total_kl = 0.0
                for g_ids, adv, old_lp in zip(gen_ids, advantages, old_logprobs):
                    new_lp = sequence_logprob(backbone.llm, prompt_ids_ref, g_ids, device, requires_grad=True)
                    ratio = torch.exp(new_lp - old_lp)
                    adv_t = torch.tensor(float(adv), device=device)
                    s1 = ratio * adv_t
                    s2 = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_t
                    total_loss = total_loss - torch.min(s1, s2)
                    total_kl += float(old_lp - new_lp.detach())
                total_loss = total_loss / len(gen_ids)
                mean_kl = total_kl / len(gen_ids)

                if mean_kl <= args.max_kl:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                    optimizer.step()
                else:
                    log.info("step=%d skipped update: KL=%.4f exceeds max_kl=%.4f",
                             global_step, mean_kl, args.max_kl)

                global_step += 1
                pbar.set_postfix(reward_mean=f"{mean_r:.3f}", kl=f"{mean_kl:.4f}", loss=f"{total_loss.detach().item():.4f}",
                                  kl_ema=f"{kl_ema:.2f}", kl_coef=f"{kl_controller.value:.4f}")

                if global_step % args.save_every == 0:
                    save_checkpoint(str(output_dir / "latest_ckpt"), backbone, optimizer, global_step)
                    log.info("step=%d checkpoint saved to %s/latest_ckpt", global_step, output_dir)

    save_checkpoint(str(output_dir / "latest_ckpt"), backbone, optimizer, global_step)
    save_checkpoint(str(output_dir / "best_ckpt"), backbone, optimizer, global_step)  # single-run, no eval-based selection yet
    log.info("Training complete. Final checkpoint: %s/latest_ckpt (also copied to best_ckpt)", output_dir)


if __name__ == "__main__":
    main()
