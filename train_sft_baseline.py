"""
SFT baseline: a plain supervised-fine-tuning baseline -- the backbone LLM
fine-tuned directly on (source, target) pairs with standard teacher-forced
cross-entropy, no memory module and no reward/RL signal of any kind.

This is deliberately the simplest possible training objective in the project.
Unlike train.py's Lmem (Eq. training_obj, which mixes gold CE with a
reward-weighted candidate CE and an auxiliary memory-reconstruction term) and
unlike train_dpo_baseline.py's GRPO-direct (policy-gradient RL on the LLM's
own generation, no memory), this script never touches the memory interface at
all -- not through injection, not through the aux-loss shadow path, not
through ablation_mode. Training calls backbone.llm(input_ids=..., labels=...)
directly and nothing else, so "no memory" is categorical rather than a flag.

Same backbone, same LoRA convention, same data, and the same downstream eval
path as the DPO/GRPO-direct baseline, so all three baselines are comparable:
  - ReMemorizeLLM is constructed only for its tokenizer and generate_summary()
    (used at eval time, not here); a throwaway ReMemorizeMemoryManager is
    passed only because the constructor requires one -- never trained, never
    read, never updated.
  - LoRA-wraps backbone.llm exactly like train_dpo_baseline.py; only the LoRA
    adapter weights are trainable.
  - Checkpoints are saved as a plain PEFT adapter dir (backbone.llm.save_pretrained),
    the same format train_dpo_baseline.py produces -- so the existing
    evaluate_dpo_baseline.py (which already generically loads any LoRA
    checkpoint via PeftModel.from_pretrained + ablation_mode="no_memory") can
    score this baseline unchanged. No new eval script needed.

Usage:
    python train_sft_baseline.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset mimic \
        --train_file Datasets/mimic_5k_train.jsonl \
        --max_train 1000 --max_source_length 2048 --max_new_tokens 280 \
        --output_dir runs/sft_baseline/mimic

Eval (reusing the existing script, unchanged):
    python evaluate_dpo_baseline.py \
        --checkpoint runs/sft_baseline/mimic/latest_ckpt \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --eval_file Datasets/mimic_5k_test.jsonl \
        --output_dir results/baselines/llama31_8b_sft_mimic
"""
from __future__ import annotations

import argparse
import logging
import signal
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from dataset import MIMICDataset, MTSDialogDataset, ReMemorizeDataset, SummarizationExample, build_dataloader
from llm_backbone import ReMemorizeLLM
from memory_manager import ReMemorizeMemoryManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sft_baseline")


def compute_ce_loss(backbone: ReMemorizeLLM, source: str, target: str) -> torch.Tensor:
    """
    Plain teacher-forced causal-LM cross-entropy on [source | target], masking
    source tokens out of the loss. Calls backbone.llm directly -- no memory
    encoding, no injection, no memory update, no auxiliary loss. This is the
    entire training objective.

    Tokenization/masking and the manual float() cast before cross_entropy
    mirror ReMemorizeLLM.forward_train (llm_backbone.py) exactly, including
    its use of the live backbone._llm_device property rather than a passed-in
    device string, and its explicit upcast to float32 before the loss (guards
    against bf16/fp16 loss-computation instability -- deliberate in the
    existing code, not skipped here).
    """
    src_ids = backbone.tokenizer(
        source, truncation=True, max_length=backbone.max_source_length
    )["input_ids"]
    tgt_ids = (
        backbone.tokenizer(target, truncation=True, max_length=backbone.max_target_length)["input_ids"]
        + [backbone.tokenizer.eos_token_id]
    )
    max_total = backbone.max_source_length + backbone.max_target_length
    full_ids = (src_ids + tgt_ids)[:max_total]
    labels = ([-100] * len(src_ids) + tgt_ids)[: len(full_ids)]

    input_ids = torch.tensor([full_ids], device=backbone._llm_device)
    label_ids = torch.tensor([labels], device=backbone._llm_device)

    logits = backbone.llm(input_ids=input_ids).logits  # (1, T, vocab)
    shift_logits = logits[0, :-1].contiguous()
    shift_labels = label_ids[0, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.float(), shift_labels, ignore_index=-100
    )


def save_checkpoint(save_dir: str, backbone: ReMemorizeLLM, optimizer, global_step: int) -> None:
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    backbone.llm.save_pretrained(str(path))  # peft: saves only the LoRA adapter weights
    torch.save(
        {"optimizer": optimizer.state_dict(), "global_step": global_step},
        path / "training_state.pt",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plain SFT baseline: LoRA fine-tuning, no memory, no RL.")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--dataset", type=str, default="mimic", choices=["mimic", "mts_dialog"])
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--max_train", type=int, default=None)

    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_source_length", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=280,
                    help="Used as max_target_length for tokenization (no generation happens during training).")

    p.add_argument("--no_flash_attn", action="store_true")

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05,
                    help="Safe here (unlike train_dpo_baseline.py): plain CE loss has no "
                         "old-vs-new log-prob comparison for dropout's stochasticity to break.")
    p.add_argument("--lora_target_modules", type=str,
                    default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    p.add_argument("--lr", type=float, default=2e-4,
                    help="Standard LoRA SFT learning rate (much higher than the DPO "
                         "baseline's 1e-6 -- plain CE loss has no importance-ratio to destabilize).")
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--save_every", type=int, default=250)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Throwaway memory manager: ReMemorizeLLM's constructor requires one, but
    # compute_ce_loss() above never calls anything that touches it.
    inert_mm = ReMemorizeMemoryManager(d_key=64, d_value=64, d_memory=128, seed=args.seed, device=device)

    backbone = ReMemorizeLLM(
        args.model,
        inert_mm,
        max_source_length=args.max_source_length,
        max_target_length=args.max_new_tokens,
        use_flash_attention=not args.no_flash_attn,
        freeze_llm=False,  # superseded by LoRA wrapping below
    )

    from peft import LoraConfig, get_peft_model, PeftModel
    for p_ in backbone.llm.parameters():
        p_.requires_grad_(False)

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

    trainable_params = [p_ for p_ in backbone.llm.parameters() if p_.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)

    if args.resume_from:
        state_path = Path(args.resume_from) / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device)
            optimizer.load_state_dict(state["optimizer"])
            global_step = state["global_step"]
            log.info("Resumed at global_step=%d", global_step)

    DatasetCls = MIMICDataset if args.dataset == "mimic" else MTSDialogDataset
    train_dataset: ReMemorizeDataset = DatasetCls.from_jsonl(args.train_file, max_examples=args.max_train)
    log.info("Loaded %d train examples.", len(train_dataset))
    dataloader = build_dataloader(train_dataset, batch_size=1, shuffle=True, seed=args.seed)

    def _sigterm_handler(signum, frame):
        log.warning("SIGTERM received -- saving emergency checkpoint before exit.")
        save_checkpoint(str(output_dir / "latest_ckpt"), backbone, optimizer, global_step)
        log.info("Emergency checkpoint saved to %s/latest_ckpt", output_dir)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    resume_skip = global_step
    backbone.llm.train()

    for epoch in range(1, args.num_epochs + 1):
        log.info("=== Epoch %d / %d ===", epoch, args.num_epochs)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.num_epochs}", unit="doc")
        for batch_idx, batch in enumerate(pbar):
            if epoch == 1 and batch_idx < resume_skip:
                continue

            for ex in batch:
                assert isinstance(ex, SummarizationExample)

                optimizer.zero_grad()
                loss = compute_ce_loss(backbone, ex.source, ex.target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()

                global_step += 1
                if global_step % args.log_every == 0:
                    pbar.set_postfix(loss=f"{loss.detach().item():.4f}")

                if global_step % args.save_every == 0:
                    save_checkpoint(str(output_dir / "latest_ckpt"), backbone, optimizer, global_step)
                    log.info("step=%d checkpoint saved to %s/latest_ckpt", global_step, output_dir)

    save_checkpoint(str(output_dir / "latest_ckpt"), backbone, optimizer, global_step)
    log.info("Training complete. Final checkpoint: %s/latest_ckpt", output_dir)


if __name__ == "__main__":
    main()
