"""
train.py — ReMemorize full training script.

Two modes:
  --dev    Skip LLM loading; use synthetic placeholder data + HeuristicRewardScorer.
           Exercises the entire GRPO + EW-RLS pipeline in seconds on CPU.
  (default) Full training with a HuggingFace LLM (BF16 + Flash-Attention-2),
           NLIRewardScorer, and real MIMIC / MTS-Dialog data.

Example (dev mode, no GPU required):
    python train.py --dev --num_epochs 2 --group_size 4

Example (full run):
    python train.py \
        --model mistralai/Mistral-7B-v0.1 \
        --dataset mimic \
        --train_file data/mimic_train.jsonl \
        --eval_file  data/mimic_eval.jsonl  \
        --num_epochs 5 --group_size 8 \
        --output_dir runs/memorize_v1
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

from dataset import (
    MIMICDataset,
    MTSDialogDataset,
    ReMemorizeDataset,
    SummarizationExample,
    build_dataloader,
)
from memory_manager import ReMemorizeMemoryManager
from reward_manager import RewardCalibrator, RewardWeights
from scorers import HeuristicRewardScorer, NLIRewardScorer
from trainer import ReMemorizeTrainer, TrainerConfig
from policy_manager import UpdateStats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("memorize")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ReMemorize")

    # Mode
    p.add_argument("--dev", action="store_true",
                   help="Dev mode: synthetic data + heuristic scorer, no LLM.")

    # Model
    p.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1",
                   help="HuggingFace model id or local path.")
    p.add_argument("--freeze_llm", action="store_true",
                   help="Freeze LLM weights; only train memory interface.")
    p.add_argument("--load_in_4bit",  action="store_true",
                   help="Load LLM backbone in 4-bit NF4 quantization (required for 70B+ models).")
    p.add_argument("--no_flash_attn", action="store_true",
                   help="Disable Flash-Attention-2 (fallback to standard attention).")

    # Memory
    p.add_argument("--d_key",    type=int, default=64)
    p.add_argument("--d_value",  type=int, default=64)
    p.add_argument("--d_memory", type=int, default=128)

    # Dataset
    p.add_argument("--dataset",    type=str, default="mimic",
                   choices=["mimic", "mts_dialog"],
                   help="Which dataset to use (real files or placeholders in --dev).")
    p.add_argument("--train_file", type=str, default=None,
                   help="Path to training JSONL (required when not in --dev mode).")
    p.add_argument("--eval_file",  type=str, default=None,
                   help="Path to eval JSONL (optional).")
    p.add_argument("--max_train",  type=int, default=None,
                   help="Truncate training set to N examples.")
    p.add_argument("--max_eval_samples", type=int, default=None,
                   help="Cap eval set to N examples per checkpoint (speeds up eval on large val sets).")
    p.add_argument("--placeholder_n", type=int, default=200,
                   help="Number of synthetic examples in --dev mode.")

    # Training
    p.add_argument("--num_epochs",  type=int,   default=3)
    p.add_argument("--batch_size",  type=int,   default=1,
                   help="Documents per gradient step (keep at 1 for GRPO).")
    p.add_argument("--group_size",  type=int,   default=4,
                   help="G candidate summaries per document (GRPO group).")
    p.add_argument("--lr",          type=float, default=1e-4,
                   help="Learning rate for memory interface parameters.")
    p.add_argument("--ppo_lr",      type=float, default=1e-5,
                   help="Learning rate for the gating policy (PPO/GRPO).")
    p.add_argument("--max_new_tokens",     type=int, default=256)
    p.add_argument("--max_source_length", type=int, default=3840,
                   help="Max source tokens fed to LLM. 3840+256=4096 = Mistral context window.")
    p.add_argument("--max_target_length", type=int, default=256,
                   help="Max target tokens for teacher-forcing.")
    p.add_argument("--lm_loss_weight", type=float, default=0.5,
                   help="Weight of supervised CE loss relative to GRPO reward.")
    p.add_argument("--rl_lm_weight", type=float, default=0.5,
                   help="Weight of reward-weighted CE loss on best GRPO candidate. "
                        "Gives mem_proj/mem_gate a reward signal beyond supervised CE. "
                        "Set to 0 to disable.")
    p.add_argument("--warmup_steps",   type=int, default=100)
    p.add_argument("--grad_clip",      type=float, default=1.0)

    # Reward
    p.add_argument("--scorer", type=str, default="nli",
                   choices=["heuristic", "nli"],
                   help="Reward scorer. 'nli' requires transformers + sentence-transformers.")
    p.add_argument("--reward_alpha", type=float, default=1.0)
    p.add_argument("--reward_beta",  type=float, default=0.3)
    p.add_argument("--reward_gamma", type=float, default=0.1)
    p.add_argument("--reward_delta", type=float, default=0.8)

    # Training ablation (Group II — requires separate training runs)
    p.add_argument(
        "--training_ablation",
        type=str,
        default="full",
        choices=["full", "no_grpo", "no_rl", "no_hallucination_penalty", "no_aux_loss"],
        help=(
            "Training-time ablation variant. "
            "'no_grpo': replace GRPO with single-sample PPO. "
            "'no_rl': supervised CE only, no policy updates. "
            "'no_hallucination_penalty': set reward_delta=0 (remove -δ·hallucination term). "
            "'no_aux_loss': set mse_reward_weight=0 (remove memory MSE bonus)."
        ),
    )

    # Output / logging
    p.add_argument("--output_dir",    type=str, default="runs/memorize")
    p.add_argument("--save_every",    type=int, default=500,
                   help="Save latest_ckpt/ every N global steps (overwrites previous).")
    p.add_argument("--eval_every",    type=int, default=200,
                   help="Run evaluation every N global steps (0 = end of epoch only).")
    p.add_argument("--log_every",     type=int, default=10)
    p.add_argument("--wandb",         action="store_true",
                   help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", type=str, default="ReMemorize")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--resume_from",   type=str, default=None,
                   help="Path to a checkpoint directory (latest_ckpt/ or step_N/) "
                        "to resume training from. Restores model weights, optimizer, "
                        "scheduler, global_step, and best_rouge1.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1/2/L F1 scores."""
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        totals: Dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        n = max(len(predictions), 1)
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for k in totals:
                totals[k] += scores[k].fmeasure
        return {k: v / n for k, v in totals.items()}
    except ImportError:
        def _f1(pred: str, ref: str) -> float:
            p_toks = set(pred.lower().split())
            r_toks = set(ref.lower().split())
            if not p_toks or not r_toks:
                return 0.0
            inter = len(p_toks & r_toks)
            prec = inter / len(p_toks)
            rec  = inter / len(r_toks)
            return 2 * prec * rec / (prec + rec + 1e-9)
        scores = [_f1(p, r) for p, r in zip(predictions, references)]
        mean = sum(scores) / max(len(scores), 1)
        return {"rouge1": mean, "rouge2": 0.0, "rougeL": mean}


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BLEU-1 and BLEU-2 scores."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoother = SmoothingFunction().method1
        bleu1_total, bleu2_total = 0.0, 0.0
        n = max(len(predictions), 1)
        for pred, ref in zip(predictions, references):
            ref_toks  = [ref.lower().split()]
            pred_toks = pred.lower().split()
            bleu1_total += sentence_bleu(ref_toks, pred_toks, weights=(1, 0, 0, 0),    smoothing_function=smoother)
            bleu2_total += sentence_bleu(ref_toks, pred_toks, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
        return {"bleu1": bleu1_total / n, "bleu2": bleu2_total / n}
    except ImportError:
        log.warning("nltk not installed; skipping BLEU.")
        return {"bleu1": 0.0, "bleu2": 0.0}


def compute_bertscore(predictions: List[str], references: List[str], device: str = "cpu") -> Dict[str, float]:
    """Compute BERTScore F1."""
    try:
        from bert_score import score as bscore
        predictions = [p if p.strip() else "." for p in predictions]
        references  = [r if r.strip() else "." for r in references]
        _, _, F1 = bscore(
            predictions, references,
            model_type="bert-base-uncased",
            num_layers=9,
            device=device,
            verbose=False,
        )
        return {"bertscore_f1": float(F1.mean())}
    except ImportError:
        log.warning("bert_score not installed; skipping BERTScore.")
        return {"bertscore_f1": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Dev-mode generate_summary (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _dev_generate(source: str, rng, variation: int = 0) -> str:
    """
    Produce a lightweight synthetic summary for dev-mode GRPO.
    Picks a random subset of source tokens to mimic different candidates.
    """
    words = source.split()
    rng.shuffle(words)
    take = max(10, len(words) // (3 + variation % 4))
    return " ".join(words[:take]) + "."


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    backbone,             # ReMemorizeLLM | None in dev mode
    eval_dataset: ReMemorizeDataset,
    args: argparse.Namespace,
    step: int,
    wb_run=None,
    dev_rng=None,
    device: str = "cpu",
) -> Dict[str, float]:
    preds: List[str] = []
    refs:  List[str] = []

    eval_items = list(eval_dataset)
    if args.max_eval_samples is not None:
        eval_items = eval_items[:args.max_eval_samples]
    for ex in eval_items:
        ref = ex.reference or ex.target
        if backbone is None:
            # dev mode: cheap synthetic prediction
            pred = _dev_generate(ex.source, dev_rng, variation=0)
        else:
            backbone.encode_and_update_memory(ex.source)
            pred = backbone.generate_summary(ex.source, dataset_type=args.dataset, max_new_tokens=args.max_new_tokens)
        preds.append(pred)
        refs.append(ref)

    scores: Dict[str, float] = {}
    scores.update(compute_rouge(preds, refs))
    scores.update(compute_bleu(preds, refs))
    scores.update(compute_bertscore(preds, refs, device=device))

    log.info(
        "eval step=%d  R1=%.4f  R2=%.4f  RL=%.4f  BLEU1=%.4f  BLEU2=%.4f  BERTScore-F1=%.4f",
        step,
        scores["rouge1"], scores["rouge2"], scores["rougeL"],
        scores["bleu1"],  scores["bleu2"],
        scores["bertscore_f1"],
    )
    if wb_run is not None:
        wb_run.log({"eval/" + k: v for k, v in scores.items()}, step=step)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── WandB ──────────────────────────────────────────────────────────────
    wb_run = None
    if args.wandb:
        try:
            import wandb
            wb_run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"memorize_{int(time.time())}",
            )
        except ImportError:
            log.warning("wandb not installed; skipping W&B logging.")

    # ── Memory manager ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)

    mm = ReMemorizeMemoryManager(
        d_key=args.d_key,
        d_value=args.d_value,
        d_memory=args.d_memory,
        seed=args.seed,
        device=device,
    )

    # ── Apply training ablation overrides ───────────────────────────────────
    abl = args.training_ablation
    if abl == "no_hallucination_penalty":
        args.reward_delta = 0.0
        log.info("Ablation 'no_hallucination_penalty': reward_delta set to 0.")
    mse_reward_weight = 0.0 if abl == "no_aux_loss" else 0.1
    if abl == "no_aux_loss":
        log.info("Ablation 'no_aux_loss': mse_reward_weight set to 0.")

    # ── Reward calibrator + weights ─────────────────────────────────────────
    calibrator = RewardCalibrator(
        ema_alpha=0.1, clip_z=2.5, factual_floor=0.5, factual_floor_penalty=0.2
    )
    reward_weights = RewardWeights(
        alpha=args.reward_alpha,
        beta=args.reward_beta,
        gamma=args.reward_gamma,
        delta=args.reward_delta,
    )

    # ── Reward scorer ───────────────────────────────────────────────────────
    if args.scorer == "nli" and not args.dev:
        log.info("Loading NLI reward scorer (DeBERTa-v3 + MiniLM)...")
        scorer = NLIRewardScorer(device=device)
    else:
        log.info("Using heuristic reward scorer.")
        scorer = HeuristicRewardScorer()

    # ── Trainer (policy + reward) ───────────────────────────────────────────
    trainer = ReMemorizeTrainer(
        memory_manager=mm,
        reward_calibrator=calibrator,
        reward_scorer=scorer,
        reward_weights=reward_weights,
        config=TrainerConfig(
            ppo_lr=args.ppo_lr,
            clip_eps=0.2,
            entropy_coef=0.005,
            group_size=args.group_size,
            max_kl=0.02,
            mse_reward_weight=mse_reward_weight,
            lr_warmup_steps=args.warmup_steps,
        ),
        seed=args.seed,
    )

    # ── LLM backbone (skipped in dev mode) ─────────────────────────────────
    backbone = None
    backbone_optimizer = None
    backbone_scheduler = None

    if not args.dev:
        log.info("Loading LLM backbone: %s", args.model)
        from llm_backbone import ReMemorizeLLM
        backbone = ReMemorizeLLM(
            model_name_or_path=args.model,
            memory_manager=mm,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            dtype=torch.bfloat16,
            gradient_checkpointing=True,
            use_flash_attention=not args.no_flash_attn,
            freeze_llm=args.freeze_llm,
            load_in_4bit=args.load_in_4bit,
        )
        backbone_optimizer = AdamW(
            backbone.memory_interface_parameters(),
            lr=args.lr,
            weight_decay=1e-2,
        )
        if not args.freeze_llm:
            backbone_optimizer.add_param_group(
                {"params": backbone.llm.parameters(), "lr": args.lr * 0.1}
            )
        backbone_scheduler = LinearLR(
            backbone_optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=args.warmup_steps,
        )

    # ── Dataset ─────────────────────────────────────────────────────────────
    dev_rng = __import__("random").Random(args.seed)

    if args.dev:
        DatasetCls = MIMICDataset if args.dataset == "mimic" else MTSDialogDataset
        train_dataset = DatasetCls.placeholder(n=args.placeholder_n, seed=args.seed)
        eval_dataset: Optional[ReMemorizeDataset] = DatasetCls.placeholder(n=20, seed=args.seed + 1)
        log.info("Dev mode: %d synthetic train / 20 eval examples.", args.placeholder_n)
    else:
        if args.dataset == "mimic":
            if args.train_file is None:
                raise ValueError("--train_file is required for the mimic dataset.")
            train_dataset = MIMICDataset.from_jsonl(args.train_file, max_examples=args.max_train)
            eval_dataset = MIMICDataset.from_jsonl(args.eval_file) if args.eval_file else None
        else:
            if args.train_file:
                train_dataset = MTSDialogDataset.from_jsonl(args.train_file, max_examples=args.max_train)
                eval_dataset = MTSDialogDataset.from_jsonl(args.eval_file) if args.eval_file else None
            else:
                log.info("No --train_file given for mts_dialog; loading from HuggingFace (SubashNeupane/dataset_SOAP_summary).")
                train_dataset = MTSDialogDataset.from_huggingface("train", seed=args.seed, max_examples=args.max_train)
                eval_dataset  = MTSDialogDataset.from_huggingface("val",   seed=args.seed)
        log.info(
            "Loaded %d train examples%s.",
            len(train_dataset),
            f" / {len(eval_dataset)} eval" if eval_dataset else "",
        )

    dataloader = build_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed
    )

    # ── Resume from checkpoint ───────────────────────────────────────────────
    global_step = 0
    best_rouge1 = 0.0
    resume_epoch = 1
    resume_skip  = 0   # steps to skip at the start of resume_epoch

    if args.resume_from:
        log.info("Resuming from checkpoint: %s", args.resume_from)
        ckpt_path = Path(args.resume_from)

        # Restore memory interface weights into already-constructed backbone
        if backbone is not None:
            mi = torch.load(ckpt_path / "memory_interface.pt", map_location="cpu")
            backbone.key_proj.load_state_dict(mi["key_proj"])
            backbone.val_proj.load_state_dict(mi["val_proj"])
            backbone.mem_proj.load_state_dict(mi["mem_proj"])
            backbone.mem_gate.load_state_dict(mi["mem_gate"])
            backbone.memory_manager.load_state_dict(mi["memory_manager"])

        # Restore scalar state and optimizer/scheduler
        ts_path = ckpt_path / "training_state.pt"
        if ts_path.exists():
            ts = torch.load(ts_path, map_location="cpu")
            global_step = ts.get("global_step", 0)
            best_rouge1 = ts.get("best_rouge1", 0.0)
            if backbone_optimizer is not None and "optimizer" in ts:
                backbone_optimizer.load_state_dict(ts["optimizer"])
            if backbone_scheduler is not None and "scheduler" in ts:
                backbone_scheduler.load_state_dict(ts["scheduler"])
            if backbone is not None and "policy_optimizer" in ts:
                backbone.memory_manager.policy.optimizer.load_state_dict(ts["policy_optimizer"])

        steps_per_epoch = len(train_dataset)
        resume_epoch = global_step // steps_per_epoch + 1
        resume_skip  = global_step % steps_per_epoch
        log.info(
            "Resumed at global_step=%d  best_rouge1=%.4f  "
            "continuing from epoch=%d skip=%d steps",
            global_step, best_rouge1, resume_epoch, resume_skip,
        )

    # ── SIGTERM handler — save latest_ckpt/ before SLURM kills the job ───────
    def _sigterm_handler(signum, frame):
        log.warning("SIGTERM received — saving emergency checkpoint before exit.")
        if backbone is not None:
            backbone.save_checkpoint(
                str(output_dir / "latest_ckpt"),
                optimizer=backbone_optimizer,
                scheduler=backbone_scheduler,
                global_step=global_step,
                best_rouge1=best_rouge1,
            )
            log.info("Emergency checkpoint saved to %s/latest_ckpt", output_dir)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # ── Training ─────────────────────────────────────────────────────────────
    for epoch in range(1, args.num_epochs + 1):
        # Skip epochs already completed when resuming
        if epoch < resume_epoch:
            log.info("Skipping epoch %d (already completed).", epoch)
            continue

        log.info("═══ Epoch %d / %d ═══", epoch, args.num_epochs)
        epoch_grpo_loss = 0.0
        epoch_lm_loss   = 0.0
        epoch_mse       = 0.0
        epoch_steps     = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.num_epochs}", unit="doc")
        for batch_idx, batch in enumerate(pbar):
            # Skip steps within the resume epoch already processed
            if epoch == resume_epoch and batch_idx < resume_skip:
                continue
            for ex in batch:
                assert isinstance(ex, SummarizationExample)
                source    = ex.source
                reference = ex.reference or ex.target

                # ── 1. Encode source into memory / extract key_window ──────
                if backbone is not None:
                    key_window, _ = backbone.encode_and_update_memory(source)
                    # FIX: save Phase 1 snapshot so every candidate starts
                    # from the same memory state (independent rollouts).
                    phase1_snapshot = backbone.memory_manager.m.clone()
                else:
                    # Dev mode: use random vectors as key proxies
                    T = min(len(source.split()), 64)
                    key_window = [
                        [dev_rng.gauss(0, 1) for _ in range(args.d_key)]
                        for _ in range(T)
                    ]
                    mm.reset_state()
                    mm.update(key_window, key_window)
                    phase1_snapshot = None

                # ── 2. Generate candidate summaries ────────────────────────
                # no_grpo: one candidate → PPO update (no group normalisation)
                # no_rl:   skip candidate generation and policy update entirely
                n_candidates = 1 if abl == "no_grpo" else args.group_size
                grpo_stats = None
                candidates = []

                if abl != "no_rl":
                    if backbone is not None:
                        for _ in range(n_candidates):
                            backbone.memory_manager.m.copy_(phase1_snapshot)
                            candidates.append(
                                backbone.generate_summary(
                                    source,
                                    dataset_type=args.dataset,
                                    max_new_tokens=args.max_new_tokens,
                                    temperature=0.3,
                                )
                            )
                        backbone.memory_manager.m.copy_(phase1_snapshot)
                    else:
                        candidates = [
                            _dev_generate(source, dev_rng, variation=g)
                            for g in range(n_candidates)
                        ]

                # ── 3. Policy update (GRPO or PPO) ─────────────────────────
                if abl == "no_rl":
                    grpo_stats = UpdateStats(0.0, 0.0, 0.0, 0.0, 0.0)
                elif abl == "no_grpo":
                    # Single-sample PPO (no group advantage normalisation)
                    ppo_batch = trainer.collect_policy_batch_from_summaries(
                        key_windows=[key_window],
                        sources=[source],
                        summaries=candidates,
                        references=[reference],
                    )
                    grpo_stats = trainer.ppo_step(ppo_batch)
                else:
                    grpo_batch = trainer.collect_grpo_batch(
                        key_windows=[key_window],
                        sources=[source],
                        summary_groups=[candidates],
                        references=[reference],
                    )
                    grpo_stats = trainer.grpo_step(grpo_batch)

                # ── 4. Supervised CE loss on gold target (backbone only) ────
                # Also runs reward-weighted CE on best GRPO candidate so that
                # mem_proj/mem_gate receive a reward signal, not only supervised
                # CE. (forward_train resets memory internally via Phase 1.)
                lm_loss_val = 0.0
                mse_val     = 0.0
                lm_info: Optional[dict] = None
                if backbone is not None and backbone_optimizer is not None:
                    backbone_optimizer.zero_grad()
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16,
                                            enabled=(device == "cuda")):
                        # Gold-target CE loss
                        lm_loss, lm_info = backbone.forward_train(source, ex.target)
                        weighted_loss = args.lm_loss_weight * lm_loss

                        # Reward-weighted CE on best candidate
                        # Skipped for no_rl / no_grpo (no rewards computed)
                        if args.rl_lm_weight > 0.0 and candidates and abl not in ("no_rl", "no_grpo"):
                            rewards  = [s.reward for s in grpo_batch]  # grpo_batch defined in full branch
                            mean_r   = sum(rewards) / len(rewards)
                            std_r    = max(
                                (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5,
                                1e-8,
                            )
                            best_idx = rewards.index(max(rewards))
                            best_adv = (rewards[best_idx] - mean_r) / std_r
                            if best_adv > 0.0:
                                rl_lm_loss, _ = backbone.forward_train(
                                    source, candidates[best_idx]
                                )
                                weighted_loss = weighted_loss + (
                                    args.lm_loss_weight * args.rl_lm_weight
                                    * best_adv * rl_lm_loss
                                )

                    weighted_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        backbone.memory_interface_parameters(), args.grad_clip
                    )
                    backbone_optimizer.step()
                    if backbone_scheduler is not None:
                        backbone_scheduler.step()
                    lm_loss_val = lm_loss.item()
                    mse_val     = lm_info.get("mse", 0.0)

                # ── 5. Accumulate stats ────────────────────────────────────
                epoch_grpo_loss += grpo_stats.loss
                epoch_lm_loss   += lm_loss_val
                epoch_mse       += mse_val
                epoch_steps     += 1
                global_step     += 1

                # ── 6. Logging + safeguard monitoring ──────────────────────
                gamma_mean  = lm_info.get("gamma_mean",  0.0) if lm_info else 0.0
                gamma_std   = lm_info.get("gamma_std",   0.0) if lm_info else 0.0
                memory_norm = lm_info.get("memory_norm", 0.0) if lm_info else 0.0
                ce_loss_val = lm_info.get("ce_loss",     lm_loss_val) if lm_info else lm_loss_val

                # Safeguard 4: warn on collapse or explosion
                if backbone is not None and lm_info:
                    if gamma_mean > 0.8:
                        log.warning(
                            "step=%d: gamma_mean=%.3f > 0.8 — risk of memory overwriting. "
                            "Consider reducing ppo_lr or increasing detach_memory_hidden.",
                            global_step, gamma_mean,
                        )
                    if memory_norm > 50.0:
                        log.warning(
                            "step=%d: memory_norm=%.2f > 50 — potential explosion. "
                            "Check grad_clip and write_proj initialisation.",
                            global_step, memory_norm,
                        )

                pbar.set_postfix(
                    step=global_step,
                    grpo=f"{grpo_stats.loss:.4f}",
                    R1=f"{best_rouge1:.4f}",
                    γ=f"{gamma_mean:.3f}",
                )

                if global_step % args.log_every == 0:
                    log.info(
                        "step=%d  grpo=%.4f  ce=%.4f  lm=%.4f  "
                        "γ_mean=%.3f  γ_std=%.3f  mem_norm=%.2f  mse=%.4f",
                        global_step,
                        grpo_stats.loss,
                        ce_loss_val,
                        lm_loss_val,
                        gamma_mean,
                        gamma_std,
                        memory_norm,
                        mse_val,
                    )
                    if wb_run is not None:
                        wb_run.log(
                            {
                                "train/grpo_loss":   grpo_stats.loss,
                                "train/ce_loss":     ce_loss_val,
                                "train/lm_loss":     lm_loss_val,
                                "train/surrogate":   grpo_stats.surrogate,
                                "train/entropy":     grpo_stats.entropy,
                                "train/approx_kl":   grpo_stats.approx_kl,
                                "train/clip_frac":   grpo_stats.clip_fraction,
                                "train/gamma_mean":  gamma_mean,
                                "train/gamma_std":   gamma_std,
                                "train/memory_norm": memory_norm,
                                "train/mse":         mse_val,
                            },
                            step=global_step,
                        )

                # ── 7. Periodic evaluation ─────────────────────────────────
                if (
                    args.eval_every > 0
                    and global_step % args.eval_every == 0
                    and eval_dataset is not None
                ):
                    scores = evaluate(backbone, eval_dataset, args, global_step, wb_run, dev_rng, device=device)
                    if scores["rouge1"] > best_rouge1:
                        best_rouge1 = scores["rouge1"]
                        if backbone is not None:
                            backbone.save_checkpoint(
                                str(output_dir / "best_ckpt"),
                                optimizer=backbone_optimizer,
                                scheduler=backbone_scheduler,
                                global_step=global_step,
                                best_rouge1=best_rouge1,
                            )
                            log.info("New best checkpoint saved (R1=%.4f).", best_rouge1)

                # ── 8. Periodic checkpoint (overwrites latest_ckpt/) ──────────
                if global_step % args.save_every == 0 and backbone is not None:
                    backbone.save_checkpoint(
                        str(output_dir / "latest_ckpt"),
                        optimizer=backbone_optimizer,
                        scheduler=backbone_scheduler,
                        global_step=global_step,
                        best_rouge1=best_rouge1,
                    )
                    log.info("latest_ckpt saved at step %d", global_step)

        pbar.close()

        # ── End-of-epoch summary ───────────────────────────────────────────
        n = max(epoch_steps, 1)
        log.info(
            "Epoch %d done | avg_grpo_loss=%.4f  avg_lm_loss=%.4f  avg_mse=%.6f",
            epoch,
            epoch_grpo_loss / n,
            epoch_lm_loss / n,
            epoch_mse / n,
        )

        # End-of-epoch evaluation (if not already done mid-epoch)
        if eval_dataset is not None and (args.eval_every == 0 or epoch_steps < args.eval_every):
            scores = evaluate(backbone, eval_dataset, args, global_step, wb_run, dev_rng, device=device)
            if backbone is not None and scores["rouge1"] > best_rouge1:
                best_rouge1 = scores["rouge1"]
                backbone.save_checkpoint(
                    str(output_dir / "best_ckpt"),
                    optimizer=backbone_optimizer,
                    scheduler=backbone_scheduler,
                    global_step=global_step,
                    best_rouge1=best_rouge1,
                )

    # ── Final checkpoint ──────────────────────────────────────────────────
    if backbone is not None:
        backbone.save_checkpoint(
            str(output_dir / "final_ckpt"),
            optimizer=backbone_optimizer,
            scheduler=backbone_scheduler,
            global_step=global_step,
            best_rouge1=best_rouge1,
        )
        log.info("Final checkpoint saved.")

    if wb_run is not None:
        wb_run.finish()

    log.info("Training complete. Best ROUGE-1: %.4f", best_rouge1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
