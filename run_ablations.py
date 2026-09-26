"""
run_ablations.py — Run ablation study across all ReMemorize memory variants.

Runs evaluate.py for each ablation mode on the same checkpoint and eval set,
then prints a side-by-side comparison table for paper reporting.

Ablation variants
-----------------
  full              Full model: adaptive gating + Phase 2 memory updates
  no_memory         LLM baseline: no memory injection, no updates
  fixed_gate        Inject memory but fix γ=0.5 (no learned gating)
  no_phase2_update  Inject Phase 1 memory but freeze during decoding

Usage
-----
python run_ablations.py \\
    --checkpoint  runs/mimic_v1/best_ckpt \\
    --eval_file   Datasets/mimic_eval.jsonl \\
    --output_dir  results/ablation_study \\
    --model       mistralai/Mistral-7B-v0.1 \\
    --max_examples 100

# Limit to specific ablations:
python run_ablations.py \\
    --checkpoint runs/mimic_v1/best_ckpt \\
    --eval_file  Datasets/mimic_eval.jsonl \\
    --output_dir results/ablation_study \\
    --model      mistralai/Mistral-7B-v0.1 \\
    --ablations  full no_memory fixed_gate

# Skip inference, just reprint the comparison table from existing results:
python run_ablations.py \\
    --output_dir results/ablation_study \\
    --compare_only
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

ALL_ABLATIONS = [
    # Group I: memory architecture (inference-time, single checkpoint)
    "full",
    "no_memory",
    "fixed_gate",
    "no_phase2_update",
    # Group II: training recipe (require separate training runs)
    "no_grpo",
    "no_rl",
    "no_hallucination_penalty",
    "no_aux_loss",
]

GROUP1_ABLATIONS = {"full", "no_memory", "fixed_gate", "no_phase2_update"}
GROUP2_ABLATIONS = {"no_grpo", "no_rl", "no_hallucination_penalty", "no_aux_loss"}

ABLATION_LABELS = {
    "full":                    "Full model (ours)",
    "no_memory":               "No memory (LLM baseline)",
    "fixed_gate":              "Fixed gate γ=0.5",
    "no_phase2_update":        "No Phase-2 update",
    "no_grpo":                 "w/o GRPO (PPO)",
    "no_rl":                   "w/o RL (CE only)",
    "no_hallucination_penalty": "w/o Halluc. penalty",
    "no_aux_loss":             "w/o Aux. loss",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ReMemorize ablation study runner")

    p.add_argument("--checkpoint",        type=str, default=None,
                   help="Path to checkpoint directory (Group I ablations).")
    p.add_argument("--eval_file",         type=str, default=None,
                   help="Path to eval JSONL.")
    p.add_argument("--train_file",        type=str, default=None,
                   help="Path to training JSONL (required for Group II ablations).")
    p.add_argument("--output_dir",        type=str, required=True,
                   help="Root output dir; each ablation gets a subdirectory.")
    p.add_argument("--model",             type=str, default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--dataset",           type=str, default="mimic",
                   choices=["mimic", "mts_dialog"])
    # Group II training args (forwarded to train.py)
    p.add_argument("--num_epochs",        type=int, default=3)
    p.add_argument("--group_size",        type=int, default=4)
    p.add_argument("--batch_size",        type=int, default=1)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--max_new_tokens",    type=int, default=256)
    p.add_argument("--max_source_length", type=int, default=3840)
    p.add_argument("--no_flash_attn",     action="store_true")
    p.add_argument("--max_examples",      type=int, default=None,
                   help="Cap number of eval examples (useful for quick tests).")
    p.add_argument("--ablations",         nargs="+", default=list(GROUP1_ABLATIONS),
                   choices=ALL_ABLATIONS,
                   help=(
                       "Which ablation variants to run. "
                       "Default: Group I only (inference-time). "
                       "Group II variants require --train_file."
                   ))
    p.add_argument("--compare_only",      action="store_true",
                   help="Skip inference; just load existing metrics.json files and compare.")
    p.add_argument("--device",            type=str, default=None)

    return p.parse_args()


def run_one_ablation(
    ablation: str,
    args: argparse.Namespace,
    ablation_output_dir: Path,
) -> Optional[Dict]:
    """
    Call evaluate.py as a subprocess for a single ablation mode.
    Returns the metrics dict on success, None on failure.
    """
    cmd = [
        sys.executable, "evaluate.py",
        "--checkpoint",        str(args.checkpoint),
        "--eval_file",         str(args.eval_file),
        "--output_dir",        str(ablation_output_dir),
        "--model",             args.model,
        "--dataset",           args.dataset,
        "--max_new_tokens",    str(args.max_new_tokens),
        "--max_source_length", str(args.max_source_length),
        "--ablation",          ablation,
        "--run_name",          ablation,
        "--notes",             f"ablation={ablation}",
    ]
    if args.no_flash_attn:
        cmd.append("--no_flash_attn")
    if args.max_examples:
        cmd += ["--max_examples", str(args.max_examples)]
    if args.device:
        cmd += ["--device", args.device]

    log.info("─── Running ablation: %s ───", ablation)
    log.info("Command: %s", " ".join(cmd))

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        log.error("Ablation %s FAILED (exit code %d)", ablation, result.returncode)
        return None

    metrics_path = ablation_output_dir / "metrics.json"
    if not metrics_path.exists():
        log.error("metrics.json not found at %s", metrics_path)
        return None

    with open(metrics_path) as f:
        return json.load(f)


def run_training_ablation(
    ablation: str,
    args: argparse.Namespace,
    ablation_output_dir: Path,
) -> Optional[Dict]:
    """
    For Group II ablations: run train.py with --training_ablation, then
    evaluate the resulting best_ckpt with evaluate.py.
    Returns the metrics dict on success, None on failure.
    """
    if not args.train_file:
        log.error(
            "Ablation '%s' is a Group II (training-recipe) variant and requires --train_file.",
            ablation,
        )
        return None

    train_output_dir = ablation_output_dir / "train_run"

    # ── Step 1: train ───────────────────────────────────────────────────────
    train_cmd = [
        sys.executable, "train.py",
        "--training_ablation", ablation,
        "--model",             args.model,
        "--dataset",           args.dataset,
        "--train_file",        str(args.train_file),
        "--output_dir",        str(train_output_dir),
        "--num_epochs",        str(args.num_epochs),
        "--group_size",        str(args.group_size),
        "--batch_size",        str(args.batch_size),
        "--lr",                str(args.lr),
        "--seed",              str(args.seed),
        "--max_new_tokens",    str(args.max_new_tokens),
        "--max_source_length", str(args.max_source_length),
    ]
    if args.eval_file:
        train_cmd += ["--eval_file", str(args.eval_file)]
    if args.no_flash_attn:
        train_cmd.append("--no_flash_attn")
    if args.max_examples:
        train_cmd += ["--max_train", str(args.max_examples)]

    log.info("─── Training ablation: %s ───", ablation)
    log.info("Command: %s", " ".join(train_cmd))
    result = subprocess.run(train_cmd, cwd=str(Path(__file__).parent))
    if result.returncode != 0:
        log.error("Training ablation %s FAILED (exit code %d)", ablation, result.returncode)
        return None

    best_ckpt = train_output_dir / "best_ckpt"
    if not best_ckpt.exists():
        log.error("best_ckpt not found at %s after training.", best_ckpt)
        return None

    # ── Step 2: evaluate ────────────────────────────────────────────────────
    return run_one_ablation(
        ablation="full",   # evaluate in full inference mode (ablation already baked in)
        args=argparse.Namespace(
            **{**vars(args), "checkpoint": str(best_ckpt)},
        ),
        ablation_output_dir=ablation_output_dir,
    )


def load_existing_metrics(ablation: str, output_dir: Path) -> Optional[Dict]:
    metrics_path = output_dir / ablation / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        return json.load(f)


def print_comparison_table(
    results: Dict[str, Optional[Dict]],
    ablations: List[str],
) -> None:
    """Print a LaTeX-friendly comparison table to stdout."""
    # Metrics scaled ×100 for percentage display (standard in NLP papers)
    # mem_* diagnostics are kept as raw floats
    PCT_METRICS = {
        "rouge1", "rouge2", "rougeL",
        "bleu1", "bleu2",
        "bertscore_f1", "minicheck_support",
    }

    metrics_to_show = [
        ("rouge1",            "ROUGE-1"),
        ("rouge2",            "ROUGE-2"),
        ("rougeL",            "ROUGE-L"),
        ("bleu1",             "BLEU-1"),
        ("bleu2",             "BLEU-2"),
        ("bertscore_f1",      "BERTScore F1"),
        ("minicheck_support", "MiniCheck"),
        ("mem_gamma_mean",    "γ mean"),
        ("mem_memory_norm",   "Memory norm"),
    ]

    def fmt_val(key: str, val: float) -> str:
        if key in PCT_METRICS:
            return f"{val * 100:.2f}"
        return f"{val:.4f}"

    col_w   = 22
    met_w   = 16
    num_cols = len(ablations)

    header_parts = [f"{'Metric':<{met_w}}"] + [
        f"{ABLATION_LABELS.get(a, a):>{col_w}}" for a in ablations
    ]
    sep = "─" * (met_w + col_w * num_cols + num_cols)

    print()
    print("=" * len(sep))
    print("ABLATION STUDY — REMMORIZE")
    print("=" * len(sep))
    print("  ".join(header_parts))
    print(sep)

    full_metrics = results.get("full") or {}

    for metric_key, metric_label in metrics_to_show:
        # Skip if no variant has this metric
        has_any = any(
            r is not None and r.get(metric_key) is not None
            for r in results.values()
        )
        if not has_any:
            continue

        row_parts = [f"{metric_label:<{met_w}}"]
        for ablation in ablations:
            r = results.get(ablation)
            if r is None:
                row_parts.append(f"{'—':>{col_w}}")
            else:
                val = r.get(metric_key)
                if val is None:
                    row_parts.append(f"{'n/a':>{col_w}}")
                else:
                    # Mark best value with * (only for quality metrics)
                    best = None
                    if metric_key not in ("mem_gamma_mean", "mem_memory_norm"):
                        best_val = max(
                            (rv.get(metric_key, 0.0) or 0.0)
                            for rv in results.values()
                            if rv is not None
                        )
                        best = best_val

                    cell = fmt_val(metric_key, val)
                    if best is not None and abs(val - best) < 1e-6:
                        cell = cell + "*"
                    row_parts.append(f"{cell:>{col_w}}")

        print("  ".join(row_parts))

    print(sep)
    print("* = best value for this metric")
    print()

    # Delta from full model
    print("DELTA FROM FULL MODEL (ablation − full):")
    print(sep)
    for metric_key, metric_label in metrics_to_show:
        has_any = any(
            r is not None and r.get(metric_key) is not None
            for r in results.values()
        )
        if not has_any:
            continue
        if metric_key in ("mem_gamma_mean", "mem_memory_norm"):
            continue

        full_val = full_metrics.get(metric_key)
        if full_val is None:
            continue

        row_parts = [f"{metric_label:<{met_w}}"]
        for ablation in ablations:
            if ablation == "full":
                row_parts.append(f"{'—':>{col_w}}")
                continue
            r = results.get(ablation)
            if r is None:
                row_parts.append(f"{'—':>{col_w}}")
            else:
                val = r.get(metric_key)
                if val is None:
                    row_parts.append(f"{'n/a':>{col_w}}")
                else:
                    delta = val - full_val
                    if metric_key in PCT_METRICS:
                        delta *= 100
                    sign = "+" if delta >= 0 else ""
                    cell = f"{sign}{delta:.2f}" if metric_key in PCT_METRICS else f"{sign}{delta:.4f}"
                    row_parts.append(cell.rjust(col_w))
        print("  ".join(row_parts))

    print(sep)
    print()

    # LaTeX table snippet
    print("LATEX TABLE SNIPPET (copy to paper):")
    print("─" * 60)
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Ablation study on memory components.}")
    print(r"\label{tab:ablation}")

    col_spec = "l" + "r" * num_cols
    print(rf"\begin{{tabular}}{{{col_spec}}}")
    print(r"\toprule")

    header_row = " & ".join(
        ["Metric"] + [ABLATION_LABELS.get(a, a) for a in ablations]
    ) + r" \\"
    print(header_row)
    print(r"\midrule")

    for metric_key, metric_label in metrics_to_show:
        has_any = any(
            r is not None and r.get(metric_key) is not None
            for r in results.values()
        )
        if not has_any or metric_key in ("mem_gamma_mean", "mem_memory_norm"):
            continue

        best_val = max(
            (r.get(metric_key) or 0.0)
            for r in results.values()
            if r is not None
        )

        cells = [metric_label]
        for ablation in ablations:
            r = results.get(ablation)
            if r is None:
                cells.append("—")
            else:
                val = r.get(metric_key)
                if val is None:
                    cells.append("—")
                else:
                    cell = fmt_val(metric_key, val)
                    if abs(val - best_val) < 1e-6:
                        cell = rf"\textbf{{{cell}}}"
                    cells.append(cell)
        print(" & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Optional[Dict]] = {}

    if args.compare_only:
        log.info("--compare_only: loading existing metrics.json files.")
        for ablation in args.ablations:
            results[ablation] = load_existing_metrics(ablation, output_dir)
            if results[ablation] is None:
                log.warning("No metrics.json found for ablation '%s'.", ablation)
    else:
        g1 = [a for a in args.ablations if a in GROUP1_ABLATIONS]
        g2 = [a for a in args.ablations if a in GROUP2_ABLATIONS]

        if g1 and not args.checkpoint:
            raise ValueError("--checkpoint is required for Group I ablations (or use --compare_only).")
        if g1 and not args.eval_file:
            raise ValueError("--eval_file is required (or use --compare_only).")
        if g2 and not args.train_file:
            raise ValueError(
                "--train_file is required for Group II ablations: "
                + str(g2)
            )

        for ablation in args.ablations:
            ablation_dir = output_dir / ablation
            ablation_dir.mkdir(parents=True, exist_ok=True)
            if ablation in GROUP2_ABLATIONS:
                results[ablation] = run_training_ablation(ablation, args, ablation_dir)
            else:
                results[ablation] = run_one_ablation(ablation, args, ablation_dir)

    print_comparison_table(results, args.ablations)

    # Save summary JSON
    summary_path = output_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "ablations": args.ablations,
                "results":   {k: v for k, v in results.items() if v is not None},
            },
            f,
            indent=2,
        )
    log.info("Ablation summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
