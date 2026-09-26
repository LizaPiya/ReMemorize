"""
evaluate_dpo_baseline.py — inference + scoring for the DPO/GRPO-direct baseline
(LoRA checkpoint from train_dpo_baseline.py, no memory module).

evaluate.py can't be reused directly for this: ReMemorizeLLM.load_checkpoint()
expects the memory-interface checkpoint format (memory_interface.pt +
config.json), not a peft LoRA adapter (adapter_model.safetensors +
adapter_config.json). This script handles LoRA loading and generation, then
reuses evaluate.py's scoring pipeline (load_eval_data, run_metrics) unchanged
so the metrics are computed identically to every other result in this project.

Usage:
    python evaluate_dpo_baseline.py \
        --checkpoint runs/dpo_baseline/mimic/best_ckpt \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --eval_file Datasets/mimic_short_5k_test.jsonl \
        --dataset mimic \
        --output_dir results/dpo_baseline/mimic \
        --run_name dpo_baseline_mimic
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from peft import PeftModel

from evaluate import load_eval_data, run_metrics
from llm_backbone import ReMemorizeLLM
from memory_manager import ReMemorizeMemoryManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("evaluate_dpo_baseline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to a LoRA adapter checkpoint dir (adapter_model.safetensors + adapter_config.json).")
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--eval_file", type=str, required=True)
    p.add_argument("--dataset", type=str, default="mimic", choices=["mimic", "mts_dialog"])
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--max_source_length", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=280)
    p.add_argument("--min_new_tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.1,
                    help="Matches evaluate.py's convention: near-greedy at eval time, "
                         "not the higher temperature used for GRPO candidate sampling.")
    p.add_argument("--no_flash_attn", action="store_true")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Throwaway memory manager -- same reason as train_dpo_baseline.py: the
    # constructor requires one, but ablation_mode="no_memory" never touches it.
    inert_mm = ReMemorizeMemoryManager(d_key=64, d_value=64, d_memory=128, seed=42, device=device)
    backbone = ReMemorizeLLM(
        args.model, inert_mm,
        max_source_length=args.max_source_length,
        max_target_length=args.max_new_tokens,
        use_flash_attention=not args.no_flash_attn,
        freeze_llm=True,  # eval only, nothing trains here
    )

    log.info("Loading LoRA adapter from %s ...", args.checkpoint)
    backbone.llm = PeftModel.from_pretrained(backbone.llm, args.checkpoint, is_trainable=False).to(device)
    backbone.llm.eval()

    examples = load_eval_data(args.eval_file, dataset_type=args.dataset)
    if args.max_examples:
        examples = examples[: args.max_examples]
    log.info("Loaded %d eval examples.", len(examples))

    predictions_path = output_dir / "predictions.jsonl"
    t0 = time.time()
    with open(predictions_path, "w") as out_f:
        for idx, ex in enumerate(examples):
            with torch.no_grad():
                prediction = backbone.generate_summary(
                    ex["source"],
                    dataset_type=args.dataset,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    temperature=args.temperature,
                    do_sample=False,  # near-greedy at eval time, matching evaluate.py's convention
                    ablation_mode="no_memory",
                )
            record = {
                "id": ex["id"],
                "source": ex["source"],
                "prediction": prediction,
                "reference": ex["reference"],
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                log.info("Generated %d / %d  (%.1fs elapsed)", idx + 1, len(examples), elapsed)

    log.info("Predictions saved to %s", predictions_path)

    records = [json.loads(line) for line in open(predictions_path)]
    metrics = run_metrics(records, output_dir, device=device, n_boot=args.n_boot)
    log.info("Metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
