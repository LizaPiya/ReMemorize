"""
evaluate.py — Standalone evaluation for ReMemorize checkpoints.

Runs inference on an eval set, saves predictions to JSONL, then computes
the full metric suite. Designed to be run independently of training so you
can re-score predictions without re-running inference.

Two-stage design:
  Stage 1 (--run inference): load checkpoint → generate predictions → save JSONL
  Stage 2 (--run metrics):   load saved predictions → compute all metrics → report

Or run both in one go (default).

Usage
-----
# Full eval from a checkpoint:
python evaluate.py \
    --checkpoint  runs/mimic_v1/best_ckpt \
    --eval_file   Datasets/mimic_eval.jsonl \
    --output_dir  results/mimic_v1_eval \
    --model       mistralai/Mistral-7B-v0.1

# Re-score existing predictions (no inference needed):
python evaluate.py \
    --run metrics \
    --predictions results/mimic_v1_eval/predictions.jsonl \
    --output_dir  results/mimic_v1_eval

# Inference only (useful when you want to run multi_judge_eval.py separately):
python evaluate.py \
    --run inference \
    --checkpoint  runs/mimic_v1/best_ckpt \
    --eval_file   Datasets/mimic_eval.jsonl \
    --output_dir  results/mimic_v1_eval \
    --model       mistralai/Mistral-7B-v0.1

Output files
------------
  predictions.jsonl    — one record per example: id, source, prediction, reference,
                         memory_diagnostics (gamma_mean, memory_norm)
  metrics.json         — aggregate scores: ROUGE, BLEU, BERTScore, MiniCheck
  per_example.csv      — per-example scores for error analysis (open in Excel/pandas)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers  (imported from train.py logic, kept self-contained here)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        totals: Dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        n = max(len(predictions), 1)
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            for k in totals:
                totals[k] += s[k].fmeasure
        return {k: v / n for k, v in totals.items()}
    except ImportError:
        log.warning("rouge_score not installed; skipping ROUGE.")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoother = SmoothingFunction().method1
        b1, b2 = 0.0, 0.0
        n = max(len(predictions), 1)
        for pred, ref in zip(predictions, references):
            r = [ref.lower().split()]
            p = pred.lower().split()
            b1 += sentence_bleu(r, p, weights=(1, 0, 0, 0),     smoothing_function=smoother)
            b2 += sentence_bleu(r, p, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
        return {"bleu1": b1 / n, "bleu2": b2 / n}
    except ImportError:
        log.warning("nltk not installed; skipping BLEU.")
        return {"bleu1": 0.0, "bleu2": 0.0}


def compute_bertscore(predictions: List[str], references: List[str], device: str = "cpu") -> Dict[str, float]:
    try:
        from bert_score import score as bscore
        # Use bert-base-uncased explicitly: bert_score 0.3.13 + transformers >=5
        # breaks with lang="en" (resolves to roberta-large whose tokenizer lost
        # build_inputs_with_special_tokens in transformers 5.x).
        # Guard empty/whitespace-only strings — bert_score calls
        # tokenizer.build_inputs_with_special_tokens([]) for empty inputs which
        # also fails in transformers 5.x.
        predictions = [p if p.strip() else "." for p in predictions]
        references  = [r if r.strip() else "." for r in references]
        _, _, F1 = bscore(
            predictions, references,
            model_type="roberta-large",
            device=device,
            verbose=False,
        )
        return {"bertscore_f1": float(F1.mean())}
    except ImportError:
        log.warning("bert_score not installed; skipping BERTScore.")
        return {"bertscore_f1": 0.0}


_MINICHECK_MODEL_NAME = "lytang/MiniCheck-Flan-T5-Large"
_MINICHECK_WINDOW_TOKENS = 500
_MINICHECK_STRIDE_TOKENS = 250


def _split_into_claims(summary: str) -> List[str]:
    """Sentence-tokenize the summary into individual factual claims."""
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                nltk.download("punkt", quiet=True)
        sents = nltk.sent_tokenize(summary)
    except Exception:
        sents = [s.strip() for s in summary.replace("?", ".").replace("!", ".").split(".")]
    return [s.strip() for s in sents if len(s.strip()) > 0]


def _slide_source_chunks(
    tokenizer,
    source: str,
    window_tokens: int = _MINICHECK_WINDOW_TOKENS,
    stride_tokens: int = _MINICHECK_STRIDE_TOKENS,
) -> List[str]:
    """
    Tokenize the FULL source and split it into overlapping chunks.

    No character-level truncation is applied: the full document is processed
    via a sliding window, ensuring that supporting evidence anywhere in the
    document can be retrieved by max-pooling across chunks.
    """
    ids = tokenizer.encode(source, add_special_tokens=False)
    if not ids:
        return [""]
    chunks: List[str] = []
    start = 0
    while start < len(ids):
        window = ids[start: start + window_tokens]
        chunks.append(tokenizer.decode(window, skip_special_tokens=True))
        if start + window_tokens >= len(ids):
            break
        start += stride_tokens
    return chunks


def compute_minicheck(
    sources: List[str],
    predictions: List[str],
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Claim-level faithfulness scoring via MiniCheck (Tang et al., EMNLP 2024).

    For each prediction, the summary is decomposed into sentence-level claims.
    Each claim is checked against every overlapping ~500-token chunk of the
    full source (no truncation). The per-claim score is the maximum support
    probability across chunks (a claim is supported if any source chunk
    supports it). The summary score is the mean of per-claim scores.

    Returns:
        minicheck_support — mean per-claim max-support probability ∈ [0, 1].

    Hallucination is the complement (1 - factuality) by construction and is
    therefore not reported as a separate metric.
    """
    try:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(_MINICHECK_MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(_MINICHECK_MODEL_NAME)
        model.to(device)
        model.eval()

        # MiniCheck-T5 emits "0" (unsupported) or "1" (supported); the support
        # probability is the softmax mass on the "1" token at the first decoder step.
        yes_id = tokenizer("1", add_special_tokens=False).input_ids[0]
        no_id  = tokenizer("0", add_special_tokens=False).input_ids[0]

        per_summary_scores: List[float] = []

        for src, pred in zip(sources, predictions):
            claims = _split_into_claims(pred)
            if not claims:
                per_summary_scores.append(0.0)
                continue

            chunks = _slide_source_chunks(tokenizer, src)

            claim_scores: List[float] = []
            for claim in claims:
                best = 0.0
                for chunk in chunks:
                    prompt = f"predict: {chunk}\nclaim: {claim}"
                    inputs = tokenizer(
                        prompt, return_tensors="pt",
                        truncation=True, max_length=512,
                    ).to(device)
                    decoder_input_ids = torch.tensor(
                        [[model.config.decoder_start_token_id]], device=device
                    )
                    with torch.no_grad():
                        logits = model(
                            **inputs,
                            decoder_input_ids=decoder_input_ids,
                        ).logits[0, 0]
                    probs = torch.softmax(logits[[no_id, yes_id]], dim=-1)
                    best = max(best, float(probs[1]))
                claim_scores.append(best)

            per_summary_scores.append(sum(claim_scores) / len(claim_scores))

        n = max(len(per_summary_scores), 1)
        return {"minicheck_support": sum(per_summary_scores) / n}
    except Exception as e:
        log.warning("MiniCheck factuality scoring failed: %s", e)
        return {"minicheck_support": None}


def compute_per_example_rouge(pred: str, ref: str) -> Dict[str, float]:
    """Per-example ROUGE for error analysis."""
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        s = scorer.score(ref, pred)
        return {k: s[k].fmeasure for k in ["rouge1", "rouge2", "rougeL"]}
    except ImportError:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Inference
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_data(eval_file: Optional[str], dataset_type: str = "mimic") -> List[Dict]:
    """Load eval data from a JSONL file or, for mts_dialog, from HuggingFace if no file given."""
    if eval_file is None and dataset_type == "mts_dialog":
        from dataset import MTSDialogDataset
        hf_examples = MTSDialogDataset.from_huggingface("test")
        return [
            {"id": ex.doc_id, "source": ex.source, "reference": ex.reference}
            for ex in hf_examples
        ]

    examples = []
    with open(eval_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if dataset_type == "mimic":
                source    = ex.get("note", ex.get("source", ""))
                reference = ex.get("bhc",  ex.get("reference", ex.get("target", "")))
            else:  # mts_dialog
                source    = ex.get("dialogue", ex.get("source", ""))
                reference = ex.get("note", ex.get("reference", ex.get("target", "")))
            examples.append({
                "id":        ex.get("hadm_id", ex.get("ID", str(i))),
                "source":    source,
                "reference": reference,
            })
    return examples


def run_inference(args: argparse.Namespace, output_dir: Path) -> Path:
    """
    Load checkpoint, run generation on eval set, save predictions.jsonl.
    Returns path to predictions file.
    """
    import torch
    from memory_manager import ReMemorizeMemoryManager
    from llm_backbone import ReMemorizeLLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Running inference on device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("Loading checkpoint from %s ...", args.checkpoint)
    backbone, resume_info = ReMemorizeLLM.load_checkpoint(
        save_dir=args.checkpoint,
        model_name_or_path=args.model,
        # memory_manager=None → auto-reconstructed from config.json
        max_source_length=args.max_source_length,
        max_target_length=args.max_new_tokens,
        use_flash_attention=not args.no_flash_attn,
        load_in_4bit=args.load_in_4bit,
    )
    backbone.eval()
    log.info("Checkpoint loaded. (global_step=%d, best_rouge1=%.4f)",
             resume_info["global_step"], resume_info["best_rouge1"])

    # ── Load eval data ────────────────────────────────────────────────────────
    examples = load_eval_data(args.eval_file, dataset_type=args.dataset)
    if args.max_examples:
        examples = examples[:args.max_examples]
    log.info("Loaded %d eval examples.", len(examples))

    ablation = getattr(args, "ablation", "full") or "full"

    # ── Generate ──────────────────────────────────────────────────────────────
    predictions_path = output_dir / "predictions.jsonl"
    t0 = time.time()

    with open(predictions_path, "w") as out_f:
        for idx, ex in enumerate(examples):
            backbone.memory_manager.reset_state()

            # Phase 1: encode source into memory
            _, mem_stats = backbone.encode_and_update_memory(
                ex["source"], ablation_mode=ablation
            )

            # Phase 2: generate summary with live memory
            prediction = backbone.generate_summary(
                ex["source"],
                dataset_type=args.dataset,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                temperature=args.temperature,
                ablation_mode=ablation,
            )

            record = {
                "id":            ex["id"],
                "source":        ex["source"],
                "prediction":    prediction,
                "reference":     ex["reference"],
                "ablation_mode": ablation,
                "memory_diagnostics": {
                    "gamma_mean":   mem_stats.get("gamma_mean",  None),
                    "gamma_std":    mem_stats.get("gamma_std",   None),
                    "memory_norm":  mem_stats.get("memory_norm", None),
                    "recon_mse":    mem_stats.get("reconstruction_mse", None),
                },
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                log.info("Generated %d / %d  (%.1fs elapsed)", idx + 1, len(examples), elapsed)

    log.info("Predictions saved to %s", predictions_path)
    return predictions_path


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Metrics
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions(predictions_path: str) -> List[Dict]:
    records = []
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def bootstrap_std(
    predictions: List[str],
    references: List[str],
    sources: List[str],
    n_boot: int = 1000,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute bootstrap std for ROUGE, BLEU, BERTScore, and MiniCheck metrics.

    Computes per-example scores once, then resamples those scores n_boot times.
    This avoids reloading neural models (BERTScore, MiniCheck) on every iteration.
    """
    import random
    import numpy as np
    rng = random.Random(seed)
    n = len(predictions)

    # ── Step 1: compute per-example scores once ───────────────────────────────
    log.info("Bootstrap: computing per-example ROUGE ...")
    try:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        per_rouge: Dict[str, List[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            for k in per_rouge:
                per_rouge[k].append(s[k].fmeasure)
    except ImportError:
        per_rouge = {"rouge1": [0.0]*n, "rouge2": [0.0]*n, "rougeL": [0.0]*n}

    log.info("Bootstrap: computing per-example BLEU ...")
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smoother = SmoothingFunction().method1
        per_bleu: Dict[str, List[float]] = {"bleu1": [], "bleu2": []}
        for pred, ref in zip(predictions, references):
            r_tok = [ref.lower().split()]
            p_tok = pred.lower().split()
            per_bleu["bleu1"].append(sentence_bleu(r_tok, p_tok, weights=(1,0,0,0), smoothing_function=smoother))
            per_bleu["bleu2"].append(sentence_bleu(r_tok, p_tok, weights=(0.5,0.5,0,0), smoothing_function=smoother))
    except ImportError:
        per_bleu = {"bleu1": [0.0]*n, "bleu2": [0.0]*n}

    log.info("Bootstrap: computing per-example BERTScore ...")
    try:
        from bert_score import score as bscore
        preds_safe = [p if p.strip() else "." for p in predictions]
        refs_safe  = [r if r.strip() else "." for r in references]
        _, _, F1 = bscore(preds_safe, refs_safe, model_type="roberta-large", device=device, verbose=False)
        per_bert = {"bertscore_f1": F1.tolist()}
    except Exception:
        per_bert = {"bertscore_f1": [0.0]*n}

    log.info("Bootstrap: computing per-example MiniCheck ...")
    try:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        tokenizer = AutoTokenizer.from_pretrained(_MINICHECK_MODEL_NAME)
        mc_model = T5ForConditionalGeneration.from_pretrained(_MINICHECK_MODEL_NAME).to(device)
        mc_model.eval()
        yes_id = tokenizer("1", add_special_tokens=False).input_ids[0]
        no_id  = tokenizer("0", add_special_tokens=False).input_ids[0]
        per_support: List[float] = []
        for src, pred in zip(sources, predictions):
            claims = _split_into_claims(pred)
            if not claims:
                per_support.append(0.0)
                continue
            chunks = _slide_source_chunks(tokenizer, src)
            claim_scores: List[float] = []
            for claim in claims:
                best = 0.0
                for chunk in chunks:
                    prompt = f"predict: {chunk}\nclaim: {claim}"
                    inputs = tokenizer(prompt, return_tensors="pt",
                                       truncation=True, max_length=512).to(device)
                    decoder_input_ids = torch.tensor(
                        [[mc_model.config.decoder_start_token_id]], device=device
                    )
                    with torch.no_grad():
                        logits = mc_model(**inputs, decoder_input_ids=decoder_input_ids).logits[0, 0]
                    probs = torch.softmax(logits[[no_id, yes_id]], dim=-1)
                    best = max(best, float(probs[1]))
                claim_scores.append(best)
            per_support.append(sum(claim_scores) / len(claim_scores))
        per_mini = {"minicheck_support": per_support}
        del mc_model
    except Exception as e:
        log.warning("Bootstrap MiniCheck failed: %s", e)
        per_mini = {"minicheck_support": [0.0]*n}

    all_per_example = {**per_rouge, **per_bleu, **per_bert, **per_mini}

    # ── Step 2: resample pre-computed per-example scores ─────────────────────
    log.info("Bootstrap std: %d resample iterations ...", n_boot)
    accum: Dict[str, List[float]] = {k: [] for k in all_per_example}
    for i in range(n_boot):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        for k, vals in all_per_example.items():
            accum[k].append(float(np.mean([vals[j] for j in idx])))
        if (i + 1) % 100 == 0:
            log.info("  Bootstrap %d / %d done", i + 1, n_boot)

    stds: Dict[str, float] = {}
    for k, vals in accum.items():
        if vals:
            mean = sum(vals) / len(vals)
            stds[f"{k}_std"] = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    return stds


def run_metrics(
    records: List[Dict],
    output_dir: Path,
    device: str = "cpu",
    n_boot: int = 1000,
) -> Dict:
    """
    Compute full metric suite on loaded predictions.
    Saves per_example.csv and metrics.json.
    Returns aggregate metrics dict including bootstrap ±std for each metric.
    """
    import csv

    predictions = [r["prediction"] for r in records]
    references  = [r["reference"]  for r in records]
    sources     = [r["source"]     for r in records]

    metrics: Dict = {}

    # ── Corpus-level metrics ──────────────────────────────────────────────────
    log.info("Computing ROUGE ...")
    metrics.update(compute_rouge(predictions, references))

    log.info("Computing BLEU ...")
    metrics.update(compute_bleu(predictions, references))

    log.info("Computing BERTScore ...")
    metrics.update(compute_bertscore(predictions, references, device=device))

    log.info("Computing MiniCheck ...")
    metrics.update(compute_minicheck(sources, predictions, device=device))

    # ── Bootstrap std ─────────────────────────────────────────────────────────
    log.info("Computing bootstrap std (n_boot=%d) ...", n_boot)
    metrics.update(bootstrap_std(predictions, references, sources, n_boot=n_boot, device=device))

    # ── Memory diagnostics (mean over corpus) ────────────────────────────────
    diag_keys = ["gamma_mean", "gamma_std", "memory_norm", "recon_mse"]
    for k in diag_keys:
        vals = [
            r["memory_diagnostics"][k]
            for r in records
            if r.get("memory_diagnostics", {}).get(k) is not None
        ]
        if vals:
            metrics[f"mem_{k}"] = float(sum(vals) / len(vals))

    # ── Per-example scores (CSV) ──────────────────────────────────────────────
    per_example_path = output_dir / "per_example.csv"
    fieldnames = [
        "id", "rouge1", "rouge2", "rougeL",
        "pred_len", "ref_len",
        "gamma_mean", "gamma_std", "memory_norm", "recon_mse",
        "prediction", "reference",
    ]
    with open(per_example_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            rouge = compute_per_example_rouge(r["prediction"], r["reference"])
            diag  = r.get("memory_diagnostics") or {}
            writer.writerow({
                "id":           r["id"],
                "rouge1":       round(rouge["rouge1"],  4),
                "rouge2":       round(rouge["rouge2"],  4),
                "rougeL":       round(rouge["rougeL"],  4),
                "pred_len":     len(r["prediction"].split()),
                "ref_len":      len(r["reference"].split()),
                "gamma_mean":   diag.get("gamma_mean"),
                "gamma_std":    diag.get("gamma_std"),
                "memory_norm":  diag.get("memory_norm"),
                "recon_mse":    diag.get("recon_mse"),
                "prediction":   r["prediction"],
                "reference":    r["reference"],
            })

    log.info("Per-example scores saved to %s  (open in Excel/pandas)", per_example_path)

    # ── Save aggregate metrics ────────────────────────────────────────────────
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Aggregate metrics saved to %s", metrics_path)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def print_report(metrics: Dict, n_examples: int, checkpoint: Optional[str] = None) -> None:
    print("\n" + "=" * 55)
    print("REMMORIZE EVALUATION REPORT")
    print("=" * 55)
    if checkpoint:
        print(f"Checkpoint   : {checkpoint}")
    print(f"N examples   : {n_examples}")
    print()

    def fmt(key: str) -> str:
        val = metrics.get(key)
        std = metrics.get(f"{key}_std")
        if val is None:
            return "[n/a]"
        s = f"{val * 100:.2f}"
        if std is not None:
            s += f" ±{std * 100:.2f}"
        return s

    print("LEXICAL / SEMANTIC METRICS:")
    for name, key in [
        ("ROUGE-1",      "rouge1"),
        ("ROUGE-2",      "rouge2"),
        ("ROUGE-L",      "rougeL"),
        ("BLEU-1",       "bleu1"),
        ("BLEU-2",       "bleu2"),
        ("BERTScore F1", "bertscore_f1"),
    ]:
        print(f"  {name:<16} {fmt(key)}")

    print()
    print("FAITHFULNESS:")
    mc = metrics.get("minicheck_support")
    if mc is not None:
        print(f"  {'MiniCheck':<16} {fmt('minicheck_support')}")
    else:
        print("  MiniCheck        [not computed]")

    print()
    print("LLM JUDGE:")
    print("  Run multi_judge_eval.py on predictions.jsonl for judge scores.")

    mem_keys = [k for k in metrics if k.startswith("mem_")]
    if mem_keys:
        print()
        print("MEMORY DIAGNOSTICS (corpus mean):")
        for k in mem_keys:
            print(f"  {k:<24} {metrics[k]:.4f}")

    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a ReMemorize checkpoint")

    p.add_argument("--run", choices=["inference", "metrics", "all"], default="all",
                   help="Which stage(s) to run.")

    # Inference args
    p.add_argument("--checkpoint",        type=str, default=None,
                   help="Path to checkpoint directory (required for inference).")
    p.add_argument("--model",             type=str, default="mistralai/Mistral-7B-v0.1",
                   help="HuggingFace model id or local path.")
    p.add_argument("--eval_file",         type=str, default=None,
                   help="Path to eval JSONL (required for inference).")
    p.add_argument("--dataset",           type=str, default="mimic",
                   choices=["mimic", "mts_dialog"])
    p.add_argument("--max_new_tokens",    type=int, default=512)
    p.add_argument("--min_new_tokens",    type=int, default=200)
    p.add_argument("--temperature",       type=float, default=0.1)
    p.add_argument("--max_source_length", type=int, default=3840)
    p.add_argument("--no_flash_attn",     action="store_true")
    p.add_argument("--load_in_4bit",      action="store_true",
                   help="Load LLM backbone in 4-bit NF4 quantization (required for 70B+ models).")
    p.add_argument("--max_examples",      type=int, default=None,
                   help="Cap number of eval examples (for testing).")
    p.add_argument("--n_boot",            type=int, default=1000,
                   help="Number of bootstrap iterations for ±std estimation.")
    p.add_argument("--ablation",          type=str, default="full",
                   choices=["full", "no_memory", "fixed_gate", "no_phase2_update"],
                   help="Ablation variant to run. 'full' = default trained model.")

    # Metrics args
    p.add_argument("--predictions", type=str, default=None,
                   help="Path to existing predictions.jsonl (--run metrics only).")

    # Shared
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write predictions, metrics, and per-example scores.")
    p.add_argument("--device",     type=str, default=None,
                   help="cuda or cpu (auto-detected if not set).")
    p.add_argument("--run_name",   type=str, default=None,
                   help="Short name for this run (e.g. 'mimic_v1_best'). Used in experiments.csv.")
    p.add_argument("--notes",      type=str, default="",
                   help="Free-text notes logged to experiments.csv.")
    p.add_argument("--experiments_log", type=str, default="results/experiments.csv",
                   help="Master experiment log. Each eval run appends one row.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = args.device or "cpu"

    predictions_path: Optional[Path] = None

    # ── Stage 1: Inference ────────────────────────────────────────────────────
    if args.run in ("inference", "all"):
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for inference.")
        if not args.eval_file:
            raise ValueError("--eval_file is required for inference.")
        predictions_path = run_inference(args, output_dir)

    # ── Stage 2: Metrics ──────────────────────────────────────────────────────
    if args.run in ("metrics", "all"):
        if predictions_path is None:
            if not args.predictions:
                raise ValueError("--predictions is required when --run metrics.")
            predictions_path = Path(args.predictions)

        records = load_predictions(str(predictions_path))
        log.info("Loaded %d predictions from %s", len(records), predictions_path)

        metrics = run_metrics(records, output_dir, device=device, n_boot=args.n_boot)
        print_report(
            metrics,
            n_examples=len(records),
            checkpoint=args.checkpoint,
        )
        log_experiment(args, metrics, n_examples=len(records), output_dir=output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment log
# ─────────────────────────────────────────────────────────────────────────────

def log_experiment(
    args: argparse.Namespace,
    metrics: Dict,
    n_examples: int,
    output_dir: Path,
) -> None:
    """
    Append one row to experiments.csv — the master result table.
    Creates the file with a header if it doesn't exist yet.
    Safe to run concurrently (appends, never overwrites).
    """
    import csv
    from datetime import datetime

    log_path = Path(args.experiments_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp":             datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run_name":              args.run_name or Path(args.output_dir).name,
        "ablation":              getattr(args, "ablation", "full") or "full",
        "checkpoint":            args.checkpoint or "",
        "eval_file":             args.eval_file or "",
        "n_examples":            n_examples,
        "rouge1":                metrics.get("rouge1",             ""),
        "rouge1_std":            metrics.get("rouge1_std",         ""),
        "rouge2":                metrics.get("rouge2",             ""),
        "rouge2_std":            metrics.get("rouge2_std",         ""),
        "rougeL":                metrics.get("rougeL",             ""),
        "rougeL_std":            metrics.get("rougeL_std",         ""),
        "bleu1":                 metrics.get("bleu1",              ""),
        "bleu1_std":             metrics.get("bleu1_std",          ""),
        "bleu2":                 metrics.get("bleu2",              ""),
        "bleu2_std":             metrics.get("bleu2_std",          ""),
        "bertscore_f1":          metrics.get("bertscore_f1",       ""),
        "bertscore_f1_std":      metrics.get("bertscore_f1_std",   ""),
        "minicheck":             metrics.get("minicheck_support",  ""),
        "minicheck_std":         metrics.get("minicheck_support_std", ""),
        "mem_gamma_mean":        metrics.get("mem_gamma_mean",     ""),
        "mem_memory_norm":       metrics.get("mem_memory_norm",    ""),
        "output_dir":            str(output_dir),
        "notes":                 args.notes,
    }

    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    log.info("Experiment logged to %s", log_path)


if __name__ == "__main__":
    main()
