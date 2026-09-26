"""
evaluate_baselines.py — Run and score plain HuggingFace baseline models.

Generates summaries from any causal-LM or seq2seq model (no ReMemorize
memory module), computes the full metric suite, and appends a row to
results/experiments.csv in the same format as evaluate.py.

Usage
-----
# Mistral-7B on MIMIC test set:
conda run -n FLP_ReMemorize python evaluate_baselines.py \
    --model  mistralai/Mistral-7B-v0.1 \
    --run_name  mistral_7b \
    --eval_file Datasets/mimic_test.jsonl \
    --dataset   mimic \
    --output_dir results/baselines/mistral_7b_mimic

# LLaMA-3.2-3B on SOAP test set:
conda run -n FLP_ReMemorize python evaluate_baselines.py \
    --model  meta-llama/Llama-3.2-3B-Instruct \
    --run_name  llama3_3b \
    --eval_file Datasets/soap_test.jsonl \
    --dataset   mts_dialog \
    --output_dir results/baselines/llama3_3b_soap
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Prompts ───────────────────────────────────────────────────────────────────
MIMIC_SYSTEM = (
    "You are a clinical documentation specialist. "
    "Write a concise Brief Hospital Course (BHC) summary of the following "
    "clinical note. Include only information supported by the note."
)
SOAP_SYSTEM = (
    "You are a clinical documentation specialist. "
    "Write a concise SOAP note summarizing the following patient-doctor conversation. "
    "Include only information supported by the conversation."
)

MIMIC_USER_TMPL = "Clinical note:\n{source}\n\nBrief Hospital Course summary:"
SOAP_USER_TMPL  = "Patient-doctor conversation:\n{source}\n\nSOAP summary:"


ALPACA_TMPL = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

# Short prompts for seq2seq models that struggle with long instructions
MIMIC_SEQ2SEQ_TMPL = "Summarize the following clinical note in one paragraph:\n{source}"
SOAP_SEQ2SEQ_TMPL  = "Summarize the following patient-doctor conversation as a SOAP note:\n{source}"

ALPACA_MODELS  = {"medalpaca", "alpaca"}
SEQ2SEQ_MODELS = {"flan", "biobart", "bart", "t5"}


def _is_alpaca_model(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(tag in name_lower for tag in ALPACA_MODELS)


def _is_seq2seq_model(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(tag in name_lower for tag in SEQ2SEQ_MODELS)


def build_prompt(source: str, dataset: str, tokenizer, model_name: str = "") -> str:
    """Build a chat-formatted or plain prompt depending on model type."""
    system = MIMIC_SYSTEM if dataset == "mimic" else SOAP_SYSTEM
    user   = (MIMIC_USER_TMPL if dataset == "mimic" else SOAP_USER_TMPL).format(
        source=source[:3000]  # truncate to keep prompt manageable
    )

    # Alpaca-format models (MedAlpaca, etc.)
    if _is_alpaca_model(model_name):
        return ALPACA_TMPL.format(instruction=system, input=user)

    # Seq2seq models (Flan-T5, BioBart) — short direct prompt works better
    if _is_seq2seq_model(model_name):
        tmpl = MIMIC_SEQ2SEQ_TMPL if dataset == "mimic" else SOAP_SEQ2SEQ_TMPL
        return tmpl.format(source=source[:1024])

    # Use apply_chat_template if available (instruct models)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    # Fallback: plain concatenation for base models
    return f"{system}\n\n{user}\n"


# ── Data loading ──────────────────────────────────────────────────────────────
def load_eval_data(eval_file: str, dataset: str) -> List[Dict]:
    examples = []
    with open(eval_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if dataset == "mimic":
                source    = ex.get("note", ex.get("source", ""))
                reference = ex.get("bhc",  ex.get("reference", ex.get("target", "")))
            else:
                source    = ex.get("dialogue", ex.get("source", ""))
                reference = ex.get("note",     ex.get("reference", ex.get("target", "")))
            examples.append({
                "id":        ex.get("hadm_id", ex.get("ID", str(i))),
                "source":    source,
                "reference": reference,
            })
    return examples


# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(args: argparse.Namespace, output_dir: Path) -> Path:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    log.info("Loading model: %s", args.model)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception:
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False, legacy=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw: dict = {}
    if device == "cuda":
        if getattr(args, "load_in_4bit", False):
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kw["device_map"] = "auto"
        else:
            load_kw["device_map"] = "auto"
            load_kw["torch_dtype"] = torch.float16
    else:
        load_kw["torch_dtype"] = torch.float32

    # Try causal LM first, fall back to seq2seq (e.g. Flan-T5, BioBART)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
        is_seq2seq = False
    except Exception:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **load_kw)
        is_seq2seq = True

    if device == "cpu":
        model = model.to(device)
    model.eval()
    log.info("Model loaded (seq2seq=%s)", is_seq2seq)

    examples = load_eval_data(args.eval_file, args.dataset)
    if args.max_examples:
        examples = examples[: args.max_examples]
    log.info("Loaded %d examples from %s", len(examples), args.eval_file)

    predictions_path = output_dir / "predictions.jsonl"
    t0 = time.time()

    with open(predictions_path, "w") as out_f:
        for idx, ex in enumerate(examples):
            prompt = build_prompt(ex["source"], args.dataset, tokenizer, args.model)

            if is_seq2seq:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_source_length,
                ).to(device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=4,
                        early_stopping=True,
                    )
                prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_source_length,
                ).to(device)
                input_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                # Decode only the newly generated tokens
                prediction = tokenizer.decode(
                    output_ids[0][input_len:], skip_special_tokens=True
                ).strip()

            record = {
                "id":         ex["id"],
                "source":     ex["source"],
                "prediction": prediction,
                "reference":  ex["reference"],
                "model":      args.model,
                "memory_diagnostics": {},
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if (idx + 1) % 5 == 0:
                log.info("Generated %d / %d  (%.1fs)", idx + 1, len(examples),
                         time.time() - t0)

    log.info("Predictions saved to %s", predictions_path)
    return predictions_path


# ── Metrics (mirrors evaluate.py) ─────────────────────────────────────────────
def compute_rouge(predictions, references):
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = max(len(predictions), 1)
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        for k in totals:
            totals[k] += s[k].fmeasure
    return {k: v / n for k, v in totals.items()}


def compute_bleu(predictions, references):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoother = SmoothingFunction().method1
    b1 = b2 = 0.0
    n = max(len(predictions), 1)
    for pred, ref in zip(predictions, references):
        r = [ref.lower().split()]
        p = pred.lower().split()
        b1 += sentence_bleu(r, p, weights=(1, 0, 0, 0),     smoothing_function=smoother)
        b2 += sentence_bleu(r, p, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    return {"bleu1": b1 / n, "bleu2": b2 / n}


def compute_bertscore(predictions, references, device="cpu"):
    from bert_score import score as bscore
    predictions = [p if p.strip() else "." for p in predictions]
    references  = [r if r.strip() else "." for r in references]
    _, _, F1 = bscore(
        predictions, references,
        model_type="roberta-large",
        device=device,
        verbose=False,
    )
    return {"bertscore_f1": float(F1.mean())}


def compute_minicheck(sources, predictions, device="cpu"):
    """
    Faithfulness + hallucination via DeBERTa-v3 NLI (same model used in training rewards).
    entailment prob = faithfulness, contradiction prob = hallucination.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = "cross-encoder/nli-deberta-v3-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()

        entail_scores, contradict_scores = [], []
        for src, pred in zip(sources, predictions):
            inputs = tokenizer(
                src[:1500], pred[:500],
                return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
            entail_scores.append(float(probs[2]))
            contradict_scores.append(float(probs[0]))

        n = max(len(entail_scores), 1)
        return {
            "minicheck_support":   sum(entail_scores)    / n,
            "hallucination_score": sum(contradict_scores) / n,
        }
    except Exception as e:
        log.warning("NLI scoring failed: %s", e)
        return {"minicheck_support": None, "hallucination_score": None}


def run_metrics(records: List[Dict], output_dir: Path, device: str) -> Dict:
    predictions = [r["prediction"] for r in records]
    references  = [r["reference"]  for r in records]
    sources     = [r["source"]     for r in records]

    metrics: Dict = {}
    log.info("Computing ROUGE ...")
    metrics.update(compute_rouge(predictions, references))
    log.info("Computing BLEU ...")
    metrics.update(compute_bleu(predictions, references))
    log.info("Computing BERTScore ...")
    metrics.update(compute_bertscore(predictions, references, device))
    log.info("Computing MiniCheck ...")
    metrics.update(compute_minicheck(sources, predictions, device))

    # Per-example CSV
    per_example_path = output_dir / "per_example.csv"
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    with open(per_example_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "rouge1", "rouge2", "rougeL", "pred_len", "ref_len",
            "prediction", "reference",
        ])
        writer.writeheader()
        for r in records:
            s = scorer.score(r["reference"], r["prediction"])
            writer.writerow({
                "id":         r["id"],
                "rouge1":     round(s["rouge1"].fmeasure, 4),
                "rouge2":     round(s["rouge2"].fmeasure, 4),
                "rougeL":     round(s["rougeL"].fmeasure, 4),
                "pred_len":   len(r["prediction"].split()),
                "ref_len":    len(r["reference"].split()),
                "prediction": r["prediction"],
                "reference":  r["reference"],
            })

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved to %s", metrics_path)
    return metrics


# ── Experiment log ────────────────────────────────────────────────────────────
def log_experiment(args, metrics, n_examples, output_dir):
    log_path = Path(args.experiments_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M"),
        "run_name":        args.run_name or Path(args.output_dir).name,
        "checkpoint":      args.model,
        "eval_file":       args.eval_file or "",
        "n_examples":      n_examples,
        "rouge1":          metrics.get("rouge1", ""),
        "rouge2":          metrics.get("rouge2", ""),
        "rougeL":          metrics.get("rougeL", ""),
        "bleu1":           metrics.get("bleu1",  ""),
        "bleu2":           metrics.get("bleu2",  ""),
        "bertscore_f1":    metrics.get("bertscore_f1", ""),
        "minicheck":       metrics.get("minicheck_support", ""),
        "mem_gamma_mean":  "",
        "mem_memory_norm": "",
        "output_dir":      str(output_dir),
        "notes":           args.notes,
    }
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    log.info("Logged to %s", log_path)


# ── Print report ──────────────────────────────────────────────────────────────
def print_report(model, metrics, n_examples):
    print("\n" + "=" * 55)
    print(f"BASELINE EVALUATION: {model}")
    print("=" * 55)
    print(f"N examples: {n_examples}\n")
    for name, key in [
        ("ROUGE-1",      "rouge1"),
        ("ROUGE-2",      "rouge2"),
        ("ROUGE-L",      "rougeL"),
        ("BLEU-1",       "bleu1"),
        ("BLEU-2",       "bleu2"),
        ("BERTScore F1", "bertscore_f1"),
    ]:
        v = metrics.get(key)
        if v is not None:
            print(f"  {name:<16} {v * 100:.2f}")
    mc = metrics.get("minicheck_support")
    print(f"\n  {'MiniCheck':<16} {mc * 100:.2f}" if mc else "\n  MiniCheck        [skipped]")
    print("=" * 55)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a plain HuggingFace baseline model")
    p.add_argument("--model",             required=True,
                   help="HuggingFace model id or local path.")
    p.add_argument("--run_name",          type=str, default=None)
    p.add_argument("--eval_file",         required=True)
    p.add_argument("--dataset",           choices=["mimic", "mts_dialog"], default="mimic")
    p.add_argument("--output_dir",        required=True)
    p.add_argument("--max_new_tokens",    type=int, default=256)
    p.add_argument("--max_source_length", type=int, default=3000)
    p.add_argument("--max_examples",      type=int, default=None)
    p.add_argument("--run",               choices=["inference", "metrics", "all"],
                   default="all")
    p.add_argument("--predictions",       type=str, default=None,
                   help="Existing predictions.jsonl (--run metrics only).")
    p.add_argument("--device",            type=str, default=None)
    p.add_argument("--load_in_4bit",      action="store_true",
                   help="Load model in 4-bit NF4 quantization (required for 70B+ models).")
    p.add_argument("--notes",             type=str, default="baseline")
    p.add_argument("--experiments_log",   type=str,
                   default="results/experiments.csv")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = args.device or "cpu"

    predictions_path: Optional[Path] = None

    if args.run in ("inference", "all"):
        predictions_path = run_inference(args, output_dir)

    if args.run in ("metrics", "all"):
        if predictions_path is None:
            if not args.predictions:
                raise ValueError("--predictions required for --run metrics")
            predictions_path = Path(args.predictions)
        records = list(json.loads(l) for l in open(predictions_path) if l.strip())
        log.info("Scoring %d predictions ...", len(records))
        metrics = run_metrics(records, output_dir, device)
        print_report(args.model, metrics, len(records))
        log_experiment(args, metrics, len(records), output_dir)


if __name__ == "__main__":
    main()
