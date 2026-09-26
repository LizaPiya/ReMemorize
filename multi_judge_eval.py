"""
multi_judge_eval.py
───────────────────
Multi-LLM-judge evaluation for ReMemorize summaries.

Runs 3 open-source judge models independently on the same predictions,
then computes ensemble scores + inter-judge agreement (Krippendorff's alpha).

Usage
-----
# Evaluate predictions from a JSONL file (source, prediction, reference):
python multi_judge_eval.py \
    --predictions results/mimic_preds.jsonl \
    --output     results/judge_scores.jsonl \
    --judges     llama mistral qwen \
    --device     cuda

# Run only one judge (for debugging / cost saving):
python multi_judge_eval.py \
    --predictions results/mimic_preds.jsonl \
    --judges llama \
    --output results/judge_scores_llama.jsonl

Predictions JSONL format (one JSON per line):
    {"id": "...", "source": "...", "prediction": "...", "reference": "..."}

Output files:
    <output>.csv        — flat CSV, one row per example, columns:
                          id, llama_hallucination, llama_hallucination_raw, ...,
                          ensemble_hallucination, ..., agreement_hallucination, ...
    <output>.stats.json — aggregate means, stds, corpus-level Krippendorff alpha
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ─────────────────────────────────────────────────────────────────────────────
# Judge model registry
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_MODELS: Dict[str, str] = {
    "llama":   "meta-llama/Llama-3.2-3B-Instruct",   # AgenticSum judge lineage (3B — weak, upgrade to 8B when available)
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",  # Different lineage
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",            # Different training data/family
}
# NOTE: Llama-3B has high parse failure rate on structured output.
# Upgrade to 8B: snapshot_download('meta-llama/Llama-3.2-8B-Instruct')
# then change llama entry to 'meta-llama/Llama-3.2-8B-Instruct'

DIMENSIONS = ["hallucination", "factual", "complete", "coherent"]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeScores:
    hallucination: float   # 0=none, 1=severe  (inverted from 1-5 scale)
    factual:       float   # 0=bad,  1=perfect
    complete:      float   # 0=bad,  1=perfect
    coherent:      float   # 0=bad,  1=perfect
    parse_failed:  bool = False

    def to_dict(self) -> Dict:
        return {
            "hallucination": self.hallucination,
            "factual":       self.factual,
            "complete":      self.complete,
            "coherent":      self.coherent,
            "parse_failed":  self.parse_failed,
        }


@dataclass
class EnsembleResult:
    scores_per_judge: Dict[str, JudgeScores]   # judge_name -> scores
    ensemble:         Dict[str, float]          # dimension -> mean score
    agreement:        Dict[str, float]          # dimension -> krippendorff alpha
    raw_1_to_5:       Dict[str, Dict[str, float]]  # judge_name -> dimension -> raw 1-5


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

# ── Standard prompt (direct scoring) ─────────────────────────────────────────
EVAL_PROMPT_TEMPLATE = """\
You are an expert medical summarization evaluator. Evaluate the generated summary \
against the source clinical document on four dimensions using a strict 1–5 scale.

SOURCE DOCUMENT:
{source}

GENERATED SUMMARY:
{summary}

Rate each dimension 1–5 using EXACTLY this format (no other text):
Hallucination: X
Factual: X
Complete: X
Coherent: X

Scoring guide:
- Hallucination: 1=no hallucinated content, 5=major fabrications present
- Factual: 1=inaccurate, 5=perfectly accurate against source
- Complete: 1=missing key clinical info, 5=comprehensive coverage
- Coherent: 1=disorganized/unclear, 5=well-structured and clear"""

# ── G-Eval style CoT prompt (reasoning before scoring) ───────────────────────
# Based on: G-Eval (Liu et al. 2023) — chain-of-thought before final scores
# reduces score collapse and forces the model to ground reasoning in the text.
EVAL_PROMPT_COT_TEMPLATE = """\
You are an expert medical summarization evaluator. Your task is to evaluate \
a generated clinical summary against the source document.

SOURCE DOCUMENT:
{source}

GENERATED SUMMARY:
{summary}

Step 1 — Reasoning (think through each dimension carefully):

Hallucination analysis: Identify any specific claims in the summary that cannot \
be verified from or are contradicted by the source document. Quote examples if present.

Factual analysis: Check whether the clinical facts (diagnoses, medications, \
procedures, dates, values) in the summary accurately reflect the source.

Completeness analysis: Identify key clinical information in the source that is \
missing from the summary (diagnoses, treatments, outcomes, follow-up plans).

Coherence analysis: Evaluate the logical flow, sentence structure, and overall \
readability of the summary as a clinical document.

Step 2 — Scores (after reasoning above, assign 1–5 for each):
Hallucination: X
Factual: X
Complete: X
Coherent: X

Scoring guide:
- Hallucination: 1=no hallucinated content, 5=major fabrications present
- Factual: 1=inaccurate, 5=perfectly accurate against source
- Complete: 1=missing key clinical info, 5=comprehensive coverage
- Coherent: 1=disorganized/unclear, 5=well-structured and clear"""


def build_prompt(source: str, summary: str, max_source_chars: int = 4000, use_cot: bool = False) -> str:
    template = EVAL_PROMPT_COT_TEMPLATE if use_cot else EVAL_PROMPT_TEMPLATE
    return template.format(
        source=source[:max_source_chars],
        summary=summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

_PATTERNS = {
    "hallucination": r"Hallucination:\s*([1-5])",
    "factual":       r"Factual:\s*([1-5])",
    "complete":      r"Complete:\s*([1-5])",
    "coherent":      r"Coherent:\s*([1-5])",
}


def parse_scores(text: str) -> Optional[Dict[str, float]]:
    """Parse 1-5 scores from judge output. Returns None on failure."""
    out: Dict[str, float] = {}
    for key, pat in _PATTERNS.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m is None:
            return None
        out[key] = float(m.group(1))
    return out


def normalize_1_to_5(x: float) -> float:
    """Map 1..5 -> 0..1"""
    return max(0.0, min(1.0, (x - 1.0) / 4.0))


NEUTRAL_FALLBACK = {d: 3.0 for d in DIMENSIONS}  # neutral on parse fail


# ─────────────────────────────────────────────────────────────────────────────
# Single judge
# ─────────────────────────────────────────────────────────────────────────────

class LLMJudge:
    """
    Wraps a single HuggingFace instruct model for scoring.
    Loaded once, reused across all examples.
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        device: str = "cuda",
        max_new_tokens: int = 80,
        load_in_4bit: bool = False,
        retry: int = 1,
        max_source_chars: int = 4000,
        use_cot: bool = False,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.retry = retry
        self.max_source_chars = max_source_chars
        self.use_cot = use_cot

        log.info("Loading judge '%s' from %s ...", name, model_id)
        self.model, self.tokenizer = self._load(model_id, device, load_in_4bit)
        log.info("Judge '%s' ready.", name)

    def _load(self, model_id: str, device: str, load_in_4bit: bool):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs = dict(
            torch_dtype=torch.float16,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.eval()
        return model, tokenizer

    def _generate(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _score_tokens_prob(self, prompt_with_label: str) -> Dict[str, float]:
        """
        G-Eval token probability scoring (Liu et al. 2023).

        For each dimension, appends "DimName: " to the CoT output and reads
        the probability distribution over tokens "1".."5". The final score is
        the weighted average: sum(i * P(token_i)) for i in 1..5.
        This gives a continuous score in [1,5] instead of a hard integer,
        making scores more discriminative.

        Called after CoT generation so the model already has reasoning context.
        """
        import torch
        import torch.nn.functional as F

        scores: Dict[str, float] = {}
        digit_tokens = {}
        for digit in ["1", "2", "3", "4", "5"]:
            ids = self.tokenizer.encode(digit, add_special_tokens=False)
            # Take last token id if tokenizer splits (e.g. " 1" -> [space, 1])
            digit_tokens[digit] = ids[-1]

        for dim in ["Hallucination", "Factual", "Complete", "Coherent"]:
            # Append the label prefix; model next-token is the score digit
            label_prompt = prompt_with_label + f"\n{dim}: "
            inputs = self.tokenizer(
                label_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=8192,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits[0, -1, :]  # (vocab,)
            probs = F.softmax(logits, dim=-1)

            # Weighted average over digits 1..5
            weighted = sum(
                float(i + 1) * float(probs[digit_tokens[str(i + 1)]])
                for i in range(5)
            )
            # Normalize by total probability mass on 1..5 to avoid near-zero issues
            total_mass = sum(float(probs[digit_tokens[str(i + 1)]]) for i in range(5))
            scores[dim.lower()] = weighted / total_mass if total_mass > 1e-8 else 3.0

        return scores

    def score(self, source: str, summary: str) -> Tuple[JudgeScores, Dict[str, float]]:
        """
        Returns (JudgeScores normalized 0-1, raw_1_to_5 dict).

        Standard mode  (use_cot=False): direct structured prompt → parse integers.
        G-Eval mode    (use_cot=True):  CoT reasoning → token probability scoring.
            - Model reasons through each dimension first (reduces score collapse).
            - Scores extracted via next-token probability distribution over 1..5.
            - Gives continuous scores, more discriminative than integer parsing.

        On parse failure (standard mode only): returns neutral scores with parse_failed=True.
        """
        prompt = build_prompt(source, summary, self.max_source_chars, use_cot=self.use_cot)
        parse_failed = False

        if self.use_cot:
            # Step 1: generate the full CoT reasoning
            cot_response = self._generate(prompt)
            # Step 2: score via token probabilities conditioned on the reasoning
            full_context = prompt + "\n" + cot_response
            raw = self._score_tokens_prob(full_context)
        else:
            raw = None
            for attempt in range(self.retry + 1):
                response = self._generate(prompt)
                raw = parse_scores(response)
                if raw is not None:
                    break
                log.warning("Judge '%s' parse failed (attempt %d): %s", self.name, attempt + 1, response[:100])
            parse_failed = raw is None
            if parse_failed:
                raw = NEUTRAL_FALLBACK.copy()

        scores = JudgeScores(
            hallucination=normalize_1_to_5(raw["hallucination"]),
            factual=normalize_1_to_5(raw["factual"]),
            complete=normalize_1_to_5(raw["complete"]),
            coherent=normalize_1_to_5(raw["coherent"]),
            parse_failed=parse_failed,
        )
        return scores, raw


# ─────────────────────────────────────────────────────────────────────────────
# Inter-judge agreement: Krippendorff's alpha
# ─────────────────────────────────────────────────────────────────────────────

def krippendorff_alpha(ratings: List[List[float]]) -> float:
    """
    Compute Krippendorff's alpha for ordinal data.
    ratings: list of rater lists, each of length N (N examples).
    Returns alpha in [-1, 1]. >0.8 = strong, 0.6-0.8 = moderate, <0.6 = weak.
    """
    ratings_arr = np.array(ratings, dtype=float)   # (n_raters, n_items)
    n_raters, n_items = ratings_arr.shape

    if n_raters < 2 or n_items < 2:
        return float("nan")

    # Observed disagreement
    do = 0.0
    n_pairs = 0
    for item in range(n_items):
        vals = ratings_arr[:, item]
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                do += (vals[i] - vals[j]) ** 2
                n_pairs += 1
    do = do / n_pairs if n_pairs > 0 else 0.0

    # Expected disagreement (from marginal distribution)
    all_vals = ratings_arr.flatten()
    de = 0.0
    n_total = len(all_vals)
    for i in range(n_total):
        for j in range(i + 1, n_total):
            de += (all_vals[i] - all_vals[j]) ** 2
    de = de / (n_total * (n_total - 1) / 2) if n_total > 1 else 1.0

    if de == 0.0:
        return 1.0
    return float(1.0 - do / de)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────

def compute_ensemble(
    judge_scores: Dict[str, JudgeScores],
    raw_scores: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns:
        ensemble: dimension -> mean score (0-1) across judges
        agreement: dimension -> Krippendorff alpha across judges
    """
    judge_names = list(judge_scores.keys())

    ensemble: Dict[str, float] = {}
    agreement: Dict[str, float] = {}

    for dim in DIMENSIONS:
        vals_normalized = [getattr(judge_scores[j], dim) for j in judge_names]
        vals_raw = [raw_scores[j][dim] for j in judge_names]

        ensemble[dim] = float(np.mean(vals_normalized))
        # Agreement computed on raw 1-5 (ordinal, more meaningful)
        agreement[dim] = krippendorff_alpha([[v] for v in vals_raw]) if len(judge_names) > 1 else float("nan")

    return ensemble, agreement


def compute_corpus_agreement(
    all_raw: List[Dict[str, Dict[str, float]]],
    judge_names: List[str],
) -> Dict[str, float]:
    """
    Compute Krippendorff's alpha over the full corpus (more reliable than per-example).
    all_raw: list of {judge_name: {dim: raw_score}} per example.
    Returns {dim: alpha}.
    """
    agreement: Dict[str, float] = {}
    for dim in DIMENSIONS:
        ratings = []
        for judge in judge_names:
            rater_scores = [ex[judge][dim] for ex in all_raw]
            ratings.append(rater_scores)
        agreement[dim] = krippendorff_alpha(ratings)
    return agreement


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate summary stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary_stats(
    results: List[Dict],
    judge_names: List[str],
) -> Dict:
    """
    Compute mean ± std for each judge and ensemble across the corpus.
    """
    stats: Dict = {"per_judge": {}, "ensemble": {}, "corpus_agreement": {}}

    for judge in judge_names:
        stats["per_judge"][judge] = {}
        for dim in DIMENSIONS:
            vals_0to1 = [r["scores"][judge][dim] for r in results if not r["parse_failures"].get(judge, False)]
            vals_1to5 = [r["raw_1_to_5"][judge][dim] for r in results if not r["parse_failures"].get(judge, False)]
            if vals_0to1:
                stats["per_judge"][judge][dim] = {
                    "mean":      float(np.mean(vals_1to5)),   # 1-5 scale (paper-ready)
                    "std":       float(np.std(vals_1to5)),
                    "mean_0to1": float(np.mean(vals_0to1)),   # 0-1 normalized (legacy)
                    "std_0to1":  float(np.std(vals_0to1)),
                    "n":         len(vals_0to1),
                }

    for dim in DIMENSIONS:
        # Ensemble mean of raw 1-5 scores (paper-ready)
        # Skip examples where all judges failed parsing (avoid NaN from np.mean([]))
        vals_1to5 = []
        for r in results:
            raw = [r["raw_1_to_5"][j][dim] for j in judge_names
                   if not r["parse_failures"].get(j, False)]
            if raw:
                vals_1to5.append(float(np.mean(raw)))
        vals_0to1 = [r["ensemble"][dim] for r in results]
        stats["ensemble"][dim] = {
            "mean":      float(np.mean(vals_1to5)) if vals_1to5 else float("nan"),
            "std":       float(np.std(vals_1to5))  if vals_1to5 else float("nan"),
            "mean_0to1": float(np.mean(vals_0to1)),   # 0-1 normalized (legacy)
            "std_0to1":  float(np.std(vals_0to1)),
        }

    # Corpus-level agreement (all examples together — more reliable)
    all_raw = [r["raw_1_to_5"] for r in results]
    stats["corpus_agreement"] = compute_corpus_agreement(all_raw, judge_names)

    # Parse failure rates
    stats["parse_failure_rates"] = {
        judge: float(np.mean([r["parse_failures"].get(judge, False) for r in results]))
        for judge in judge_names
    }

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions(path: str) -> List[Dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def run(args: argparse.Namespace) -> None:
    # ── Load judges ──────────────────────────────────────────────────────────
    judges: Dict[str, LLMJudge] = {}
    if args.use_cot:
        log.info("G-Eval CoT mode enabled: reasoning + token probability scoring.")

    for name in args.judges:
        if name not in JUDGE_MODELS:
            log.error("Unknown judge '%s'. Available: %s", name, list(JUDGE_MODELS.keys()))
            sys.exit(1)
        judges[name] = LLMJudge(
            name=name,
            model_id=JUDGE_MODELS[name],
            device=args.device,
            load_in_4bit=args.load_in_4bit,
            retry=args.retry,
            max_source_chars=args.max_source_chars,
            use_cot=args.use_cot,
        )

    # ── Load predictions ─────────────────────────────────────────────────────
    examples = load_predictions(args.predictions)
    log.info("Loaded %d examples from %s", len(examples), args.predictions)

    if args.max_examples:
        examples = examples[: args.max_examples]
        log.info("Capped to %d examples (--max_examples)", len(examples))

    # ── Score + collect results ───────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict] = []
    judge_names = list(judges.keys())

    for idx, ex in enumerate(examples):
        ex_id = ex.get("id", str(idx))
        source = ex["source"]
        prediction = ex["prediction"]

        judge_scores: Dict[str, JudgeScores] = {}
        raw_scores: Dict[str, Dict[str, float]] = {}
        parse_failures: Dict[str, bool] = {}

        for judge_name, judge in judges.items():
            scores, raw = judge.score(source, prediction)
            judge_scores[judge_name] = scores
            raw_scores[judge_name] = raw
            parse_failures[judge_name] = scores.parse_failed

        ensemble, per_example_agreement = compute_ensemble(judge_scores, raw_scores)

        record = {
            "id":             ex_id,
            "scores":         {j: judge_scores[j].to_dict() for j in judge_scores},
            "raw_1_to_5":     raw_scores,
            "ensemble":       ensemble,
            "agreement":      per_example_agreement,
            "parse_failures": parse_failures,
        }
        results.append(record)

        if (idx + 1) % 10 == 0:
            log.info("Scored %d / %d examples", idx + 1, len(examples))

    # ── Save per-example CSV (flattened) ──────────────────────────────────────
    import csv

    # CSV columns: raw 1-5 scores as primary, 0-1 normalized as secondary
    # ensemble_* columns are mean raw 1-5 across judges
    csv_path = output_path.with_suffix(".csv")
    flat_fields = ["id"]
    for j in judge_names:
        for dim in DIMENSIONS:
            flat_fields.append(f"{j}_{dim}_1to5")     # primary: raw 1-5
            flat_fields.append(f"{j}_{dim}_0to1")     # secondary: normalized 0-1
        flat_fields.append(f"{j}_parse_failed")
    for dim in DIMENSIONS:
        flat_fields.append(f"ensemble_{dim}_1to5")    # mean raw 1-5 across judges
        flat_fields.append(f"agreement_{dim}_alpha")  # Krippendorff alpha

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for r in results:
            row: Dict = {"id": r["id"]}
            for j in judge_names:
                for dim in DIMENSIONS:
                    row[f"{j}_{dim}_1to5"] = round(r["raw_1_to_5"][j][dim], 2)
                    row[f"{j}_{dim}_0to1"] = round(r["scores"][j][dim], 4)
                row[f"{j}_parse_failed"] = r["parse_failures"].get(j, False)
            for dim in DIMENSIONS:
                # Ensemble as mean raw 1-5 (more interpretable than 0-1)
                raw_vals = [r["raw_1_to_5"][j][dim] for j in judge_names]
                row[f"ensemble_{dim}_1to5"]   = round(float(np.mean(raw_vals)), 2)
                row[f"agreement_{dim}_alpha"]  = round(r["agreement"].get(dim, float("nan")), 4)
            writer.writerow(row)

    log.info("Per-example scores saved to %s", csv_path)

    # ── Summary stats ─────────────────────────────────────────────────────────
    stats = compute_summary_stats(results, judge_names)

    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MULTI-JUDGE EVALUATION REPORT")
    print("=" * 60)
    print(f"Examples evaluated : {len(results)}")
    print(f"Judges used        : {', '.join(judge_names)}")
    print(f"Scoring mode       : {'G-Eval CoT + token probabilities' if args.use_cot else 'Direct (standard)'}")
    print()

    print("PER-JUDGE MEANS (1-5 scale):")
    header = f"{'Dimension':<16}" + "".join(f"{j:>12}" for j in judge_names) + f"{'ensemble':>12}"
    print(header)
    print("-" * len(header))
    for dim in DIMENSIONS:
        row_str = f"{dim:<16}"
        raw_means = []
        for j in judge_names:
            # Compute mean of raw 1-5 scores for this judge+dim
            raw_vals = [r["raw_1_to_5"][j][dim] for r in results
                        if not r["parse_failures"].get(j, False)]
            m = float(np.mean(raw_vals)) if raw_vals else float("nan")
            raw_means.append(m)
            row_str += f"{m:>12.2f}"
        ensemble_mean = float(np.mean(raw_means)) if raw_means else float("nan")
        row_str += f"{ensemble_mean:>12.2f}"
        print(row_str)

    print()
    print("CORPUS-LEVEL INTER-JUDGE AGREEMENT (Krippendorff α):")
    print("  (>0.8 strong, 0.6-0.8 moderate, <0.6 weak)")
    for dim in DIMENSIONS:
        alpha = stats["corpus_agreement"].get(dim, float("nan"))
        print(f"  {dim:<16} α = {alpha:.4f}")

    print()
    print("PARSE FAILURE RATES:")
    for j, rate in stats["parse_failure_rates"].items():
        print(f"  {j:<16} {rate:.1%}")

    # ── Per-example 1-5 scores table ─────────────────────────────────────────
    print()
    print("PER-EXAMPLE SCORES (raw 1-5, ensemble mean):")
    ex_header = f"{'id':<20}" + "".join(f"{d[:5]:>8}" for d in DIMENSIONS)
    print(ex_header)
    print("-" * len(ex_header))
    for r in results:
        ex_id = str(r["id"])[:19]
        row_str = f"{ex_id:<20}"
        for dim in DIMENSIONS:
            raw_vals = [r["raw_1_to_5"][j][dim] for j in judge_names
                        if not r["parse_failures"].get(j, False)]
            ens = float(np.mean(raw_vals)) if raw_vals else float("nan")
            row_str += f"{ens:>8.1f}"
        print(row_str)

    print()
    print(f"Per-example scores : {csv_path}")
    print(f"Aggregate stats    : {stats_path}")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-LLM-judge evaluation for ReMemorize")
    p.add_argument("--predictions",    required=True,  help="JSONL file with source/prediction/reference")
    p.add_argument("--output",         required=True,  help="Output base path; .csv (per-example) and .stats.json (aggregate) will be written")
    p.add_argument("--judges",         nargs="+", default=list(JUDGE_MODELS.keys()),
                   help=f"Which judges to use. Choices: {list(JUDGE_MODELS.keys())}")
    p.add_argument("--device",         default="cuda", help="cuda or cpu")
    p.add_argument("--load_in_4bit",   action="store_true", help="4-bit quantization (saves VRAM)")
    p.add_argument("--use_cot",        action="store_true",
                   help="G-Eval mode: CoT reasoning + token probability scoring instead of direct integer parsing")
    p.add_argument("--retry",          type=int, default=1, help="Retries on parse failure (standard mode only)")
    p.add_argument("--max_source_chars", type=int, default=4000)
    p.add_argument("--max_examples",   type=int, default=None, help="Cap number of examples (for testing)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
