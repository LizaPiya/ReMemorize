from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Any, Dict, List, Optional, Set

from reward_manager import RewardComponents

# ---------------------------------------------------------------------------
# Lazy imports for heavyweight models — only loaded when NLIRewardScorer is
# instantiated, so the rest of the codebase has no hard torch/transformers dep.
# ---------------------------------------------------------------------------
_nli_pipeline = None   # shared across NLIRewardScorer instances
_sent_model = None     # sentence-transformer for coherence / completeness


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _tokens(text: str) -> List[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [t for t in cleaned.split() if t]


def _token_set(text: str) -> Set[str]:
    return set(_tokens(text))


def _slide_chunks(
    tokenizer,
    text: str,
    window_tokens: int = 450,
    stride_tokens: int = 250,
) -> List[str]:
    """
    Tokenize the FULL text and split it into overlapping chunks, so a model
    with a fixed max input length (e.g. DeBERTa's 512 tokens) still sees the
    entire document via max-pooling across chunks rather than a truncated
    prefix. Mirrors evaluate.py's _slide_source_chunks (same window/stride
    defaults) so the two stay consistent.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
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


def _sentences(text: str) -> List[str]:
    out: List[str] = []
    cur = []
    for ch in text:
        cur.append(ch)
        if ch in ".!?":
            s = "".join(cur).strip()
            if s:
                out.append(s)
            cur = []
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


class BaseRewardScorer(ABC):
    @abstractmethod
    def score(self, source: str, summary: str, reference: Optional[str] = None) -> RewardComponents:
        raise NotImplementedError


class HeuristicRewardScorer(BaseRewardScorer):
    """
    Lightweight local scorer (no external model calls).
    Replace with LLM-judge/NLI scorers for production experiments.
    """

    def score(self, source: str, summary: str, reference: Optional[str] = None) -> RewardComponents:
        src_toks = _token_set(source)
        sum_toks = _token_set(summary)
        ref_toks = _token_set(reference) if reference else set()

        # Factuality proxy: precision of summary tokens that appear in source.
        factual = len(sum_toks & src_toks) / max(len(sum_toks), 1)
        factual = _clamp01(factual)

        # Hallucination proxy: unsupported token ratio.
        halluc = 1.0 - factual
        halluc = _clamp01(halluc)

        # Coherence proxy: sentence-to-sentence lexical continuity,
        # penalized for near-duplicate sentences (repetition detection).
        sents = _sentences(summary)
        if len(sents) <= 1:
            coherence = 0.7 if sents else 0.0
        else:
            overlaps = []
            repetition_penalty = 0.0
            for a, b in zip(sents[:-1], sents[1:]):
                A = _token_set(a)
                B = _token_set(b)
                overlap = len(A & B) / max(len(A | B), 1)
                overlaps.append(overlap)
                # Sentences with >80% overlap are near-duplicates
                if overlap > 0.8:
                    repetition_penalty += 0.3
            coherence = _clamp01(
                sum(overlaps) / len(overlaps) + 0.2 - repetition_penalty
            )

        # Completeness proxy:
        # - with reference: recall against reference tokens
        # - without reference: source-coverage recall of source tokens
        if ref_toks:
            completeness = len(sum_toks & ref_toks) / max(len(ref_toks), 1)
        else:
            completeness = len(sum_toks & src_toks) / max(len(src_toks), 1)
        completeness = _clamp01(completeness)

        return RewardComponents(
            factual=factual,
            coherence=coherence,
            completeness=completeness,
            hallucination=halluc,
        )


class LLMJudgeRewardScorer(BaseRewardScorer):
    """
    LLM-as-a-judge scorer adapted from your previous AgenticSum evaluation script.
    Expects a pre-loaded HF-compatible model/tokenizer.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_source_chars: int = 4000,
        max_input_tokens: int = 8192,
        max_new_tokens: int = 80,
        do_sample: bool = False,
        temperature: float = 0.0,
        retry_on_parse_failure: int = 1,
        use_reference_for_completeness: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_source_chars = max_source_chars
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.retry_on_parse_failure = max(0, retry_on_parse_failure)
        self.use_reference_for_completeness = use_reference_for_completeness

    @staticmethod
    def _prompt(source: str, summary: str, reference: Optional[str], use_reference: bool) -> str:
        ref_block = ""
        if use_reference and reference:
            ref_block = f"\nREFERENCE SUMMARY (for completeness only):\n{reference}\n"
        return f"""Evaluate this medical summary against the source document. Rate 1-5 for each criterion.

SOURCE DOCUMENT:
{source}
{ref_block}
GENERATED SUMMARY:
{summary}

Rate 1-5:
- Hallucination (1=none, 5=major fabrications)
- Factual (1=inaccurate, 5=perfectly accurate)
- Complete (1=missing key info, 5=comprehensive)
- Coherent (1=poor writing, 5=excellent)

Respond in exactly this format:
Hallucination: X
Factual: X
Complete: X
Coherent: X"""

    @staticmethod
    def _parse_1_to_5(text: str) -> Optional[Dict[str, float]]:
        patterns = {
            "hallucination": r"Hallucination:\s*([1-5])",
            "factual": r"Factual:\s*([1-5])",
            "complete": r"Complete:\s*([1-5])",
            "coherent": r"Coherent:\s*([1-5])",
        }
        out: Dict[str, float] = {}
        for key, pat in patterns.items():
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m is None:
                return None
            out[key] = float(m.group(1))
        return out

    @staticmethod
    def _normalize_1_to_5(x: float) -> float:
        # Maps 1..5 -> 0..1
        return _clamp01((x - 1.0) / 4.0)

    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
        )
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def score(self, source: str, summary: str, reference: Optional[str] = None) -> RewardComponents:
        source_cut = source[: self.max_source_chars]
        prompt = self._prompt(
            source=source_cut,
            summary=summary,
            reference=reference,
            use_reference=self.use_reference_for_completeness,
        )

        parsed: Optional[Dict[str, float]] = None
        response = ""
        for _ in range(self.retry_on_parse_failure + 1):
            response = self._generate(prompt)
            parsed = self._parse_1_to_5(response)
            if parsed is not None:
                break

        # Neutral fallback on parse failures to stabilize training.
        if parsed is None:
            parsed = {"hallucination": 3.0, "factual": 3.0, "complete": 3.0, "coherent": 3.0}

        halluc = self._normalize_1_to_5(parsed["hallucination"])
        factual = self._normalize_1_to_5(parsed["factual"])
        complete = self._normalize_1_to_5(parsed["complete"])
        coherent = self._normalize_1_to_5(parsed["coherent"])

        return RewardComponents(
            factual=factual,
            coherence=coherent,
            completeness=complete,
            hallucination=halluc,
        )


class NLIRewardScorer(BaseRewardScorer):
    """
    NLI-based reward scorer using DeBERTa-v3 for factuality and hallucination,
    and sentence embeddings for coherence and completeness.

    Factuality  = P(entailment   | source, summary)   — does summary follow from source?
    Hallucination = P(contradiction | source, summary) — does source contradict summary?

    These two signals are genuinely independent: a summary can have low entailment
    (incomplete) without containing active contradictions.

    Coherence   = mean cosine similarity between consecutive summary sentence embeddings.
    Completeness = mean cosine recall of source sentences covered by the summary.

    Models used (all small, fast on RTX 6000 Ada):
      - cross-encoder/nli-deberta-v3-base  (~180M params, ~15ms/sample)
      - sentence-transformers/all-MiniLM-L6-v2  (~22M params, ~5ms/sentence)

    Usage:
        scorer = NLIRewardScorer(device="cuda")
        comps  = scorer.score(source, summary)
    """

    _NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
    _SENT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # DeBERTa NLI label order: contradiction=0, neutral=1, entailment=2
    _LABEL_CONTRADICTION = 0
    _LABEL_ENTAILMENT = 2

    def __init__(
        self,
        device: Optional[str] = None,
        chunk_window_tokens: int = 450,
        chunk_stride_tokens: int = 250,
        batch_size: int = 16,
    ) -> None:
        self.chunk_window_tokens = chunk_window_tokens
        self.chunk_stride_tokens = chunk_stride_tokens
        self.batch_size = batch_size
        self._device = device or ("cuda" if self._cuda_available() else "cpu")
        self._nli = None
        self._sent = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_models(self) -> None:
        """Lazy-load on first call to avoid import cost at module level."""
        if self._nli is not None:
            return
        from transformers import pipeline
        self._nli = pipeline(
            "text-classification",
            model=self._NLI_MODEL,
            device=0 if "cuda" in self._device else -1,
            top_k=None,
            truncation=True,
            max_length=512,
        )
        from sentence_transformers import SentenceTransformer
        self._sent = SentenceTransformer(self._SENT_MODEL, device=self._device)

    @staticmethod
    def _cosine(a, b) -> float:
        import numpy as np
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _nli_scores(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Returns {'entailment': p, 'contradiction': p, 'neutral': p}."""
        result = self._nli({"text": premise, "text_pair": hypothesis})
        # transformers 4.x: [[{label, score}, ...]]  transformers 5.x: [{label, score}, ...]
        items = result[0] if result and isinstance(result[0], list) else result
        return {item["label"].lower(): item["score"] for item in items}

    def score(self, source: str, summary: str, reference: Optional[str] = None) -> RewardComponents:
        self._load_models()

        # ---- factuality and hallucination (NLI, max-pooled over the full document) ----
        # DeBERTa's 512-token limit means a single pass can't see the whole
        # document (avg. ~2,267 tokens for MIMIC-IV-Ext-BHC), so the source is
        # split into overlapping chunks and the summary is scored against each;
        # a claim is supported/contradicted if any chunk supports/contradicts it.
        chunks = _slide_chunks(
            self._nli.tokenizer, source,
            window_tokens=self.chunk_window_tokens,
            stride_tokens=self.chunk_stride_tokens,
        )
        factual, hallucination = 0.0, 0.0
        for chunk in chunks:
            nli = self._nli_scores(premise=chunk, hypothesis=summary)
            factual = max(factual, _clamp01(nli.get("entailment", 0.0)))
            hallucination = max(hallucination, _clamp01(nli.get("contradiction", 0.0)))

        # ---- coherence (sentence embedding similarity) ----
        sents = _sentences(summary)
        if len(sents) <= 1:
            coherence = 0.7 if sents else 0.0
        else:
            embeddings = self._sent.encode(sents, batch_size=self.batch_size, show_progress_bar=False)
            sims = [
                self._cosine(embeddings[i], embeddings[i + 1])
                for i in range(len(embeddings) - 1)
            ]
            coherence = _clamp01(float(sum(sims) / len(sims)))

        # ---- completeness (source sentence recall) ----
        # No 512-token model limit here (sentence embeddings, not DeBERTa), so
        # this only needed the truncation removed, not chunking.
        src_sents = _sentences(source)
        if not src_sents or not sents:
            completeness = 0.0
        else:
            src_emb = self._sent.encode(src_sents, batch_size=self.batch_size, show_progress_bar=False)
            sum_emb = self._sent.encode(sents, batch_size=self.batch_size, show_progress_bar=False)
            # For each source sentence, find max cosine similarity with any summary sentence.
            # Completeness = mean of these max similarities (soft recall).
            import numpy as np
            sim_matrix = src_emb @ sum_emb.T  # (n_src, n_sum)
            norms_src = np.linalg.norm(src_emb, axis=1, keepdims=True).clip(1e-9)
            norms_sum = np.linalg.norm(sum_emb, axis=1, keepdims=True).clip(1e-9)
            sim_matrix = sim_matrix / norms_src / norms_sum.T
            completeness = _clamp01(float(sim_matrix.max(axis=1).mean()))

        return RewardComponents(
            factual=factual,
            coherence=coherence,
            completeness=completeness,
            hallucination=hallucination,
        )
