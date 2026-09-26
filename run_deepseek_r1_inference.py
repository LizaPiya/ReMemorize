"""
Custom DeepSeek-R1-Distill-Llama-8B inference for ReMemorize baselines.

Why a separate script (not evaluate_baselines.py):
  1. DeepSeek-R1 needs trust_remote_code=True for its tokenizer.
  2. It emits <think>...</think> reasoning preambles that must be stripped.
  3. It needs its own chat template applied with `apply_chat_template`.
  4. Recommended sampling: temperature=0.6, top_p=0.95 (per DeepSeek docs).

Output format matches the existing baselines: predictions.jsonl with
{id, source, prediction, reference} so evaluate.py --run metrics works.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


SYSTEM_PROMPT = (
    "You are a clinical summarization assistant. Produce a concise, faithful "
    "summary of the clinical note below. Output only the summary text. "
    "Do not include reasoning, headings, or explanations of your process."
)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
LEADING_THINK_RE = re.compile(r"^.*?</think>\s*", re.DOTALL)

# GPT-2/LLaMA byte-level BPE markers that survive tokenizer.decode()
_BPE_BYTES = bytes(range(256))
_BPE_CHARS = "".join(
    chr(c) if (c >= 33 and c <= 126) or (c >= 161 and c <= 172) or (c >= 174 and c <= 255)
    else chr(256 + c)
    for c in range(256)
)
_BPE_TO_BYTE = {c: b for c, b in zip(_BPE_CHARS, _BPE_BYTES)}


def _decode_bpe_artifacts(text: str) -> str:
    """Convert GPT-2 byte-level BPE marker chars (Ġ, Ċ, …) to real bytes."""
    if not any(c in _BPE_TO_BYTE for c in text):
        return text
    out = bytearray()
    for ch in text:
        if ch in _BPE_TO_BYTE:
            out.append(_BPE_TO_BYTE[ch])
        else:
            out.extend(ch.encode("utf-8"))
    return out.decode("utf-8", errors="replace")


def clean_prediction(text: str) -> str:
    """Strip DeepSeek-R1 reasoning content and BPE byte artifacts."""
    if "</think>" in text:
        text = LEADING_THINK_RE.sub("", text, count=1)
    text = THINK_RE.sub("", text)
    text = re.sub(r"^\s*<think>\s*", "", text)
    text = _decode_bpe_artifacts(text)
    return text.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    p.add_argument("--eval_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset", required=True, choices=["mimic", "mts_dialog"])
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_source_chars", type=int, default=12000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    user_template = (
        "Clinical note:\n{source}\n\n"
        "Write a brief, faithful summary of the clinical note above. "
        "Only include facts that are explicitly stated in the note."
    )

    examples = [json.loads(l) for l in open(args.eval_file)]
    print(f"Running inference on {len(examples)} examples")

    t0 = time.time()
    with open(pred_path, "w") as fout:
        for i, ex in enumerate(examples):
            if args.dataset == "mimic":
                source = ex.get("note", ex.get("source", ""))
                reference = ex.get("bhc", ex.get("target") or ex.get("reference") or "")
            else:  # mts_dialog / soap
                source = ex.get("dialogue", ex.get("source", ""))
                reference = ex.get("note", ex.get("reference") or ex.get("target") or "")
            source = source[: args.max_source_chars]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_template.format(source=source)},
            ]
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                truncation=True,
                max_length=8192,
            )
            input_ids = (encoded.input_ids if hasattr(encoded, "input_ids") else encoded).to(model.device)
            input_len = input_ids.shape[1]

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0][input_len:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=True)
            prediction = clean_prediction(raw)

            fout.write(json.dumps({
                "id":         ex.get("id", str(i)),
                "source":     source,
                "prediction": prediction,
                "reference":  reference,
            }) + "\n")
            fout.flush()

            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                print(f"  {i+1}/{len(examples)}  elapsed={elapsed:.0f}s  rate={rate:.2f} ex/s")

    print(f"Done. Wrote {len(examples)} predictions to {pred_path}")


if __name__ == "__main__":
    main()
