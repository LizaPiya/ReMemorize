"""
MemSum extractive baseline for ReMemorize.

Reads an evaluation JSONL (source, reference per line), runs MemSum's
sentence extractor on each source, joins extracted sentences as the
prediction, and writes predictions.jsonl in the same format as other
baselines so evaluate.py --run metrics can score it.

Usage:
    python run_memsum_baseline.py \
        --memsum_dir /path/to/MemSum \
        --model_path /path/to/memsum-pubmed/model.pt \
        --vocab_path /path/to/word_embedding/vocabulary_200dim.pkl \
        --eval_file Datasets/mimic_5k_test.jsonl \
        --output_dir results/baselines/memsum_pubmed_mimic
"""
import argparse
import json
import os
import sys
from pathlib import Path

import nltk
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        nltk.download("punkt", quiet=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memsum_dir", required=True, help="Path to cloned MemSum repo")
    parser.add_argument("--model_path", required=True, help="Path to memsum model.pt")
    parser.add_argument("--vocab_path", required=True, help="Path to vocabulary_200dim.pkl")
    parser.add_argument("--eval_file", required=True, help="JSONL with source/reference per line")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="mimic", choices=["mimic", "mts_dialog"],
                        help="Dataset type — controls field name mapping")
    parser.add_argument("--max_sentences", type=int, default=10,
                        help="Max sentences MemSum may extract per document")
    parser.add_argument("--p_stop_thres", type=float, default=0.2)
    parser.add_argument("--max_doc_len", type=int, default=500,
                        help="MemSum's per-document sentence cap")
    args = parser.parse_args()

    # Make MemSum importable
    sys.path.insert(0, args.memsum_dir)
    from src.summarizer import MemSum  # noqa: E402

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    print(f"Loading MemSum: {args.model_path}")
    model = MemSum(args.model_path, args.vocab_path, gpu=0, max_doc_len=args.max_doc_len)

    n = 0
    with open(args.eval_file) as fin, open(pred_path, "w") as fout:
        for line in fin:
            ex = json.loads(line)
            if args.dataset == "mimic":
                source = ex.get("note", ex.get("source", ""))
                reference = ex.get("bhc", ex.get("target") or ex.get("reference") or "")
            else:  # mts_dialog / soap
                source = ex.get("dialogue", ex.get("source", ""))
                reference = ex.get("note", ex.get("reference") or ex.get("target") or "")

            sentences = nltk.sent_tokenize(source)
            if not sentences:
                pred = ""
            else:
                extracted = model.extract(
                    [sentences],
                    p_stop_thres=args.p_stop_thres,
                    max_extracted_sentences_per_document=args.max_sentences,
                )[0]
                if not extracted:
                    extracted = sentences[:3]
                pred = " ".join(extracted)

            out = {
                "id": ex.get("id", str(n)),
                "source": source,
                "prediction": pred,
                "reference": reference,
            }
            fout.write(json.dumps(out) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"  processed {n}")

    print(f"Done. Wrote {n} predictions to {pred_path}")


if __name__ == "__main__":
    main()
