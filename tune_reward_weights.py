from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Tuple

import numpy as np

from reward_manager import (
    RewardCalibrator,
    RewardComponents,
    RewardWeights,
    calibrated_composite_reward,
    composite_reward,
)


def _pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return 0.0
    a = np.array(xs, dtype=np.float64)
    b = np.array(ys, dtype=np.float64)
    mat = np.corrcoef(a, b)
    val = float(mat[0, 1])
    return 0.0 if np.isnan(val) else val


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] == "[":
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError("JSON file must be a list of objects.")
        return data
    rows = []
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _sample_simplex4(rng: Random) -> RewardWeights:
    a = [rng.gammavariate(1.0, 1.0) for _ in range(4)]
    s = sum(a)
    return RewardWeights(alpha=a[0] / s, beta=a[1] / s, gamma=a[2] / s, delta=a[3] / s)


def _score_weights(
    rows: List[Dict[str, Any]],
    weights: RewardWeights,
    objective: str,
    target_key: str,
    use_calibrated: bool,
) -> float:
    preds: List[float] = []
    targets: List[float] = []
    calibrator = RewardCalibrator() if use_calibrated else None

    for row in rows:
        comp = RewardComponents(
            factual=float(row["factual"]),
            coherence=float(row["coherence"]),
            completeness=float(row.get("completeness", row.get("efficiency", 0.0))),
            hallucination=float(row["hallucination"]),
        )
        if use_calibrated:
            score = calibrated_composite_reward(comp, calibrator=calibrator, weights=weights).calibrated_reward
        else:
            score = composite_reward(
                r_factual=comp.factual,
                r_coherence=comp.coherence,
                r_completeness=comp.completeness,
                r_hallucination=comp.hallucination,
                weights=weights,
            )
        preds.append(score)
        if target_key in row:
            targets.append(float(row[target_key]))

    if objective == "mean_reward":
        return float(np.mean(preds))
    if not targets:
        raise ValueError(f"Objective '{objective}' requires target key '{target_key}' in validation data.")
    if objective == "pearson_target":
        return _pearson(preds, targets)
    if objective == "neg_mse_target":
        return -float(np.mean([(p - t) ** 2 for p, t in zip(preds, targets)]))
    raise ValueError(f"Unknown objective: {objective}")


def tune_reward_weights(
    rows: List[Dict[str, Any]],
    trials: int,
    seed: int,
    objective: str,
    target_key: str,
    use_calibrated: bool,
) -> Tuple[RewardWeights, float]:
    rng = Random(seed)

    candidates: List[RewardWeights] = [
        RewardWeights(alpha=1.0, beta=0.3, gamma=0.1, delta=0.8),
        RewardWeights(alpha=1.0, beta=0.2, gamma=0.1, delta=1.0),
        RewardWeights(alpha=1.0, beta=0.5, gamma=0.2, delta=0.8),
    ]
    candidates.extend(_sample_simplex4(rng) for _ in range(max(0, trials)))

    best_w = candidates[0]
    best_score = float("-inf")
    for w in candidates:
        score = _score_weights(
            rows=rows,
            weights=w,
            objective=objective,
            target_key=target_key,
            use_calibrated=use_calibrated,
        )
        if score > best_score:
            best_score = score
            best_w = w
    return best_w, best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune alpha/beta/gamma/delta on validation reward components.")
    parser.add_argument("--data", required=True, help="Path to JSON/JSONL validation file.")
    parser.add_argument("--output", default="reward_weights.best.json", help="Output JSON file for best weights.")
    parser.add_argument("--trials", type=int, default=500, help="Number of random simplex samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--objective",
        choices=["mean_reward", "pearson_target", "neg_mse_target"],
        default="mean_reward",
        help="Validation objective to maximize.",
    )
    parser.add_argument(
        "--target-key",
        default="target",
        help="Target column name used by pearson_target / neg_mse_target objectives.",
    )
    parser.add_argument(
        "--use-calibrated",
        action="store_true",
        help="Use calibrated composite reward instead of raw composite reward.",
    )
    args = parser.parse_args()

    rows = _load_rows(Path(args.data))
    if not rows:
        raise ValueError("Validation file is empty.")

    required_core = {"factual", "coherence", "hallucination"}
    missing = required_core - set(rows[0].keys())
    if missing:
        raise ValueError(f"Validation rows missing required keys: {sorted(missing)}")
    if "completeness" not in rows[0] and "efficiency" not in rows[0]:
        raise ValueError("Validation rows must include either 'completeness' or legacy 'efficiency'.")

    best_w, best_score = tune_reward_weights(
        rows=rows,
        trials=args.trials,
        seed=args.seed,
        objective=args.objective,
        target_key=args.target_key,
        use_calibrated=args.use_calibrated,
    )

    out = {
        "alpha": best_w.alpha,
        "beta": best_w.beta,
        "gamma": best_w.gamma,
        "delta": best_w.delta,
        "best_objective_score": best_score,
        "objective": args.objective,
        "use_calibrated": args.use_calibrated,
        "trials": args.trials,
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
