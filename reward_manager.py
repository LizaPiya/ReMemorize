from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class RewardWeights:
    alpha: float = 1.0
    beta: float = 0.3
    gamma: float = 0.1
    delta: float = 0.8


@dataclass
class RewardComponents:
    factual: float
    coherence: float
    completeness: float
    hallucination: float


@dataclass
class RewardCalibrationStats:
    factual_z: float
    coherence_z: float
    completeness_z: float
    hallucination_z: float
    calibrated_reward: float


class RunningMoments:
    """
    EMA-based running moments for online reward normalization.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def update(self, x: float) -> None:
        if not self.initialized:
            self.mean = x
            self.var = 1.0
            self.initialized = True
            return
        delta = x - self.mean
        self.mean = (1.0 - self.alpha) * self.mean + self.alpha * x
        self.var = (1.0 - self.alpha) * self.var + self.alpha * (delta * delta)

    def zscore(self, x: float, eps: float = 1e-8) -> float:
        std = np.sqrt(max(self.var, eps))
        return (x - self.mean) / std


class RewardCalibrator:
    """
    Reward calibration:
    - per-component online z-score normalization
    - z-score clipping to reduce outlier influence
    - optional factual floor penalty
    """

    def __init__(
        self,
        ema_alpha: float = 0.05,
        clip_z: float = 3.0,
        factual_floor: float = 0.0,
        factual_floor_penalty: float = 0.0,
    ) -> None:
        self.clip_z = clip_z
        self.factual_floor = factual_floor
        self.factual_floor_penalty = factual_floor_penalty
        self._moments: Dict[str, RunningMoments] = {
            "factual": RunningMoments(alpha=ema_alpha),
            "coherence": RunningMoments(alpha=ema_alpha),
            "completeness": RunningMoments(alpha=ema_alpha),
            "hallucination": RunningMoments(alpha=ema_alpha),
        }

    def calibrate(self, comp: RewardComponents, w: RewardWeights) -> RewardCalibrationStats:
        self._moments["factual"].update(comp.factual)
        self._moments["coherence"].update(comp.coherence)
        self._moments["completeness"].update(comp.completeness)
        self._moments["hallucination"].update(comp.hallucination)

        fz = float(np.clip(self._moments["factual"].zscore(comp.factual), -self.clip_z, self.clip_z))
        cz = float(np.clip(self._moments["coherence"].zscore(comp.coherence), -self.clip_z, self.clip_z))
        gz = float(np.clip(self._moments["completeness"].zscore(comp.completeness), -self.clip_z, self.clip_z))
        hz = float(np.clip(self._moments["hallucination"].zscore(comp.hallucination), -self.clip_z, self.clip_z))

        reward = w.alpha * fz + w.beta * cz + w.gamma * gz - w.delta * hz
        if comp.factual < self.factual_floor:
            reward -= self.factual_floor_penalty

        return RewardCalibrationStats(
            factual_z=fz,
            coherence_z=cz,
            completeness_z=gz,
            hallucination_z=hz,
            calibrated_reward=reward,
        )


def composite_reward(
    r_factual: float,
    r_coherence: float,
    r_completeness: float,
    r_hallucination: float,
    weights: RewardWeights | None = None,
) -> float:
    w = weights or RewardWeights()
    return (
        w.alpha * r_factual
        + w.beta * r_coherence
        + w.gamma * r_completeness
        - w.delta * r_hallucination
    )


def calibrated_composite_reward(
    components: RewardComponents,
    calibrator: RewardCalibrator,
    weights: RewardWeights | None = None,
) -> RewardCalibrationStats:
    return calibrator.calibrate(components, w=weights or RewardWeights())


def training_objective(
    supervised_loss: float,
    reward: float,
    lambda_rl: float,
) -> float:
    return supervised_loss - lambda_rl * reward
