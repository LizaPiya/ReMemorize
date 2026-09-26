from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import List, Optional, Sequence

import numpy as np

from memory_manager import ReMemorizeMemoryManager
from policy_manager import PolicySample, UpdateStats, build_policy_sample
from reward_manager import (
    RewardCalibrator,
    RewardComponents,
    RewardWeights,
    calibrated_composite_reward,
)
from scorers import BaseRewardScorer, HeuristicRewardScorer


@dataclass
class TrainerConfig:
    ppo_lr: float = 1e-5              # paper Table 1 value
    clip_eps: float = 0.2
    entropy_coef: float = 0.005
    group_size: int = 4               # G candidates per document for GRPO
    max_kl: float = 0.02              # PPO KL early-stopping threshold
    mse_reward_weight: float = 0.1    # λ_mem: weight of dense MSE reward shaping
    lr_warmup_steps: int = 100        # linear warm-up steps for policy optimizer


class ReMemorizeTrainer:
    """
    Coordinates EW-RLS memory updates, NLI reward scoring, and PPO/GRPO
    policy optimisation.

    Key improvements over the original:
    - Dense reward shaping: text-quality reward + λ_mem·(1/(1+MSE))
      gives the policy per-window feedback rather than episode-only signal.
    - Proper GRPO: collect_grpo_batch generates G candidates per document
      and assigns them the same group_id for within-group normalisation.
    - LR warm-up: linear schedule applied to the policy AdamW optimizer.
    - reset_state: memory state is reset between documents so each document
      gets a fresh accumulator (correct for online document-level memory).
    """

    def __init__(
        self,
        memory_manager: ReMemorizeMemoryManager,
        reward_calibrator: RewardCalibrator,
        reward_scorer: Optional[BaseRewardScorer] = None,
        reward_weights: Optional[RewardWeights] = None,
        config: Optional[TrainerConfig] = None,
        seed: int = 0,
    ) -> None:
        self.memory_manager = memory_manager
        self.reward_calibrator = reward_calibrator
        self.reward_scorer = reward_scorer or HeuristicRewardScorer()
        self.reward_weights = reward_weights or RewardWeights()
        self.config = config or TrainerConfig()
        self.rng = Random(seed)
        self._step = 0                 # global step counter for LR scheduling

    # ------------------------------------------------------------------
    # LR scheduling
    # ------------------------------------------------------------------

    def _current_lr(self) -> float:
        """Linear warm-up then constant."""
        if self._step < self.config.lr_warmup_steps:
            return self.config.ppo_lr * (self._step + 1) / self.config.lr_warmup_steps
        return self.config.ppo_lr

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _text_reward(self, comp: RewardComponents) -> float:
        """Calibrated composite text-quality reward."""
        return calibrated_composite_reward(
            comp, calibrator=self.reward_calibrator, weights=self.reward_weights
        ).calibrated_reward

    def _dense_reward(self, text_reward: float, mse: float) -> float:
        """
        Combined reward: text quality + memory coherence bonus.
        1/(1+mse) is bounded in (0,1] and rewards lower reconstruction error.
        """
        mem_bonus = 1.0 / (1.0 + mse)
        return text_reward + self.config.mse_reward_weight * mem_bonus

    # ------------------------------------------------------------------
    # Batch collection (standard PPO path)
    # ------------------------------------------------------------------

    def collect_policy_batch(
        self,
        key_windows: Sequence[List[List[float]]],
        reward_components: Sequence[RewardComponents],
        value_windows: Optional[Sequence[List[List[float]]]] = None,
    ) -> List[PolicySample]:
        """
        For each (key_window, reward_components) pair:
          1. Run memory_manager.update() to get RLS stats + per-token readouts
          2. Compute dense reward (text quality + MSE shaping)
          3. Build PolicySample with memory-conditioned old_logprob

        value_windows: LLM value vectors (d_value,) per token. When None,
            falls back to key_windows as a self-supervised pretext task
            (only valid when d_key == d_value).
        """
        if len(key_windows) != len(reward_components):
            raise ValueError("key_windows and reward_components must match in length.")
        if value_windows is not None and len(value_windows) != len(key_windows):
            raise ValueError("value_windows and key_windows must match in length.")

        batch: List[PolicySample] = []
        for i, (keys, comp) in enumerate(zip(key_windows, reward_components)):
            vals = value_windows[i] if value_windows is not None else keys
            # Reset memory state for each new document segment
            self.memory_manager.reset_state()
            stats = self.memory_manager.update(keys, vals)
            readouts = stats["readouts"]
            mse = stats["reconstruction_mse"]

            text_r = self._text_reward(comp)
            reward = self._dense_reward(text_r, mse)
            group_id = i // max(1, self.config.group_size)

            batch.append(
                build_policy_sample(
                    self.memory_manager.policy,
                    keys,
                    readouts,
                    reward=reward,
                    group_id=group_id,
                    rng=self.rng,
                )
            )
        return batch

    # ------------------------------------------------------------------
    # GRPO batch collection (primary path)
    # ------------------------------------------------------------------

    def collect_grpo_batch(
        self,
        key_windows: Sequence[List[List[float]]],
        sources: Sequence[str],
        summary_groups: Sequence[List[str]],   # G candidate summaries per document
        references: Optional[Sequence[Optional[str]]] = None,
        value_windows: Optional[Sequence[List[List[float]]]] = None,
    ) -> List[PolicySample]:
        """
        GRPO-style batch: each document has G candidate summaries.
        All G samples share the same group_id so advantages are
        normalised within the group, requiring no value function.

        summary_groups[i] = [summary_1, summary_2, ..., summary_G]

        value_windows: LLM value vectors (d_value,) per token. When None,
            falls back to key_windows as a self-supervised pretext task
            (only valid when d_key == d_value).
        """
        n = len(key_windows)
        if len(sources) != n or len(summary_groups) != n:
            raise ValueError("key_windows, sources, and summary_groups must match in length.")
        if value_windows is not None and len(value_windows) != n:
            raise ValueError("value_windows and key_windows must match in length.")

        batch: List[PolicySample] = []
        for doc_idx in range(n):
            keys = key_windows[doc_idx]
            vals = value_windows[doc_idx] if value_windows is not None else keys
            candidates = summary_groups[doc_idx]
            group_id = doc_idx

            for summary in candidates:
                ref = references[doc_idx] if references is not None else None
                comp = self.reward_scorer.score(
                    source=sources[doc_idx], summary=summary, reference=ref
                )
                # Fresh memory state per candidate (each candidate is an independent rollout)
                self.memory_manager.reset_state()
                stats = self.memory_manager.update(keys, vals)
                readouts = stats["readouts"]
                mse = stats["reconstruction_mse"]

                text_r = self._text_reward(comp)
                reward = self._dense_reward(text_r, mse)

                batch.append(
                    build_policy_sample(
                        self.memory_manager.policy,
                        keys,
                        readouts,
                        reward=reward,
                        group_id=group_id,
                        rng=self.rng,
                    )
                )
        return batch

    def collect_policy_batch_from_summaries(
        self,
        key_windows: Sequence[List[List[float]]],
        sources: Sequence[str],
        summaries: Sequence[str],
        references: Optional[Sequence[Optional[str]]] = None,
    ) -> List[PolicySample]:
        """Single-summary convenience wrapper (for PPO baseline)."""
        n = len(key_windows)
        if len(sources) != n or len(summaries) != n:
            raise ValueError("key_windows, sources, and summaries must match in length.")
        if references is not None and len(references) != n:
            raise ValueError("references must match key_windows length.")

        components: List[RewardComponents] = []
        for i in range(n):
            ref = references[i] if references is not None else None
            components.append(
                self.reward_scorer.score(source=sources[i], summary=summaries[i], reference=ref)
            )
        return self.collect_policy_batch(key_windows=key_windows, reward_components=components)

    # ------------------------------------------------------------------
    # Policy update steps
    # ------------------------------------------------------------------

    def ppo_step(self, batch: List[PolicySample]) -> UpdateStats:
        """PPO update with standardised advantages (baseline comparison)."""
        rewards = [s.reward for s in batch]
        if not rewards:
            return UpdateStats(0.0, 0.0, 0.0, 0.0, 0.0)

        r_arr = np.array(rewards, dtype=np.float64)
        advantages = ((r_arr - np.mean(r_arr)) / (np.std(r_arr) + 1e-8)).tolist()

        stats = self.memory_manager.policy.ppo_update(
            batch=batch,
            advantages=advantages,
            lr=self._current_lr(),
            clip_eps=self.config.clip_eps,
            entropy_coef=self.config.entropy_coef,
            max_kl=self.config.max_kl,
        )
        self._step += 1
        return stats

    def grpo_step(self, batch: List[PolicySample]) -> UpdateStats:
        """GRPO update (primary algorithm): within-group advantage normalisation."""
        stats = self.memory_manager.policy.grpo_update(
            batch=batch,
            lr=self._current_lr(),
            clip_eps=self.config.clip_eps,
            entropy_coef=self.config.entropy_coef,
            max_kl=self.config.max_kl,
        )
        self._step += 1
        return stats
