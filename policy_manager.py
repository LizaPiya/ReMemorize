from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW


@dataclass
class PolicySample:
    key_window: List[List[float]]
    memory_readouts: List[List[float]]   # m_{t-1} snapshots (d_mem,) at rollout time
    actions: List[int]
    old_logprob: float
    reward: float
    group_id: int = 0


@dataclass
class UpdateStats:
    loss: float
    surrogate: float
    entropy: float
    approx_kl: float
    clip_fraction: float


class GatingPolicy(nn.Module):
    """
    Memory-conditioned Bernoulli token gating policy.

    For each token at position i the gate probability is:
        p_i = sigmoid(MLP([k_i ; proj(m_{i-1})]))
    where m_{i-1} ∈ ℝ^{d_mem} is the current GRU memory state.
    This creates an information-theoretic gate: tokens already
    well-represented in memory receive lower gate weights.

    The output bias of gate_mlp is initialised to -2.0 so that
    sigmoid(-2) ≈ 0.12 at the start of training, preventing
    aggressive early memory overwriting (safeguard 3).
    """

    def __init__(
        self,
        d_key: int,
        d_mem: int,
        d_hidden: int = 256,
        seed: int = 0,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Project memory state down to d_key dims before concatenation.
        # Keeps MLP input size independent of d_mem.
        self.readout_proj = nn.Linear(d_mem, d_key, bias=False)

        # 2-layer MLP: input = [k_i ; proj(m_{i-1})]  (dim = 2 * d_key)
        gate_dim = d_key * 2
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

        self._init_weights(seed)
        # Bias toward small initial γ: sigmoid(-2.0) ≈ 0.12  (safeguard 3)
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)
        self.to(self.device)
        self.optimizer = AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)

    def _init_weights(self, seed: int) -> None:
        rng = torch.Generator()
        rng.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02, generator=rng)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def probabilities(
        self,
        key_window: torch.Tensor,        # (T, d_key)  float32, on device
        memory_readouts: torch.Tensor,   # (T, d_value) float32, on device
    ) -> torch.Tensor:                   # (T,) in [0, 1]
        proj = self.readout_proj(memory_readouts)            # (T, d_key)
        inp = torch.cat([key_window, proj], dim=-1)          # (T, 2*d_key)
        logits = self.gate_mlp(inp).squeeze(-1)              # (T,)
        return torch.sigmoid(logits)

    def __call__(
        self,
        key_window: List[List[float]],
        memory_readouts: List[List[float]],
    ) -> List[float]:
        """Bridge for memory_manager: returns List[float] gate probs."""
        K = torch.tensor(key_window, dtype=torch.float32, device=self.device)
        R = torch.tensor(memory_readouts, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = self.probabilities(K, R)
        return probs.tolist()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_actions(
        self,
        key_window: List[List[float]],
        memory_readouts: List[List[float]],
        rng: Optional[Random] = None,
    ) -> Tuple[List[int], List[float], float]:
        K = torch.tensor(key_window, dtype=torch.float32, device=self.device)
        R = torch.tensor(memory_readouts, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs_t = self.probabilities(K, R)
        probs = probs_t.tolist()
        r = rng or Random()
        actions = [1 if r.random() < p else 0 for p in probs]
        logp = self.logprob_of_actions(key_window, memory_readouts, actions)
        return actions, probs, logp

    def logprob_of_actions(
        self,
        key_window: List[List[float]],
        memory_readouts: List[List[float]],
        actions: List[int],
    ) -> float:
        K = torch.tensor(key_window, dtype=torch.float32, device=self.device)
        R = torch.tensor(memory_readouts, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs_t = self.probabilities(K, R)
        eps = 1e-8
        pc = probs_t.clamp(eps, 1.0 - eps)
        a_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        lp = a_t * pc.log() + (1.0 - a_t) * (1.0 - pc).log()
        return float(lp.sum())

    def _entropy(self, probs_t: torch.Tensor) -> float:
        eps = 1e-8
        p = probs_t.clamp(eps, 1.0 - eps)
        h = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        return float(h.sum())

    # ------------------------------------------------------------------
    # PPO update (used as baseline comparison)
    # ------------------------------------------------------------------

    def ppo_update(
        self,
        batch: List[PolicySample],
        advantages: List[float],
        lr: float = 1e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        max_kl: float = 0.02,          # KL early stopping threshold
        ppo_epochs: int = 1,           # gradient steps per batch
    ) -> UpdateStats:
        if not batch:
            return UpdateStats(0.0, 0.0, 0.0, 0.0, 0.0)
        if len(batch) != len(advantages):
            raise ValueError("batch and advantages must have same length.")

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        n = len(batch)
        surrogate_sum = 0.0
        entropy_sum = 0.0
        kl_sum = 0.0
        clipped = 0.0

        for _ in range(ppo_epochs):
            self.optimizer.zero_grad()
            total_loss = torch.zeros(1, device=self.device)
            epoch_kl = 0.0

            for sample, adv in zip(batch, advantages):
                K = torch.tensor(sample.key_window, dtype=torch.float32, device=self.device)
                R = torch.tensor(sample.memory_readouts, dtype=torch.float32, device=self.device)

                probs_t = self.probabilities(K, R)
                eps = 1e-8
                pc = probs_t.clamp(eps, 1.0 - eps)
                a_t = torch.tensor(sample.actions, dtype=torch.float32, device=self.device)

                logp_new_t = (a_t * pc.log() + (1.0 - a_t) * (1.0 - pc).log()).sum()
                logp_new = logp_new_t.item()

                ratio = torch.exp(logp_new_t - float(sample.old_logprob))
                adv_t = torch.tensor(float(adv), device=self.device)
                s1 = ratio * adv_t
                s2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv_t
                surrogate_t = torch.min(s1, s2)

                h_t = -(pc * pc.log() + (1.0 - pc) * (1.0 - pc).log()).sum()
                total_loss = total_loss + (-surrogate_t - entropy_coef * h_t)

                surrogate_sum += surrogate_t.item()
                entropy_sum += self._entropy(probs_t.detach())
                approx_kl = max(0.0, float(sample.old_logprob) - logp_new)
                kl_sum += approx_kl
                epoch_kl += approx_kl

                ratio_val = ratio.item()
                if (adv >= 0.0 and ratio_val > 1.0 + clip_eps) or (
                    adv < 0.0 and ratio_val < 1.0 - clip_eps
                ):
                    clipped += 1.0

            (total_loss / n).backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()

            # KL early stopping
            if epoch_kl / n > max_kl:
                break

        mean_surrogate = surrogate_sum / n
        mean_entropy = entropy_sum / n
        loss = -mean_surrogate - entropy_coef * mean_entropy
        return UpdateStats(
            loss=loss,
            surrogate=mean_surrogate,
            entropy=mean_entropy,
            approx_kl=kl_sum / n,
            clip_fraction=clipped / n,
        )

    # ------------------------------------------------------------------
    # GRPO update (primary algorithm)
    # ------------------------------------------------------------------

    def grpo_update(
        self,
        batch: List[PolicySample],
        lr: float = 1e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        group_eps: float = 1e-6,
        max_kl: float = 0.02,
    ) -> UpdateStats:
        """
        Group Relative Policy Optimization.
        Normalizes advantages within groups (same document / same prompt),
        requiring no value function. Primary training algorithm.
        """
        grouped: Dict[int, List[float]] = {}
        for s in batch:
            grouped.setdefault(s.group_id, []).append(s.reward)

        advantages: List[float] = []
        for s in batch:
            rewards = grouped[s.group_id]
            mean_r = sum(rewards) / max(len(rewards), 1)
            var_r = sum((x - mean_r) ** 2 for x in rewards) / max(len(rewards), 1)
            std_r = float(torch.tensor(max(var_r, group_eps)).sqrt())
            advantages.append((s.reward - mean_r) / std_r)

        return self.ppo_update(
            batch=batch,
            advantages=advantages,
            lr=lr,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            max_kl=max_kl,
        )


def build_policy_sample(
    policy: GatingPolicy,
    key_window: List[List[float]],
    memory_readouts: List[List[float]],
    reward: float,
    group_id: int = 0,
    rng: Optional[Random] = None,
) -> PolicySample:
    actions, _, old_logprob = policy.sample_actions(key_window, memory_readouts, rng=rng)
    return PolicySample(
        key_window=key_window,
        memory_readouts=memory_readouts,
        actions=actions,
        old_logprob=old_logprob,
        reward=reward,
        group_id=group_id,
    )
