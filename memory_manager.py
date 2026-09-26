from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from policy_manager import GatingPolicy


class ReMemorizeMemoryManager(nn.Module):
    """
    GRU-style gated recurrent memory for adaptive clinical summarization.

    Maintains a single memory vector  m_t ∈ ℝ^{d_m}  that evolves across
    both source encoding AND summary decoding — preserving the core thesis
    that hallucination is a failure of adaptive memory dynamics *during*
    generation, not only during source reading.

    ── Update rule (identical at every step, source or decoding) ─────────

        c_t  = tanh( write_proj(v_t) )             candidate write content
        γ_t  = σ( MLP([ k_t ; proj(m_{t-1}) ]) )  scalar gate ∈ (0, 1)
        m_t  = (1 − γ_t) · m_{t-1}  +  γ_t · c_t  gated update

    The gate is memory-conditioned: content already represented in m_{t-1}
    suppresses γ_t, giving an information-theoretic compression signal.

    ── Two-phase lifecycle (driven by llm_backbone.py) ──────────────────

    Phase 1 — Source encoding:
        reset_state() → update(source_keys, source_values) → m_{T_src}

    Phase 2 — Decoding (per generated token t):
        1. backbone injects m_{t-1} into decoder hidden state h_t
        2. backbone samples next token from augmented logits
        3. backbone calls _gru_step(val_proj(h_t), γ_t) → m_t

    ── Engineering safeguards ────────────────────────────────────────────
    · γ gate bias initialised to −2.0 inside GatingPolicy
      → sigmoid(−2) ≈ 0.12: small initial writes, stable warm-up.
    · _gru_step(detach=True) detaches v_t before write_proj, preventing
      runaway feedback through the memory path during early training.
    · update() returns full γ statistics + memory_norm for monitoring.
    """

    def __init__(
        self,
        d_key: int,
        d_value: Optional[int] = None,
        d_memory: Optional[int] = None,
        seed: int = 0,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        d_value = d_value if d_value is not None else d_key
        d_mem   = d_memory if d_memory is not None else d_key

        self.d_key    = d_key
        self.d_value  = d_value
        self.d_memory = d_mem
        self.device   = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Write projection: v_t → candidate write content
        self.write_proj = nn.Linear(d_value, d_mem)
        nn.init.normal_(self.write_proj.weight, std=0.02)
        nn.init.zeros_(self.write_proj.bias)

        # Gating policy — γ bias initialised to -2.0 inside GatingPolicy (safeguard 3)
        self.policy = GatingPolicy(
            d_key=d_key, d_mem=d_mem, seed=seed + 1, device=device
        )

        # Memory state: analytic buffer, NOT a learnable parameter
        self.register_buffer("m", torch.zeros(d_mem, dtype=torch.float32))

        self.to(self.device)
        self.trajectory: List[Dict] = []

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset memory to zero and clear trajectory. Call before each new document."""
        self.m.zero_()
        self.trajectory.clear()

    def memory_norm(self) -> float:
        """L2 norm of current memory state — used for explosion monitoring."""
        return float(self.m.norm())

    # ------------------------------------------------------------------
    # Core GRU step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _gru_step(
        self,
        v_t: torch.Tensor,   # (d_value,) float32
        gamma: float,        # scalar gate in [0, 1]
        detach: bool = False,
    ) -> None:
        """
        Single GRU-style memory update:
            c_t = tanh( write_proj(v_t) )
            m   ← (1 − γ) · m  +  γ · c_t

        detach=True detaches v_t before write_proj (safeguard 2): use during
        early training / decoding to prevent runaway feedback loops.
        """
        v = v_t.detach() if detach else v_t
        c_t = torch.tanh(self.write_proj(v))      # (d_mem,)
        self.m.mul_(1.0 - gamma).add_(c_t.mul(gamma))

    # ------------------------------------------------------------------
    # Sequence update  (source encoding + GRPO rollout simulation)
    # ------------------------------------------------------------------

    def update(
        self,
        keys:   List[List[float]],
        values: List[List[float]],
        gammas: Optional[List[float]] = None,
        detach: bool = False,
    ) -> Dict:
        """
        Process a full token sequence online.

        For each token i:
          1. Snapshot m_{i-1} as the readout (stored for PolicySample).
          2. Measure reconstruction error ||m_{i-1} − c_i||² (dense reward signal).
          3. Query gating policy for γ_i.
          4. GRU update → m_i.

        Returns γ statistics and memory_norm for training monitoring (safeguard 4).
        """
        if not keys or not values:
            raise ValueError("keys and values cannot be empty.")
        if len(keys) != len(values):
            raise ValueError("keys and values must have the same length.")
        if len(keys[0]) != self.d_key:
            raise ValueError(
                f"Key dim mismatch: expected {self.d_key}, got {len(keys[0])}."
            )
        if len(values[0]) != self.d_value:
            raise ValueError(
                f"Value dim mismatch: expected {self.d_value}, got {len(values[0])}."
            )

        V = torch.tensor(values, dtype=torch.float32, device=self.device)
        n = len(keys)

        gammas_used:          List[float]       = []
        readouts_stored:      List[List[float]] = []
        err_sq_sum          = 0.0
        weighted_err_sq_sum = 0.0

        with torch.no_grad():
            C = torch.tanh(self.write_proj(V))     # (T, d_mem) — batch candidates

        for i in range(n):
            m_prev = self.m.clone()
            readouts_stored.append(m_prev.tolist())

            err = m_prev - C[i]
            err_sq_i = float((err * err).mean())
            err_sq_sum += err_sq_i

            if gammas is not None:
                gamma_i = float(max(0.0, min(1.0, gammas[i])))
            else:
                gamma_i = float(
                    self.policy([keys[i]], [m_prev.tolist()])[0]
                )
            weighted_err_sq_sum += gamma_i * err_sq_i
            gammas_used.append(gamma_i)

            self._gru_step(V[i], gamma_i, detach=detach)

        mse          = err_sq_sum / max(n, 1)
        weighted_mse = weighted_err_sq_sum / max(n, 1)

        g = torch.tensor(gammas_used, dtype=torch.float32)
        self.trajectory.append({
            "keys": keys, "values": values,
            "gammas": gammas_used, "readouts": readouts_stored,
        })

        return {
            "window_tokens":               float(n),
            "gamma_mean":                  float(g.mean()),
            "gamma_std":                   float(g.std()) if n > 1 else 0.0,
            "gamma_min":                   float(g.min()),
            "gamma_max":                   float(g.max()),
            "reconstruction_mse":          mse,
            "weighted_reconstruction_mse": weighted_mse,
            "memory_norm":                 self.memory_norm(),
            "readouts":                    readouts_stored,
        }

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self) -> List[float]:
        """
        Return current memory state m as a Python list.

        The backbone calls this at each decoding step to obtain m_{t-1}
        for injection, then updates m_t via _gru_step after sampling.
        """
        return self.m.tolist()
