"""
Linalg lowering agent — proposes a shortlist of XeGPU lowering configs.

This is the "LLM shortlist" half of the hybrid tile-search: given a Linalg
matmul, the problem shape, and the knowledge-base config patterns/constraints,
the LLM proposes a handful of promising LoweringConfigs. The pipeline then
sweeps (lowers + runs + times) only those, and the CoVeR gate keeps the fastest
correct one. Invalid or shape-incompatible proposals are filtered before the
sweep, and a known-good default is always appended as a safety net.
"""

from __future__ import annotations

import logging

import dspy
from pydantic import BaseModel, Field

from xe_forge.core.linalg_lowering import DEFAULT_CONFIG, LoweringConfig

logger = logging.getLogger(__name__)


class ConfigProposal(BaseModel):
    """One proposed lowering configuration."""

    wg_m: int = Field(description="Workgroup tile M (e.g. 128, 256)")
    wg_n: int = Field(description="Workgroup tile N (e.g. 128, 256)")
    sg_m: int = Field(description="Subgroup tile M, multiple of 8 (e.g. 32)")
    sg_n: int = Field(description="Subgroup tile N, multiple of 16 (e.g. 32)")
    k_tile: int = Field(description="K-loop tile, multiple of 16 (e.g. 32, 64)")
    rationale: str = Field(default="", description="Why this config suits the shape")


class LoweringConfigSignature(dspy.Signature):
    """Propose XeGPU workgroup lowering configs for a Linalg matmul on Intel Xe.

    You are an expert in Intel Xe GPU performance and XeGPU workgroup tiling.
    Given a matmul and its M/N/K shape, propose 3-5 promising lowering configs.

    A config is (wg_m, wg_n, sg_m, sg_n, k_tile):
      - wg_m/wg_n: the workgroup output tile.
      - sg_m/sg_n: the per-subgroup tile; subgroup grid = [wg_m/sg_m, wg_n/sg_n].
      - k_tile: the k-loop step.

    HARD CONSTRAINTS (configs violating these are discarded):
      - wg_m % sg_m == 0 and wg_n % sg_n == 0
      - sg_m % 8 == 0, sg_n % 16 == 0, k_tile % 16 == 0   (DPAS alignment)
      - (wg_m/sg_m) * (wg_n/sg_n) * 16 <= 1024            (thread budget)
      - M % wg_m == 0, N % wg_n == 0, K % k_tile == 0     (fits the shape)

    GUIDANCE:
      - Large square tiles (256x256, sg 32x32) maximize DPAS reuse on big,
        compute-bound GEMMs.
      - Smaller / skinny tiles (128x256, 128x128) raise occupancy and fit
        shapes not divisible by 256, or relieve register pressure.
      - Larger k_tile (64) amortizes load latency on large-K, memory-bound cases.
    Propose a spread that brackets the likely optimum, not near-duplicates.
    """

    kernel_code: dspy.Code["mlir"] = dspy.InputField(desc="The Linalg matmul kernel.")
    problem_shape: str = dspy.InputField(desc="M, N, K dimensions and dtype.")
    knowledge_base_context: str = dspy.InputField(
        desc="KB config patterns + constraints. Empty if KB disabled."
    )
    configs: list[ConfigProposal] = dspy.OutputField(
        desc="3-5 distinct, constraint-satisfying lowering configs to try."
    )


class LinalgLoweringAgent:
    """Proposes a validated, shape-compatible shortlist of LoweringConfigs."""

    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base
        self.predictor = dspy.Predict(LoweringConfigSignature)

    def _kb_context(self) -> str:
        if self.knowledge_base is None:
            return ""
        try:
            from xe_forge.models import OptimizationStage

            return self.knowledge_base.format_for_stage(OptimizationStage.DEVICE_SPECIFIC)
        except Exception:
            return ""

    def propose(
        self,
        kernel_code: str,
        m: int,
        n: int,
        k: int,
        dtype: str = "f16",
        max_configs: int = 5,
    ) -> list[LoweringConfig]:
        """Return validated, shape-fitting configs (LLM shortlist + default)."""
        shortlist: list[LoweringConfig] = []
        try:
            pred = self.predictor(
                kernel_code=kernel_code,
                problem_shape=f"M={m}, N={n}, K={k}, dtype={dtype}",
                knowledge_base_context=self._kb_context(),
            )
            for p in pred.configs or []:
                cfg = LoweringConfig(p.wg_m, p.wg_n, p.sg_m, p.sg_n, p.k_tile)
                if cfg.validate():
                    logger.info("discarding invalid proposed config %s", cfg)
                    continue
                if cfg.fits_shape(m, n, k):
                    logger.info("discarding shape-incompatible config %s", cfg)
                    continue
                if cfg not in shortlist:
                    shortlist.append(cfg)
        except Exception as e:
            logger.warning("Lowering-config proposal failed (%s); using default only", e)

        # Safety net: always include a known-good default if it fits the shape.
        if not DEFAULT_CONFIG.fits_shape(m, n, k) and DEFAULT_CONFIG not in shortlist:
            shortlist.append(DEFAULT_CONFIG)
        if not shortlist:
            # Nothing fit — fall back to the smallest safe tile that divides the shape.
            shortlist = self._fallback_configs(m, n, k)
        return shortlist[:max_configs]

    @staticmethod
    def _fallback_configs(m: int, n: int, k: int) -> list[LoweringConfig]:
        """Deterministic divisor-based configs when the LLM/default don't fit."""
        out: list[LoweringConfig] = []
        for wg in (256, 128, 64):
            for kt in (32, 16):
                cfg = LoweringConfig(wg, wg, 32, 32, kt)
                if not cfg.validate() and not cfg.fits_shape(m, n, k) and cfg not in out:
                    out.append(cfg)
        return out
