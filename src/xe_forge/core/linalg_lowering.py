"""
Linalg -> XeGPU WG lowering config: tunable tile/subgroup parameters.

A LoweringConfig captures the 5 free knobs of the Linalg->WG lowering. Every
XeGPU #xegpu.layout in the pipeline is *derived* from these plus the fixed DPAS
instruction shapes, so the optimizer only ever picks the 5 knobs and can never
produce an internally inconsistent layout set. Invalid combinations (bad
divisibility / DPAS alignment) are rejected up front by `validate()`; anything
that passes validation but is still wrong for a given kernel shape is caught by
the CoVeR verify gate (fails to lower, or [ALLCLOSE: FALSE]).

Renders the two transform-dialect libraries used by pipelines/linalg_to_wg via
Jinja2, matching the repo's existing template approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

# Fixed DPAS (XMX) instruction tile shapes on Intel Xe — not tunable.
DPAS_A_TILE = (8, 16)
DPAS_B_TILE = (16, 16)
DPAS_C_TILE = (8, 16)
NB_WORKITEMS = 16  # subgroup (SIMD) width

_TEMPLATE_DIR = (
    Path(__file__).resolve().parents[3]
    / "pipelines"
    / "linalg_to_wg"
    / "templates"
)


@dataclass(frozen=True)
class LoweringConfig:
    """The 5 free knobs of the Linalg->XeGPU-WG lowering."""

    wg_m: int  # workgroup tile M
    wg_n: int  # workgroup tile N
    sg_m: int  # subgroup tile M
    sg_n: int  # subgroup tile N
    k_tile: int  # k-loop tile

    # ---- derived quantities -------------------------------------------------
    @property
    def sg_grid_m(self) -> int:
        return self.wg_m // self.sg_m

    @property
    def sg_grid_n(self) -> int:
        return self.wg_n // self.sg_n

    @property
    def nb_threads(self) -> int:
        """Threads per workgroup = number of subgroups * SIMD width."""
        return self.sg_grid_m * self.sg_grid_n * NB_WORKITEMS

    # ---- validity -----------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of constraint violations (empty == valid)."""
        errs: list[str] = []
        if self.wg_m % self.sg_m:
            errs.append(f"wg_m {self.wg_m} not divisible by sg_m {self.sg_m}")
        if self.wg_n % self.sg_n:
            errs.append(f"wg_n {self.wg_n} not divisible by sg_n {self.sg_n}")
        # DPAS alignment: subgroup tile must be a multiple of the DPAS shape.
        if self.sg_m % DPAS_A_TILE[0]:
            errs.append(f"sg_m {self.sg_m} not a multiple of DPAS M {DPAS_A_TILE[0]}")
        if self.sg_n % DPAS_B_TILE[1]:
            errs.append(f"sg_n {self.sg_n} not a multiple of DPAS N {DPAS_B_TILE[1]}")
        if self.k_tile % DPAS_A_TILE[1]:
            errs.append(f"k_tile {self.k_tile} not a multiple of DPAS K {DPAS_A_TILE[1]}")
        # Hardware ceiling: 1024 work-items per workgroup on Xe.
        if self.nb_threads > 1024:
            errs.append(f"nb_threads {self.nb_threads} exceeds 1024 (too many subgroups)")
        if self.nb_threads == 0:
            errs.append("nb_threads is 0 (sg tile larger than wg tile)")
        return errs

    @property
    def is_valid(self) -> bool:
        return not self.validate()

    def fits_shape(self, m: int, n: int, k: int) -> list[str]:
        """Additional checks against a concrete problem shape (M, N, K)."""
        errs = []
        if m % self.wg_m:
            errs.append(f"M {m} not divisible by wg_m {self.wg_m}")
        if n % self.wg_n:
            errs.append(f"N {n} not divisible by wg_n {self.wg_n}")
        if k % self.k_tile:
            errs.append(f"K {k} not divisible by k_tile {self.k_tile}")
        return errs

    # ---- rendering ----------------------------------------------------------
    def _ctx(self) -> dict:
        return {
            "wg_m": self.wg_m,
            "wg_n": self.wg_n,
            "sg_m": self.sg_m,
            "sg_n": self.sg_n,
            "k_tile": self.k_tile,
            "sg_grid_m": self.sg_grid_m,
            "sg_grid_n": self.sg_grid_n,
            "nb_threads": self.nb_threads,
        }

    def render(self, out_dir: str | Path, template_dir: str | Path = _TEMPLATE_DIR) -> tuple[Path, Path]:
        """Render both transform libraries into out_dir. Returns their paths."""
        errs = self.validate()
        if errs:
            raise ValueError(f"invalid LoweringConfig: {'; '.join(errs)}")
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ctx = self._ctx()
        tile_path = out_dir / "tile_vectorize.mlir"
        anno_path = out_dir / "wg_annotate.mlir"
        tile_path.write_text(env.get_template("tile_vectorize.mlir.j2").render(**ctx))
        anno_path.write_text(env.get_template("wg_annotate.mlir.j2").render(**ctx))
        return tile_path, anno_path


def render_timing_harness(
    config: LoweringConfig,
    m: int,
    n: int,
    k: int,
    kernel_name: str = "test_kernel",
    template_dir: str | Path = _TEMPLATE_DIR,
) -> str:
    """Render the kernel-only rtclock timing harness for *config* at shape MxNxK.

    Grid = [M/wg_m, N/wg_n], threads = nb_threads (from the config). The result
    has a ``// KERNEL`` slot (splice the lowered gpu.module) and an ``// ALIASES``
    slot (hoisted affine-map defs), consumed by MlirExecutor._splice_kernel.
    """
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template("timing_harness.mlir.j2").render(
        m=m,
        n=n,
        k=k,
        grid_m=m // config.wg_m,
        grid_n=n // config.wg_n,
        nb_threads=config.nb_threads,
        kernel_name=kernel_name,
    )


# The config proven end-to-end in Phase 0 (512^3, [ALLCLOSE: TRUE]).
DEFAULT_CONFIG = LoweringConfig(wg_m=256, wg_n=256, sg_m=32, sg_n=32, k_tile=32)


def detect_mlir_level(code: str) -> str:
    """Classify an MLIR kernel as 'linalg' or 'xegpu_wg'.

    'linalg' -> a high-level kernel that still needs the Linalg->WG lowering.
    'xegpu_wg' -> already at XeGPU workgroup level (create_nd_tdesc / dpas), the
    form the existing WG stages operate on directly.
    """
    if "xegpu." in code or "gpu.launch_func" in code:
        return "xegpu_wg"
    if "linalg." in code:
        return "linalg"
    # Default to WG so unknown inputs go through the existing (safer) path.
    return "xegpu_wg"


def extract_matmul_dims(code: str) -> tuple[int, int, int] | None:
    """Best-effort (M, N, K) from a linalg.matmul's operand shapes.

    Parses `linalg.matmul ins(%a, %b : tensor<MxKxTy>, tensor<KxNxTy>)`.
    Returns None if it cannot be determined.
    """
    import re

    m = re.search(
        r"linalg\.matmul\s+ins\([^:]*:\s*tensor<(\d+)x(\d+)x[^,>]+>,\s*tensor<(\d+)x(\d+)x[^>]+>",
        code,
    )
    if not m:
        return None
    a0, a1, b0, b1 = (int(g) for g in m.groups())
    # A is MxK, B is KxN
    return a0, b1, a1
