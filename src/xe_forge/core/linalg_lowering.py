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
MAX_SUBGROUPS = 64  # default (small-GRF) ceiling: 1024 work-items / workgroup
# Intel Xe (BMG) large-register-file mode: doubles per-thread GRF but halves the
# subgroup budget to 32/workgroup. Enabled at lowering time via the igc option
# below; helps register-pressure-bound kernels (large tiles) at the cost of
# occupancy. See LoweringConfig.large_grf.
MAX_SUBGROUPS_LARGE_GRF = 32
LARGE_GRF_IGC_OPTION = "igc-cmd-options=-ze-opt-large-register-file"

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
    large_grf: bool = False  # BMG large register file (<=32 subgroups; igc option)

    # ---- derived quantities -------------------------------------------------
    @property
    def sg_grid_m(self) -> int:
        return self.wg_m // self.sg_m

    @property
    def sg_grid_n(self) -> int:
        return self.wg_n // self.sg_n

    @property
    def num_subgroups(self) -> int:
        """Subgroups per workgroup = sg_grid_m * sg_grid_n."""
        return self.sg_grid_m * self.sg_grid_n

    @property
    def nb_threads(self) -> int:
        """Threads per workgroup = number of subgroups * SIMD width."""
        return self.num_subgroups * NB_WORKITEMS

    @property
    def max_subgroups(self) -> int:
        """Subgroup ceiling: 32 in large-GRF mode, else 64."""
        return MAX_SUBGROUPS_LARGE_GRF if self.large_grf else MAX_SUBGROUPS

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
        # Subgroup ceiling depends on GRF mode: 32 (large GRF) or 64 (default).
        if self.num_subgroups > self.max_subgroups:
            mode = "large-GRF" if self.large_grf else "default"
            errs.append(
                f"{self.num_subgroups} subgroups exceeds {self.max_subgroups} "
                f"({mode} mode); reduce tile or enlarge sg tile"
            )
        if self.num_subgroups == 0:
            errs.append("0 subgroups (sg tile larger than wg tile)")
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

    # ---- run-time lowering options -----------------------------------------
    def run_pipeline_options(self) -> str:
        """Options string for --gpu-lower-to-xevm-pipeline used to RUN this config.

        Large-GRF mode is a compile-time (igc) flag applied when the final kernel
        is lowered+run, not a change to the tiling/layout transforms. Returns e.g.
        ``xegpu-op-level=workgroup`` or, with large_grf,
        ``xegpu-op-level=workgroup igc-cmd-options=-ze-opt-large-register-file``.
        """
        opts = "xegpu-op-level=workgroup"
        if self.large_grf:
            opts += " " + LARGE_GRF_IGC_OPTION
        return opts

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

    def render(
        self,
        out_dir: str | Path,
        template_dir: str | Path = _TEMPLATE_DIR,
        batched: bool = False,
    ) -> tuple[Path, Path]:
        """Render both transform libraries into out_dir. Returns their paths.

        *batched* selects the batched-matmul stage-1 template
        (tile_vectorize_batch.mlir.j2: one batch per workgroup + rank-reduce +
        cast-away-leading-unit-dim) instead of the plain-matmul one. The stage-3
        layout-annotation library (wg_annotate) is shared: after rank-reduction the
        batched kernel has the same 2-D xegpu.dpas/load_nd/store_nd shape.
        """
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
        tile_tmpl = "tile_vectorize_batch.mlir.j2" if batched else "tile_vectorize.mlir.j2"
        tile_path.write_text(env.get_template(tile_tmpl).render(**ctx))
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


def render_profiling_harness(
    config: LoweringConfig,
    m: int,
    n: int,
    k: int,
    kernel_name: str = "test_kernel",
    template_dir: str | Path = _TEMPLATE_DIR,
) -> str:
    """Render the single-launch IMEX-profiling harness for *config* at MxNxK.

    Unlike the rtclock timing harness (which loops the launch on the host), this
    emits ONE gpu.launch_func; IMEX level-zero profiling (env-enabled) does the
    warmup/timed loops in the runtime and prints "Median: <ms>". Requires a
    launch-idempotent kernel (zero-init accumulator). Same // KERNEL / // ALIASES
    splice slots as the timing harness.
    """
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    return env.get_template("profiling_harness.mlir.j2").render(
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


def extract_batch_matmul_dims(code: str) -> tuple[int, int, int, int] | None:
    """Best-effort (batch, M, N, K) from a linalg.batch_matmul's operand shapes.

    Parses `linalg.batch_matmul ins(%a, %b : tensor<GxMxKxTy>, tensor<GxKxNxTy>)`.
    Returns None if it cannot be determined. Requires both operands to share the
    batch size G (a plain 3-D batched matmul).
    """
    import re

    m = re.search(
        r"linalg\.batch_matmul\s+ins\([^:]*:\s*"
        r"tensor<(\d+)x(\d+)x(\d+)x[^,>]+>,\s*tensor<(\d+)x(\d+)x(\d+)x[^>]+>",
        code,
    )
    if not m:
        return None
    ga, am, ak, gb, bk, bn = (int(g) for g in m.groups())
    if ga != gb or ak != bk:
        return None  # not a well-formed batched matmul (batch/K must agree)
    # A is G x M x K, B is G x K x N
    return ga, am, bn, ak
