import logging
import os
from datetime import datetime
from pathlib import Path

import dspy
import httpx
import litellm

from xe_forge.agents import AnalyzerAgent, Optimizer, OptimizerAgent, OptimizerReActAgent
from xe_forge.config import Config, get_config
from xe_forge.core.device_query import get_device_config_for_pipeline
from xe_forge.knowledge.loader import KnowledgeBase, load_knowledge_base
from xe_forge.models import (
    DSL,
    IssueType,
    OptimizationResult,
    OptimizationStage,
    StageResult,
)
from xe_forge.planner import DEFAULT_STAGE_ORDER as PLANNER_DEFAULT_STAGE_ORDER
from xe_forge.planner import PlannerAgent

logger = logging.getLogger(__name__)


def _extract_gemm_dims(
    input_shapes: list[tuple[int, ...]] | None,
) -> tuple[int, int, int]:
    """Extract M, N, K from GEMM input shapes [(M, K), (K, N)]."""
    if input_shapes and len(input_shapes) >= 2:
        a, b = input_shapes[0], input_shapes[1]
        if len(a) >= 2 and len(b) >= 2:
            return a[-2], b[-1], a[-1]
    return 1024, 1024, 1024


def _split_linalg_harness(code: str) -> tuple[str, str | None]:
    """Split a self-contained Linalg file into (bare compute kernel, harness).

    Expects a module containing the compute ``func.func`` (with ``linalg.matmul``)
    plus a ``@main`` harness and helper decls. Returns:
      - bare: just the compute func (for the lowering pipeline), and
      - harness: the enclosing module with the compute func replaced by a
        ``// KERNEL`` marker and an ``// ALIASES`` slot before the module, so the
        lowered WG kernel can be spliced back in (see MlirExecutor._splice_kernel).
    Returns (code, None) if the expected structure is not found.
    """
    import re

    # Find the compute func (the one containing linalg.matmul).
    funcs = list(re.finditer(r"\n?\s*func\.func\s+@(\w+)\s*\(", code))
    compute_span = None
    compute_name = None
    for mt in funcs:
        # brace-match this func's body
        start = mt.start()
        b = code.find("{", mt.end() - 1)
        depth = 0
        j = b
        while j < len(code):
            if code[j] == "{":
                depth += 1
            elif code[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        segment = code[start : j + 1]
        if "linalg.matmul" in segment:
            compute_span = (start, j + 1)
            compute_name = mt.group(1)
            bare = segment.strip()
            break
    if compute_span is None or compute_name is None:
        return code, None

    # Build the harness: same file, compute func -> "// KERNEL".
    body = code[: compute_span[0]] + "\n  // KERNEL\n" + code[compute_span[1] :]
    # The spliced gpu.launch_func requires the enclosing module to carry the
    # gpu.container_module attribute. If the input has no explicit attributed
    # module wrapper, wrap the body in one; otherwise ensure the attribute is set.
    if "gpu.container_module" not in body:
        if body.lstrip().startswith("module"):
            # Has a module but lacks the attribute — add it.
            body = body.replace("module {", "module attributes {gpu.container_module} {", 1)
        else:
            # Bare top-level funcs — wrap them in a container module.
            body = "module attributes {gpu.container_module} {\n" + body + "\n}\n"
    # "// ALIASES" slot at the very top so hoisted affine-map aliases land at
    # module-attribute scope (before the module keyword).
    harness = "// ALIASES\n" + body
    return bare, harness


DEFAULT_STAGE_ORDER: list[OptimizationStage] = [
    OptimizationStage.ANALYSIS,
    OptimizationStage.ALGORITHMIC,
    OptimizationStage.DISCOVERY,
    OptimizationStage.DTYPE_FIX,
    OptimizationStage.FUSION,
    OptimizationStage.MEMORY_ACCESS,
    OptimizationStage.BLOCK_POINTERS,
    OptimizationStage.PERSISTENT_KERNEL,
    OptimizationStage.DEVICE_SPECIFIC,
    OptimizationStage.AUTOTUNING,
]


class XeForgePipeline:
    config: Config
    analyzer: AnalyzerAgent
    optimizer: Optimizer

    def __init__(
        self, config=None, executor=None, validator=None, trial_manager=None, profiler=None
    ):
        self.config = config or get_config()
        self.trial_manager = trial_manager
        self.profiler = profiler
        self._setup_logging()
        self._setup_llm()

        if executor is None:
            if self.config.device_config.dsl == DSL.SYCL:
                from xe_forge.core import SyclExecutor

                executor = SyclExecutor(
                    verify=self.config.optimization.require_correctness,
                )
            elif self.config.device_config.dsl == DSL.MLIR:
                from xe_forge.core.mlir_executor import MlirExecutor

                executor = MlirExecutor(
                    require_correctness=self.config.optimization.require_correctness,
                )
            else:
                from xe_forge.core import KernelBenchExecutor

                executor = KernelBenchExecutor(
                    device=self.config.device_config.device,
                    require_correctness=self.config.optimization.require_correctness,
                    rtol=self.config.optimization.correctness_rtol,
                    atol=self.config.optimization.correctness_atol,
                )

        self.knowledge_base: KnowledgeBase | None = None
        if self.config.knowledge.enabled:
            self.knowledge_base = load_knowledge_base(
                self.config.knowledge.knowledge_dir,
                dsl=self.config.device_config.dsl,
                device_type=self.config.device_config.device,
            )
            logger.info("  Knowledge base: %s", self.knowledge_base.summary())
        else:
            logger.info("  Knowledge base: disabled (set KNOWLEDGE_BASE_ENABLED=true to enable)")

        self.analyzer = AnalyzerAgent(
            knowledge_base=self.knowledge_base,
            dsl=self.config.device_config.dsl,
        )
        self.planner = PlannerAgent()

        match self.config.agent.strategy:
            case "cover":
                Agent = OptimizerAgent
            case "react":
                Agent = OptimizerReActAgent
            case _:
                Agent = OptimizerAgent

        self.optimizer = Agent(
            executor=executor,
            validator=validator,
            max_iterations=self.config.agent.max_iterations,
            knowledge_base=self.knowledge_base,
            dsl=self.config.device_config.dsl,
        )
        self.executor = executor
        self.validator = validator

        logger.info("XeForgePipeline initialized (LLM-knowledge mode)")
        logger.info(f"  LLM: {self.config.llm.model}")
        logger.info(
            f"  Agent: {self.config.agent.strategy} (max_iters={self.config.agent.max_iterations})"
        )

    def _setup_logging(self):
        log_level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        Path(self.config.logging.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.logging.kernel_dir).mkdir(parents=True, exist_ok=True)

    def _setup_llm(self):
        if self.config.llm.api_base:
            os.environ["OPENAI_API_BASE"] = self.config.llm.api_base
        if self.config.llm.api_key:
            os.environ["OPENAI_API_KEY"] = self.config.llm.api_key
        try:
            # This custom client is what litellm actually uses for requests, so it
            # MUST carry the timeout — otherwise httpx's no-timeout default wins and
            # a stalled generation hangs forever regardless of the dspy.LM timeout.
            litellm.client_session = httpx.Client(
                verify=False, timeout=httpx.Timeout(self.config.llm.timeout)
            )
            lm = dspy.LM(
                model=self.config.llm.model,
                api_base=self.config.llm.api_base,
                model_type=self.config.llm.model_type,
                api_key=self.config.llm.api_key or "",
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                # Bound each request so a transient endpoint stall can't hang the
                # whole run; retry a couple times to ride out dropped requests.
                timeout=self.config.llm.timeout,
                num_retries=self.config.llm.num_retries,
                cache=False,
            )
            dspy.configure(lm=lm, warn_on_type_mismatch=False)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {e}") from e

    def _resolve_tolerances(self, spec=None, variant_type="bench-gpu", rtol=None, atol=None):
        ertol = self.config.optimization.correctness_rtol
        eatol = self.config.optimization.correctness_atol
        if spec:
            sr, sa = spec.get_rtol(variant_type), spec.get_atol(variant_type)
            if sr is not None:
                ertol = sr
            if sa is not None:
                eatol = sa
        if rtol is not None:
            ertol = rtol
        if atol is not None:
            eatol = atol
        return ertol, eatol

    def _lowering_kb(self):
        """Load (and cache) the mlir/linalg-scoped KB for lowering-config proposals.

        The pipeline's shared self.knowledge_base is mlir/xpu-scoped (WG patterns);
        the lowering configs live in mlir/linalg/, so the lowering agent needs its
        own KB view. Returns None if the KB is disabled.
        """
        if not self.config.knowledge.enabled:
            return None
        if getattr(self, "_cached_lowering_kb", None) is None:
            from xe_forge.knowledge.loader import load_knowledge_base

            self._cached_lowering_kb = load_knowledge_base(
                self.config.knowledge.knowledge_dir, dsl="mlir", device_type="linalg"
            )
        return self._cached_lowering_kb

    def _maybe_lower_linalg(self, kernel_code: str, flop: float | None) -> str | None:
        """If *kernel_code* is Linalg-level, tile-search + lower it to WG-level.

        Runs the hybrid search: LLM proposes lowering configs (grounded by the
        mlir/linalg KB), then each is lowered + timed on the GPU (kernel-only
        rtclock) and the fastest correct one wins. Returns a runnable, self-contained
        WG-level kernel (best config spliced into a timing harness), or None if the
        kernel is already WG-level / cannot be lowered.
        """
        from xe_forge.core.linalg_lowering import (
            detect_mlir_level,
            extract_matmul_dims,
            render_timing_harness,
        )

        if detect_mlir_level(kernel_code) != "linalg":
            return None
        executor = self.executor
        if not hasattr(executor, "sweep_configs"):
            logger.warning("Executor has no sweep_configs; skipping Linalg lowering.")
            return None

        # A multi-matmul chain (>=2 linalg.matmul) is an MLP, not a single GEMM —
        # route it FIRST. extract_matmul_dims would otherwise match just the first
        # matmul and mis-lower the chain as one GEMM.
        if kernel_code.count("linalg.matmul") >= 2:
            mlp = self._maybe_lower_mlp(kernel_code)
            if mlp is not None:
                return mlp
            # fall through: maybe attention (batch_matmul-based) or unsupported.

        dims = extract_matmul_dims(kernel_code)
        if dims is None:
            # Not a single plain matmul. Try the multi-layer MLP path (a chain of
            # transpose-B matmul + bias/activation epilogues) -> N chained WG kernels.
            mlp = self._maybe_lower_mlp(kernel_code)
            if mlp is not None:
                return mlp
            # Else the fused-attention path (transpose + batch_matmul QK^T + softmax
            # + batch_matmul PV) via the lighthouse schedule.
            attn = self._maybe_lower_attention(kernel_code)
            if attn is not None:
                return attn
            logger.warning("Could not extract matmul dims; skipping Linalg lowering.")
            return None
        m, n, k = dims

        # Bare compute kernel for the lowering pipeline (strip any harness the input
        # brought; the timing harness is generated per config).
        bare, _ = _split_linalg_harness(kernel_code)

        # LLM shortlist (KB-grounded) -> timed sweep (hybrid search).
        from xe_forge.agents.linalg_lowering_agent import LinalgLoweringAgent

        agent = LinalgLoweringAgent(knowledge_base=self._lowering_kb())
        configs = agent.propose(bare, m, n, k)
        logger.info(
            "STAGE: LINALG_LOWERING — sweeping %d configs for %dx%dx%d", len(configs), m, n, k
        )
        best_cfg, best_wg, best_r, _ = executor.sweep_configs(
            bare, None, configs, flop=flop, dims=(m, n, k)
        )
        if best_cfg is None:
            logger.warning("No lowering config produced a correct kernel; keeping input.")
            return None
        logger.info(
            "LINALG_LOWERING best config: %s (%s)",
            best_cfg,
            f"{best_r.execution_time_ms:.4f}ms" if best_r and best_r.execution_time_ms else "correct",
        )
        # Return a runnable, self-contained WG kernel (best config's timing harness).
        from xe_forge.core.mlir_executor import _kernel_name

        harness = render_timing_harness(best_cfg, m, n, k, kernel_name=_kernel_name(best_wg))
        return executor._splice_kernel(harness, best_wg, kernel_only=True)

    def _maybe_lower_attention(self, kernel_code: str) -> str | None:
        """Lower a fused-attention Linalg graph to a runnable WG kernel.

        Recognizes the attention pattern (batch_matmul QK^T + softmax +
        batch_matmul PV), extracts (Z, H, n_ctx, n_head), and reuses lighthouse's
        fused_attention.py --dump-kernel=xegpu-wg as the lowering engine (we only
        consume the dumped .mlir — no run-time lighthouse dependency). Returns a
        self-contained WG kernel wrapped in a runnable single-launch @main harness
        (so the GRF sweep can time it), or None if this isn't attention / lowering
        is unavailable.
        """
        from xe_forge.core.attention_lowering import (
            detect_attention_shape,
            lower_attention_to_wg,
            synthesize_run_harness,
        )

        shape = detect_attention_shape(kernel_code)
        if shape is None:
            return None
        z, h, n_ctx, n_head = shape
        logger.info(
            "STAGE: LINALG_LOWERING — fused attention Z=%d H=%d n_ctx=%d n_head=%d "
            "(lowering via lighthouse)",
            z,
            h,
            n_ctx,
            n_head,
        )
        wg = lower_attention_to_wg(z, h, n_ctx, n_head)
        if wg is None:
            return None
        harness = synthesize_run_harness(wg)
        if harness is None:
            logger.warning(
                "Attention lowered to WG but harness synthesis failed; keeping input."
            )
            return None
        logger.info("Attention lowered to WG-level kernel (runnable harness synthesized).")
        # Unlike the matmul config sweep (which picks GRF as part of the search),
        # the lighthouse attention kernel has NOT chosen a GRF mode — the GRF sweep
        # is exactly where its ~1.7x win comes from. Flag it so optimize() runs the
        # sweep on the lowered kernel despite a LINALG_LOWERING stage being recorded.
        self._grf_sweep_after_lowering = True
        return harness

    def _maybe_lower_mlp(self, kernel_code: str) -> str | None:
        """Lower a multi-layer MLP (chain of nn.Linear + activation) to a runnable
        WG kernel.

        Recognizes the MLP chain (>=2 transpose-B matmuls each with a bias/activation
        epilogue), lowers it to N chained XeGPU-WG kernels via
        MlirExecutor.lower_mlp_to_wg (autotuning each layer's tile, and picking
        large-GRF tiles when the executor is in large-GRF mode), then synthesizes a
        runnable @main that launches the N kernels in sequence with intermediate
        buffers. Returns the harness, or None if this isn't an MLP / can't be lowered.
        """
        from xe_forge.core.mlp_lowering import (
            fold_transpose_into_matmul,
            parse_mlp,
            synthesize_mlp_run_harness,
        )

        executor = self.executor
        if not hasattr(executor, "lower_mlp_to_wg"):
            return None
        folded = fold_transpose_into_matmul(kernel_code)
        layers = parse_mlp(folded)  # cheap structural check (no autotune)
        if not layers:
            return None
        # Autotune per-layer tiles once; use large-GRF tiles when the executor is in
        # large-GRF mode (the module is then run with the igc flag — see below).
        want_grf = bool(getattr(executor, "large_grf", False))
        logger.info(
            "STAGE: LINALG_LOWERING — MLP chain of %d layers (autotune, large_grf=%s)",
            len(layers),
            want_grf,
        )
        tuned = parse_mlp(folded, executor=executor, large_grf=want_grf)
        if not tuned:
            return None
        # Lower with the exact tuned tiles (no second autotune → harness grid matches).
        ok, wg, err, used = executor.lower_mlp_to_wg(kernel_code, layers=tuned)
        if not ok:
            logger.warning("MLP lowering failed (%s); keeping input.", (err or "")[:200])
            return None
        harness = synthesize_mlp_run_harness(wg, used)
        if harness is None:
            logger.warning("MLP lowered to WG but harness synthesis failed; keeping input.")
            return None
        logger.info(
            "MLP lowered to %d chained XeGPU-WG kernels (runnable harness synthesized).",
            len(layers),
        )
        # Tiles (incl. GRF) are already chosen by the autotune; the WG stages should
        # NOT re-run the GRF sweep on this multi-kernel module.
        self._grf_sweep_after_lowering = False
        return harness

    def _maybe_sweep_grf(self, kernel_code: str, flop: float | None):
        """Autonomously choose large-GRF vs default for a runnable WG-level kernel.

        Large register file is a compile-time pipeline flag (not an IR edit), so
        both variants run the same kernel IR — correctness is guaranteed and only
        timing differs. Requires a *runnable* kernel (a @main that launches once);
        the executor's IMEX profiling times each variant. On a win, flips the
        executor's default pipeline to large-GRF for all subsequent runs and
        returns a StageResult; otherwise returns None.
        """
        executor = self.executor
        if not hasattr(executor, "sweep_grf"):
            return None
        # sweep_grf runs the kernel via execute(), which needs an @main entry
        # point that launches once. WG inputs lacking one (e.g. lighthouse's
        # @__benchmark/@payload form) aren't directly runnable — skip with a note
        # rather than burning two failing lower+run attempts.
        if "@main" not in kernel_code:
            logger.info(
                "GRF sweep: no @main entry point in WG kernel; skipping "
                "(kernel not directly runnable)."
            )
            return None
        try:
            best_large, res = executor.sweep_grf(kernel_code, flop=flop)
        except Exception as e:
            logger.warning("GRF sweep failed (%s); skipping.", e)
            return None
        d, l = res.get("default_grf"), res.get("large_grf")
        if d is None and l is None:
            logger.info("GRF sweep: kernel not runnable as-is; skipping.")
            return None
        speedup = (d / l) if (best_large and d and l) else 1.0
        if best_large:
            # Persist the choice: subsequent executor runs use large-GRF.
            if hasattr(executor, "large_grf"):
                executor.large_grf = True
                if "large-register-file" not in executor.pipeline:
                    executor.pipeline = executor.pipeline + " igc-cmd-options=-ze-opt-large-register-file"
            logger.info("GRF sweep: large-GRF chosen (%.2fx: %.4f -> %.4f ms)", speedup, d, l)
            return StageResult(
                stage=OptimizationStage.DEVICE_SPECIFIC,
                success=True,
                input_code=kernel_code,
                output_code=kernel_code,  # IR unchanged; flag applied at lowering
                changes_made=[f"enabled large register file (igc) — {speedup:.2f}x"],
                speedup=speedup,
            )
        logger.info("GRF sweep: default GRF kept (large-GRF not faster).")
        return None

    def optimize(
        self,
        kernel_code=None,
        reference_code=None,
        kernel_name=None,
        input_shapes=None,
        reference_fn=None,
        stages=None,
        spec_path=None,
        variant_type="bench-gpu",
        target_dtype=None,
        rtol=None,
        atol=None,
        *,
        triton_code=None,
        pytorch_code=None,
    ):
        # Backward compat aliases
        if kernel_code is None:
            kernel_code = triton_code
        if reference_code is None:
            reference_code = pytorch_code or ""
        # torch is only needed for dtype mapping / tensor-based executors. The
        # MLIR (XeGPU) path uses self-contained kernels and never touches it, so
        # tolerate its absence there.
        try:
            import torch
        except ImportError:
            torch = None

        spec, flop, dtype, init_args, spec_dims, input_dtypes = None, None, None, None, None, None
        if spec_path:
            from xe_forge.core.spec_loader import load_spec

            spec = load_spec(spec_path)
            variant_type = spec.resolve_variant(variant_type)
            input_shapes = spec.get_input_shapes(variant_type)
            spec_dims = spec.get_dims(variant_type)
            flop = spec.get_flop(variant_type)
            dtype = spec.get_dtype(variant_type)
            input_dtypes = spec.get_input_dtypes(variant_type)
            init_args = spec.get_init_args(variant_type)
            logger.info(
                f"Loaded spec: variant={variant_type}, shapes={input_shapes}, "
                f"dims={spec_dims}, flop={flop}, dtype={dtype}"
            )
            if init_args:
                logger.info(f"  Model init args: {init_args}")

        ertol, eatol = self._resolve_tolerances(spec, variant_type, rtol, atol)
        if hasattr(self.executor, "rtol"):
            self.executor.rtol = ertol
        if hasattr(self.executor, "atol"):
            self.executor.atol = eatol

        if target_dtype and torch is not None:
            dm = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            dtype = dm.get(target_dtype, dtype)

        display_name = kernel_name or "Model"
        logger.info(f"Starting optimization for kernel: {display_name}")

        val_orig_tflops, val_orig_ms = None, None
        _is_mlir = self.config.device_config.dsl == DSL.MLIR

        # The MLIR path uses a self-contained executor and a different baseline
        # path (below); avoid importing the torch-backed executors entirely.
        if _is_mlir:
            _is_sycl = False
            _bench_ex = self.executor
        else:
            from xe_forge.core.executor import KernelBenchExecutor
            from xe_forge.core.sycl_executor import SyclExecutor

            _is_sycl = isinstance(self.executor, SyclExecutor)
            _bench_ex = (
                self.executor
                if isinstance(self.executor, (KernelBenchExecutor, SyclExecutor))
                else KernelBenchExecutor(device=self.config.device_config.device)
            )
        if self.executor and not _is_mlir and (_is_sycl or input_shapes):
            try:
                if _is_sycl:
                    _sycl_dims = spec_dims or dict(
                        zip(("M", "N", "K"), _extract_gemm_dims(input_shapes), strict=False)
                    )
                    orig_r = _bench_ex.execute(
                        kernel_code=kernel_code,
                        dims=_sycl_dims,
                    )
                else:
                    orig_r = _bench_ex.execute(
                        kernel_code,
                        None,
                        input_shapes,
                        flop=flop,
                        dtype=dtype,
                        init_args=init_args,
                        input_dtypes=input_dtypes,
                    )
                if orig_r.success:
                    val_orig_tflops, val_orig_ms = orig_r.tflops, orig_r.execution_time_ms
                    logger.info(f"Original: {val_orig_tflops:.2f} TFLOPS, {val_orig_ms:.2f} ms")
                else:
                    logger.error(f"Baseline FAILED: {orig_r.error_message}")
                    if hasattr(orig_r, "error_traceback"):
                        logger.debug(orig_r.error_traceback)
            except Exception as e:
                logger.warning(f"Failed to measure original: {e}")

        # MLIR baseline: the kernel is self-contained, so just run it once and
        # read its embedded timing/correctness.
        if self.executor and _is_mlir:
            try:
                orig_r = self.executor.execute(kernel_code=kernel_code, flop=flop)
                if orig_r.success:
                    val_orig_tflops, val_orig_ms = orig_r.tflops, orig_r.execution_time_ms
                    if val_orig_ms is not None:
                        logger.info(
                            "Original: %s, %.4f ms",
                            f"{val_orig_tflops:.2f} TFLOPS" if val_orig_tflops else "n/a TFLOPS",
                            val_orig_ms,
                        )
                    if orig_r.output_correct is False:
                        logger.warning("Baseline kernel does not pass its own correctness check.")
                else:
                    logger.error("Baseline FAILED: %s", orig_r.error_message)
            except Exception as e:
                logger.warning(f"Failed to measure original MLIR kernel: {e}")

        if self.trial_manager and kernel_name:
            try:
                import tempfile

                tmp = Path(tempfile.mkdtemp()) / f"{kernel_name}_baseline.py"
                tmp.write_text(kernel_code)
                self.trial_manager.init(kernel_name, str(tmp))
                logger.info("Trial tree initialized for '%s'", kernel_name)
            except Exception as e:
                logger.warning("Could not initialize trial tree: %s", e)

        candidates = []
        best_k = max(1, self.config.optimization.best_k)

        for attempt in range(best_k):
            if best_k > 1:
                logger.info(f"Attempt {attempt + 1}/{best_k}")

            result = OptimizationResult(
                kernel_name=display_name, original_code=kernel_code, timestamp=datetime.now()
            )
            result.original_tflops, result.original_ms = val_orig_tflops, val_orig_ms

            etd = target_dtype or self.config.optimization.target_dtype
            if etd is None and dtype is not None:
                etd = {
                    torch.float16: "float16",
                    torch.bfloat16: "bfloat16",
                    torch.float32: "float32",
                }.get(dtype)

            device_type = self.config.device_config.device
            xpu_config = get_device_config_for_pipeline(
                device_type=device_type,
                input_shapes=input_shapes,
                config=self.config,
                dtype=etd or "float16",
            )

            # --- MLIR two-level flow: lower Linalg -> XeGPU WG *before* analysis,
            # so the WG-level analyzer/stages operate on the lowered kernel.
            lowering_stage_result = None
            self._grf_sweep_after_lowering = False  # reset per attempt
            if self.config.device_config.dsl == DSL.MLIR:
                logger.info("=" * 60 + "\nSTAGE: LINALG_LOWERING\n" + "=" * 60)
                lowered = self._maybe_lower_linalg(kernel_code, flop)
                if lowered is not None:
                    lowering_stage_result = StageResult(
                        stage=OptimizationStage.LINALG_LOWERING,
                        success=True,
                        input_code=kernel_code,
                        output_code=lowered,
                        changes_made=["lowered Linalg to XeGPU WG-level (best config)"],
                    )
                    kernel_code = lowered  # WG stages now operate on the lowered kernel

            # --- Device-specific GRF sweep: for a runnable WG-level kernel, pick
            # the faster of {default, large} register file. Correctness-free (same
            # IR, compile-flag only); flips the executor to large-GRF on a win so
            # all subsequent runs inherit it. Skipped for kernels without @main.
            grf_stage_result = None
            from xe_forge.core.linalg_lowering import detect_mlir_level as _detect_level

            # Matmul lowering picks GRF within its config search, so skip the sweep
            # there; but an attention-lowered kernel (lighthouse output) hasn't, and
            # sets _grf_sweep_after_lowering to opt back in.
            _grf_after_lowering = getattr(self, "_grf_sweep_after_lowering", False)
            if (
                self.config.device_config.dsl == DSL.MLIR
                and (lowering_stage_result is None or _grf_after_lowering)
                and _detect_level(kernel_code) == "xegpu_wg"
            ):
                logger.info("=" * 60 + "\nSTAGE: GRF_SWEEP\n" + "=" * 60)
                grf_stage_result = self._maybe_sweep_grf(kernel_code, flop)

            logger.info("=" * 60 + "\nSTAGE: ANALYSIS\n" + "=" * 60)
            analysis = self.analyzer.analyze(
                kernel_code,
                reference_code,
                display_name,
                input_shapes,
                flop,
                target_dtype=etd,
            )
            result.analysis = analysis
            if lowering_stage_result is not None:
                result.stages_applied.append(lowering_stage_result)
            if grf_stage_result is not None:
                result.stages_applied.append(grf_stage_result)

            logger.info(f"Detected {len(analysis.detected_issues)} issues:")
            for iss in analysis.detected_issues:
                logger.info(f"  [{iss.severity}] {iss.issue_type.value}: {iss.description}")

            if not analysis.detected_issues:
                result.success, result.optimized_code = True, kernel_code
                candidates.append(result)
                continue

            logger.info("=" * 60 + "\nSTAGE: PLANNING\n" + "=" * 60)
            from xe_forge.knowledge.patterns import get_stage_for_issue

            stages_needed: dict[OptimizationStage, list[str]] = {}
            for iss in analysis.detected_issues:
                st = get_stage_for_issue(iss.issue_type)
                stages_needed.setdefault(st, []).append(iss.issue_type.value)

            from xe_forge.dsl_registry import get_stages_for_dsl

            _supported = set(get_stages_for_dsl(self.config.device_config.dsl))
            stages_needed = {s: v for s, v in stages_needed.items() if s in _supported}

            if stages:
                stages_to_apply = [
                    s for s in stages if s in stages_needed and s != OptimizationStage.ANALYSIS
                ]
                logger.info("Stage order: manual override")
            else:
                stages_to_apply = self.planner.plan(
                    stages_needed=stages_needed,
                    analysis=analysis,
                    input_shapes=input_shapes,
                    flop=flop,
                )

            logger.info("Optimization plan:")
            for s in PLANNER_DEFAULT_STAGE_ORDER:
                if s == OptimizationStage.ANALYSIS:
                    continue
                if s in stages_needed:
                    if s in stages_to_apply:
                        pos = stages_to_apply.index(s) + 1
                        issues_str = ", ".join(stages_needed[s])
                        logger.info(f"  + {s.value} [#{pos}]: {issues_str}")
                    else:
                        issues_str = ", ".join(stages_needed[s])
                        logger.info(f"  ~ {s.value} (deferred): {issues_str}")
                else:
                    logger.info(f"  - {s.value}: skipped")

            if not stages_to_apply:
                result.success, result.optimized_code = True, kernel_code
                candidates.append(result)
                continue

            current_code = kernel_code
            current_ms: float | None = val_orig_ms
            vtune_report = ""
            last_trial_id: str | None = None

            for stage_idx, stage in enumerate(stages_to_apply):
                logger.info("=" * 60 + f"\nSTAGE: {stage.value.upper()}\n" + "=" * 60)
                logger.info(f"Issues: {', '.join(stages_needed.get(stage, []))}")

                stage_result = self.optimizer.optimize_stage(
                    code=current_code,
                    stage=stage,
                    analysis=analysis,
                    xpu_config=xpu_config,
                    kernel_name=kernel_name,
                    input_shapes=input_shapes,
                    spec_dims=spec_dims,
                    flop=flop,
                    dtype=dtype,
                    pytorch_code=reference_code,
                    init_args=init_args,
                    vtune_report=vtune_report,
                    perf_context={
                        "original_ms": val_orig_ms,
                        "original_tflops": val_orig_tflops,
                        "current_ms": current_ms,
                        "speedup_so_far": (
                            round(val_orig_ms / current_ms, 3)
                            if val_orig_ms and current_ms and current_ms > 0
                            else None
                        ),
                    },
                    input_dtypes=input_dtypes,
                )
                result.stages_applied.append(stage_result)

                if (
                    stage_result.success
                    and stage_result.output_code
                    and stage_result.output_code != current_code
                ):
                    current_code = stage_result.output_code
                    if stage_result.speedup and val_orig_ms:
                        current_ms = val_orig_ms / stage_result.speedup
                    elif (
                        stage_result.metrics_after
                        and "execution_time_ms" in stage_result.metrics_after
                    ):
                        current_ms = stage_result.metrics_after["execution_time_ms"]
                    logger.info(
                        f"Stage {stage.value} OK"
                        + (f" ({stage_result.speedup:.2f}x)" if stage_result.speedup else "")
                    )
                elif not stage_result.success:
                    logger.warning(f"Stage {stage.value} failed: {stage_result.error_message}")

                if self.trial_manager and kernel_name and stage_result.output_code:
                    try:
                        import tempfile

                        tmp = Path(tempfile.mkdtemp()) / f"{kernel_name}_stage_{stage.value}.py"
                        tmp.write_text(stage_result.output_code)
                        trial_id = self.trial_manager.save_trial(
                            kernel_name,
                            str(tmp),
                            parent=last_trial_id,
                            strategy=f"stage:{stage.value}",
                        )
                        speedup = (
                            val_orig_ms / current_ms
                            if val_orig_ms and current_ms and current_ms > 0
                            else None
                        )
                        self.trial_manager.record_result(
                            kernel_name,
                            trial_id,
                            correctness="pass" if stage_result.success else "fail",
                            speedup=speedup,
                            baseline_us=(val_orig_ms or 0) * 1000,
                            triton_us=(current_ms or 0) * 1000,
                        )
                        if stage_result.success:
                            last_trial_id = trial_id
                        logger.info("Trial %s recorded for stage %s", trial_id, stage.value)
                    except Exception as e:
                        logger.warning("Could not record trial for stage %s: %s", stage.value, e)

                if (
                    self.profiler
                    and stage_idx > 0
                    and stage_result.success
                    and stage_result.output_code
                    and spec_path
                ):
                    try:
                        import tempfile

                        tmp = Path(tempfile.mkdtemp()) / f"{kernel_name}_profile.py"
                        tmp.write_text(stage_result.output_code)
                        profile_result = self.profiler.profile(
                            str(tmp),
                            spec_path=spec_path,
                            variant=variant_type,
                        )
                        if not profile_result.error:
                            vtune_report = profile_result.format_for_llm()
                            logger.info("VTune profile updated after stage %s", stage.value)
                        else:
                            logger.warning("VTune profiling error: %s", profile_result.error)
                    except Exception as e:
                        logger.warning("VTune profiling failed after stage %s: %s", stage.value, e)

                if stage == OptimizationStage.DISCOVERY and stage_result.success:
                    open_ended_issues = [
                        i
                        for i in analysis.detected_issues
                        if i.issue_type == IssueType.OPEN_ENDED and i.open_ended_proposal
                    ]
                    for oi in open_ended_issues:
                        logger.info(
                            "DISCOVERY succeeded — promote to named IssueType:\n%s",
                            oi.open_ended_proposal,
                        )

                analysis = self.analyzer.analyze(
                    current_code,
                    reference_code,
                    display_name,
                    input_shapes,
                    flop,
                    target_dtype=etd,
                )

            if self.executor and (_is_sycl or input_shapes) and current_code != kernel_code:
                try:
                    if _is_sycl:
                        opt_r = _bench_ex.execute(
                            kernel_code=current_code,
                            dims=_sycl_dims,
                        )
                    else:
                        opt_r = _bench_ex.execute(
                            current_code,
                            kernel_name,
                            input_shapes,
                            flop=flop,
                            dtype=dtype,
                            init_args=init_args,
                            input_dtypes=input_dtypes,
                        )
                    if opt_r.success:
                        result.optimized_tflops, result.optimized_ms = (
                            opt_r.tflops,
                            opt_r.execution_time_ms,
                        )
                        if result.original_ms and result.optimized_ms:
                            result.total_speedup = result.original_ms / result.optimized_ms
                            logger.info(f"Total speedup: {result.total_speedup:.2f}x")
                except Exception as e:
                    logger.warning(f"Failed to measure optimized: {e}")
                    if current_ms and current_ms != val_orig_ms:
                        result.optimized_ms = current_ms
                        if result.original_ms and result.optimized_ms:
                            result.total_speedup = result.original_ms / result.optimized_ms
                            logger.info(
                                f"Total speedup (from stage measurements): {result.total_speedup:.2f}x"
                            )

            result.optimized_code, result.success = current_code, True
            candidates.append(result)

        if not candidates:
            return OptimizationResult(
                kernel_name=display_name, original_code=kernel_code, timestamp=datetime.now()
            )

        result = max(
            candidates, key=lambda r: r.total_speedup if r.total_speedup is not None else -1.0
        )
        self._save_results(result)

        logger.info("=" * 60 + "\nOPTIMIZATION COMPLETE\n" + "=" * 60)
        ok = [s for s in result.stages_applied if s.success]
        fail = [s for s in result.stages_applied if not s.success]
        logger.info(f"Stages: {len(ok)}/{len(result.stages_applied)} succeeded")
        if fail:
            logger.info(f"Failed: {[s.stage.value for s in fail]}")
        if result.total_speedup:
            logger.info(f"Speedup: {result.total_speedup:.2f}x")
        return result

    def optimize_file(
        self,
        input_path,
        output_path=None,
        kernel_name=None,
        spec_path=None,
        variant_type="bench-gpu",
        target_dtype=None,
    ):
        with open(input_path) as f:
            kernel_code = f.read()
        result = self.optimize(
            kernel_code,
            "",
            kernel_name,
            spec_path=spec_path,
            variant_type=variant_type,
            target_dtype=target_dtype,
        )
        if output_path and result.optimized_code:
            with open(output_path, "w") as f:
                f.write(result.optimized_code)
        return result

    def _save_results(self, result):
        if not self.config.logging.save_intermediate:
            return
        ext = ".cpp" if DSL(self.config.device_config.dsl).code_language == "cpp" else ".py"
        comment = "//" if ext == ".cpp" else "#"
        ts = result.timestamp.strftime("%Y%m%d_%H%M%S")
        kd = Path(self.config.logging.kernel_dir)
        with open(kd / f"{result.kernel_name}_{ts}_original{ext}", "w") as f:
            f.write(f"{comment} Original: {result.kernel_name}\n\n{result.original_code}")
        if result.optimized_code and result.optimized_code != result.original_code:
            with open(kd / f"{result.kernel_name}_{ts}_optimized{ext}", "w") as f:
                f.write(f"{comment} Optimized: {result.kernel_name}\n")
                if result.total_speedup:
                    f.write(f"{comment} Speedup: {result.total_speedup:.2f}x\n")
                f.write(
                    f"{comment} Stages: {[s.stage.value for s in result.stages_applied if s.success]}\n\n"
                )
                f.write(result.optimized_code)
