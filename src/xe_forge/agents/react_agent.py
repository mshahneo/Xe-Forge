"""
Optimizer Agent - Uses ReAct for iterative kernel optimization with tool-based verification.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Callable

import dspy

from xe_forge.agents.base import Optimizer
from xe_forge.knowledge.patterns import get_stage_for_issue

try:
    from xe_forge.knowledge.loader import KnowledgeBase
except ImportError:
    KnowledgeBase = None
from xe_forge.models import (
    DSL,
    DetectedIssue,
    KernelAnalysis,
    OptimizationStage,
    StageResult,
)

logger = logging.getLogger(__name__)


SUCCESS_MESSAGE = "Success! Optimization verified and kernel is faster."


class _BestCandidate:
    """Remembers the best kernel the verify tool actually proved on the GPU.

    A stage's verify tool is called once per ReAct iteration, and each call
    compiles, runs and TIMES a candidate against the stage's input kernel. Only
    the agent's FINAL answer used to be kept, so a stage that verified a correct,
    faster kernel at iteration k and then returned a broken answer at iteration
    k+1 was recorded as a total failure and fell back to its input — throwing away
    a result that was already measured on hardware. Observed cost: an FA
    memory_access stage proved 0.9586 ms, then regressed to its 0.9931 ms input.

    Speedups from different iterations of one stage are comparable because every
    verify call compares against the same original_code, so keeping the max is
    well-defined.
    """

    def __init__(self) -> None:
        self.code: str | None = None
        self.speedup: float = 1.0
        self.comparison = None

    def offer(self, code: str, speedup: float | None, comparison=None) -> None:
        """Record *code* if it beats the best seen so far (and the input kernel)."""
        if not code or speedup is None or speedup <= self.speedup:
            return
        self.code = code
        self.speedup = speedup
        self.comparison = comparison


def _metrics_from_comparison(comparison) -> tuple[dict | None, dict | None]:
    """(metrics_before, metrics_after) for a comparison, or Nones when unavailable.

    All three comparison types (Triton/SYCL/MLIR) expose time_us + tflops, but the
    tflops half is optional (it needs a FLOP count), so each side is emitted only
    when complete.
    """

    def _side(time_us, tflops) -> dict | None:
        if time_us is None or tflops is None:
            return None
        return {"time_us": time_us, "tflops": tflops}

    if comparison is None:
        return None, None
    return (
        _side(getattr(comparison, "original_time_us", None), comparison.original_tflops),
        _side(getattr(comparison, "optimized_time_us", None), comparison.optimized_tflops),
    )


def _extract_gemm_dims(
    input_shapes: list[tuple[int, ...]] | None,
) -> tuple[int, int, int]:
    """Extract M, N, K from GEMM input shapes [(M, K), (K, N)]."""
    if input_shapes and len(input_shapes) >= 2:
        a, b = input_shapes[0], input_shapes[1]
        if len(a) >= 2 and len(b) >= 2:
            return a[-2], b[-1], a[-1]
    return 1024, 1024, 1024


def _verify_sycl(code, original_code, executor, input_shapes, spec_dims=None, best=None):
    """Verify a SYCL C++ kernel: basic structure check + runtime comparison."""
    if "#include" not in code:
        return "MISSING: C++ code must contain #include directives."
    if "sycl" not in code.lower() and "cutlass" not in code.lower():
        return "MISSING: Code does not appear to be a SYCL/CUTLASS kernel."

    if executor:
        try:
            _dims = spec_dims or dict(
                zip(("M", "N", "K"), _extract_gemm_dims(input_shapes), strict=False)
            )
            comparison = executor.compare_kernels(
                original_code=original_code,
                optimized_code=code,
                dims=_dims,
            )
            if not comparison.optimized_correct:
                return comparison.feedback_message or "Optimized kernel failed."
            if comparison.is_slower:
                sd = 1.0 / comparison.speedup if comparison.speedup > 0 else float("inf")
                return (
                    f"PERFORMANCE REGRESSION: {sd:.2f}x SLOWER.\n"
                    f"Original: {comparison.original_time_ms:.4f}ms ({comparison.original_tflops or 0:.3f} TFlop/s)\n"
                    f"Optimized: {comparison.optimized_time_ms:.4f}ms ({comparison.optimized_tflops or 0:.3f} TFlop/s)"
                )
            logger.info(
                f"SYCL optimization verified: {comparison.speedup:.2f}x speedup "
                f"({comparison.original_tflops or 0:.3f} -> {comparison.optimized_tflops or 0:.3f} TFlop/s)"
            )
            if best is not None:
                best.offer(code, comparison.speedup, comparison)
            return SUCCESS_MESSAGE
        except Exception as e:
            return f"RUNTIME ERROR: {e!s}"

    logger.warning("No executor - accepting SYCL code based on static checks only")
    return SUCCESS_MESSAGE


def _verify_mlir(code, original_code, executor, flop=None, best=None):
    """Verify an XeGPU WG-level MLIR kernel: structure check + runtime compare.

    The kernel file is self-contained (host @main + embedded CPU reference), so
    correctness is the optimized kernel's own [ALLCLOSE: TRUE]; speedup uses
    rtclock timing when the harness prints it. Mirrors optimizer_agent._verify_mlir
    so the ReAct and CoVeR strategies verify MLIR identically.
    """
    # Cheap structural pre-checks before paying for lowering+execution.
    if "gpu.launch_func" not in code:
        return "MISSING: module must keep the gpu.launch_func kernel launch."
    if "func.func @main" not in code:
        return "MISSING: module must keep the @main host harness (correctness oracle)."
    if "printAllclose" not in code and "[ALLCLOSE" not in code:
        return (
            "MISSING: keep the in-file correctness check (printAllclose / "
            "[ALLCLOSE]) — it is the verification oracle."
        )

    if executor:
        try:
            comparison = executor.compare_kernels(
                original_code=original_code,
                optimized_code=code,
                flop=flop,
            )
            if not comparison.optimized_correct:
                return comparison.feedback_message or "Optimized kernel failed correctness."
            if getattr(comparison, "lowered_identical", False):
                # Dead edit: lowers to identical IR. Don't accept it, and tell the
                # LLM why so it tries a change the compiler actually keeps.
                logger.info("MLIR optimization rejected: no-op (lowers to identical IR)")
                return comparison.feedback_message or (
                    "NO-OP: optimized kernel lowers to IR identical to the original."
                )
            if comparison.is_slower:
                sd = 1.0 / comparison.speedup if comparison.speedup > 0 else float("inf")
                return (
                    f"PERFORMANCE REGRESSION: {sd:.2f}x SLOWER.\n"
                    f"Original: {comparison.original_time_ms:.4f}ms, "
                    f"Optimized: {comparison.optimized_time_ms:.4f}ms"
                )
            logger.info("MLIR optimization verified: %.2fx speedup", comparison.speedup)
            if best is not None:
                best.offer(code, comparison.speedup, comparison)
            return SUCCESS_MESSAGE
        except Exception as e:
            return f"RUNTIME ERROR: {e!s}"

    logger.warning("No executor - accepting MLIR code based on static checks only")
    return SUCCESS_MESSAGE


class OptimizationReActSignature(dspy.Signature):
    """Apply optimization transformation to Triton kernel.

    You are an expert Triton kernel optimizer for Intel XPU.

    Your task: Optimize the kernel for maximum performance.
    You may change the algorithm/computation approach if it produces equivalent outputs.
    Maintain the same model signature, including the weights' shapes and names. This is necessary for having identical initialization process for formal verification which is done by a correctness tool.

    === OPTIMIZATION PRIORITIES ===
    1. Apply the specific optimization patterns from knowledge_patterns
    2. Use block pointers with tl.make_block_ptr() for better memory access
    3. Use optimal tile sizes: BLOCK_M=256, BLOCK_N=256, BLOCK_K=32 for XPU
    4. Set num_warps=32 for Intel XPU
    5. Add GROUP_SIZE_M swizzling for better L2 cache utilization
    6. Use boundary_check=(0, 1) tuple format, NOT booleans

    === CODE REQUIREMENTS ===
    - Include ALL imports (torch, triton, triton.language as tl)
    - Include the @triton.jit decorator and kernel function
    - Include the Model class with forward() method
    - num_warps MUST be a power of 2 (1, 2, 4, 8, 16, 32)
    - num_stages MUST be a positive integer
    - Block sizes (BLOCK_M, BLOCK_N, BLOCK_K) MUST be powers of 2
    - Block sizes should not exceed 256 for most cases

    === WHAT TO CHANGE ===
    Focus on the issues listed and apply the patterns from knowledge_patterns.
    Be aggressive with optimizations - the verification tool will check correctness.
    """

    original_code: dspy.Code["python"] = dspy.InputField(  # noqa: UP037
        desc="Original Triton kernel code for reference"
    )
    current_code: dspy.Code["python"] = dspy.InputField(  # noqa: UP037
        desc="Current Triton kernel code to optimize"
    )
    stage: str = dspy.InputField(
        desc="Optimization stage to apply (e.g., dtype_fix, block_pointers, device_specific)"
    )
    issues: list[DetectedIssue] = dspy.InputField(desc="Specific issues to fix in this stage")
    knowledge_patterns: str = dspy.InputField(
        desc="Optimization patterns and examples to follow - APPLY THESE"
    )
    xpu_config: str = dspy.InputField(desc="Intel XPU configuration parameters")

    optimized_code: dspy.Code["python"] = dspy.OutputField(  # noqa: UP037
        desc="Complete optimized Triton kernel code. Must include all imports, decorators, kernel function, and Model class."
    )


class SyclOptimizationReActSignature(dspy.Signature):
    """Optimize a SYCL/CUTLASS C++ kernel for Intel XPU.

    You are an expert SYCL/CUTLASS kernel optimizer for Intel XPU.

    Your task: Optimize the C++ kernel for maximum performance.
    You may change template parameters, dispatch policies, and data types
    if the outputs remain numerically equivalent.

    === OPTIMIZATION PRIORITIES ===
    1. TileShape: try Shape<_256,_256,_32> or Shape<_128,_128,_64> for BMG
    2. PipelineStages: 2-4 (balance prefetching vs register pressure)
    3. MMA Atom: XE_DPAS_TT<8, float, bfloat16_t> for BMG
    4. Dispatch Policy: MainloopXeL1Staged for L1 caching
    5. Data types: bfloat16_t inputs, float accumulators
    6. Memory layout: match RowMajor/ColumnMajor to access patterns

    === CODE REQUIREMENTS ===
    - Complete, valid SYCL C++ with all #include directives
    - CUTLASS template types, ExampleRunner, and main()
    - Must compile with icpx -fsycl
    - Keep the Cutlass GEMM Performance output format
    """

    original_code: dspy.Code["cpp"] = dspy.InputField(  # noqa: UP037
        desc="Original SYCL C++ kernel code for reference"
    )
    current_code: dspy.Code["cpp"] = dspy.InputField(  # noqa: UP037
        desc="Current SYCL C++ kernel code to optimize"
    )
    stage: str = dspy.InputField(desc="Optimization stage to apply")
    issues: list[DetectedIssue] = dspy.InputField(desc="Specific issues to fix in this stage")
    knowledge_patterns: str = dspy.InputField(desc="Optimization patterns and examples to follow")
    xpu_config: str = dspy.InputField(desc="Intel XPU configuration parameters")

    optimized_code: dspy.Code["cpp"] = dspy.OutputField(  # noqa: UP037
        desc="Complete optimized SYCL C++ kernel with all #includes, templates, ExampleRunner, and main()."
    )


class MlirOptimizationReActSignature(dspy.Signature):
    """Optimize an XeGPU workgroup-level MLIR kernel for Intel XPU.

    You are an expert in MLIR, the XeGPU dialect, Intel Xe GPU architecture
    (Xe-cores, subgroups, DPAS systolic array, 2D block loads), and
    high-performance kernel tuning.

    The input is a self-contained MLIR module: a host `func.func @main` that
    allocates inputs, launches a `gpu.module` kernel via `gpu.launch_func`, and
    verifies the result against an embedded CPU reference (printing
    `[ALLCLOSE: TRUE]`). Optimize for maximum performance while keeping the
    result numerically equivalent. Call the verification tool on each candidate;
    it lowers and runs the kernel and tells you correctness + speedup (or why it
    was rejected as incorrect, a no-op, or slower). Iterate until it succeeds.

    === HARD CONSTRAINTS (do not break these) ===
    - Edit ONLY the `gpu.module` kernel body, the `#xegpu.layout` attributes,
      and the `gpu.launch_func` grid/block geometry.
    - DO NOT modify the `@main` harness, the CPU reference function, the input
      fill values, or the `[ALLCLOSE]` check — they are the correctness oracle.
    - Re-emit the ENTIRE module verbatim except your intended edit. Do not
      reflow, retype, or "clean up" untouched lines (memref/strided types,
      attributes) — a single dropped character makes it fail to parse.
    - Keep the kernel entry symbol name and its `gpu.launch_func` reference
      consistent, and keep the module valid MLIR that lowers cleanly through
      `--gpu-lower-to-xevm-pipeline=xegpu-op-level=workgroup`.
    - The `sg_layout` product (number of subgroups) must remain consistent with
      the threads-per-workgroup in `gpu.launch_func`.

    === XeGPU WG-LEVEL OPTIMIZATION KNOBS ===
    - `#xegpu.layout<sg_layout=[..], sg_data=[..], inst_data=[..]>`: distribute
      the workgroup tile across subgroups; match sg_data / inst_data to
      DPAS-friendly shapes (e.g. [8,16], [16,16]).
    - Prefetch: insert `xegpu.prefetch_nd` with `l1_hint/l2_hint/l3_hint =
      #xegpu.cache_hint<cached>` ahead of `load_nd` to hide memory latency.
    - K-loop tiling: adjust the `scf.for` step / accumulator tile to trade
      register pressure against reuse.

    === STAGE-SPECIFIC GUIDANCE ===
    ALGORITHMIC: reduce FLOPs / memory traffic via algebraic rewrites that stay
      numerically equivalent — e.g. fold a quotient of exponentials
      exp(a)/exp(b) → exp(a-b) (carry any fastmath flag onto the surviving exp),
      hoist loop-invariant work, eliminate redundant loads/recomputation. APPLY
      the specific rewrites in knowledge_patterns.
    MEMORY_ACCESS: add/strengthen prefetch and cache hints; coalesce loads; pick
      sg_data tiles that map to contiguous 2D block loads.
    DEVICE_SPECIFIC: tune sg_layout / sg_data / inst_data and launch geometry for
      the Xe-core (DPAS shape, subgroup size 16).
    DTYPE_FIX: f16/bf16 inputs with f32 DPAS accumulators; avoid f64.
    FUSION: fold the elementwise epilogue into the kernel after the dpas loop.
    """

    original_code: str = dspy.InputField(desc="Original MLIR module for reference")
    current_code: str = dspy.InputField(desc="Current MLIR module to optimize")
    stage: str = dspy.InputField(desc="Optimization stage to apply")
    issues: list[DetectedIssue] = dspy.InputField(desc="Specific issues to fix in this stage")
    knowledge_patterns: str = dspy.InputField(
        desc="Optimization patterns and examples to follow - APPLY THESE"
    )
    xpu_config: str = dspy.InputField(desc="Intel XPU configuration parameters")

    optimized_code: dspy.Code["mlir"] = dspy.OutputField(  # noqa: UP037
        desc="Complete optimized MLIR module. Keep the @main harness and CPU reference "
        "unchanged; edit only the gpu.module kernel, #xegpu.layout attrs, and launch geometry."
    )


class OptimizerReActAgent(Optimizer):
    """
    Agent that applies optimization transformations to Triton kernels using ReAct.

    We use existing dspy.ReAct implementation.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        executor=None,
        validator: Callable | None = None,
        max_iterations: int = 5,
        dsl: DSL | str = DSL.TRITON,
    ):
        self.knowledge_base = knowledge_base
        self.executor = executor
        self.validator = validator
        self.max_iterations = max_iterations
        self.dsl = DSL(dsl) if isinstance(dsl, str) else dsl

        if not executor:
            logger.warning("No executor provided - kernels will NOT be verified at runtime!")

    def _create_verify_tool(
        self,
        original_code: str,
        kernel_name: str | None,
        input_shapes: list[tuple[int, ...]] | None,
        flop: float | None,
        dtype=None,
        spec_dims: dict[str, int] | None = None,
        input_dtypes: list | None = None,
        best: _BestCandidate | None = None,
    ) -> Callable:
        """Create a verification tool for ReAct.

        Returns SUCCESS_MESSAGE if valid and faster, detailed error otherwise.

        When *best* is given, every candidate this tool proves correct-and-faster on
        the GPU is offered to it, so the stage can fall back to a measured winner if
        the agent's final answer turns out to be worse (see _BestCandidate).
        """
        executor = self.executor
        dsl = self.dsl

        def compile_and_verify(optimized_code: dspy.Code["python"]) -> str:  # noqa: UP037
            """Compile and verify the optimized kernel.
            Returns SUCCESS_MESSAGE on success, or detailed error message.
            """
            code: str = optimized_code.code

            if dsl == DSL.SYCL:
                return _verify_sycl(
                    code,
                    original_code,
                    executor,
                    input_shapes,
                    spec_dims,
                    best=best,
                )

            if dsl == DSL.MLIR:
                return _verify_mlir(code, original_code, executor, flop, best=best)

            # --- Triton path (unchanged) ---
            try:
                ast.parse(code)
            except SyntaxError as e:
                return (
                    f"SYNTAX ERROR at line {e.lineno}: {e.msg}\n"
                    f"Problematic line: {e.text.strip() if e.text else 'unknown'}"
                )

            for check, msg in [
                ("import triton" in code or "from triton" in code, "MISSING: import triton"),
                (
                    "triton.language" in code or "import triton.language" in code,
                    "MISSING: import triton.language",
                ),
                ("@triton.jit" in code, "MISSING: @triton.jit decorator"),
                ("class Model" in code, "MISSING: class Model"),
            ]:
                if not check:
                    return msg

            warps_match = re.search(r"num_warps\s*=\s*(\d+)", code)
            if warps_match:
                num_warps = int(warps_match.group(1))
                if num_warps <= 0 or (num_warps & (num_warps - 1)) != 0:
                    return f"INVALID num_warps={num_warps}: Must be a power of 2."

            for block_name in [
                "BLOCK_M",
                "BLOCK_N",
                "BLOCK_K",
                "BLOCK_SIZE_M",
                "BLOCK_SIZE_N",
                "BLOCK_SIZE_K",
            ]:
                block_match = re.search(rf"{block_name}\s*[=:]\s*(\d+)", code)
                if block_match:
                    block_size = int(block_match.group(1))
                    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
                        return f"INVALID {block_name}={block_size}: Must be a power of 2."

            if executor and input_shapes:
                try:
                    comparison = executor.compare_kernels(
                        original_code=original_code,
                        optimized_code=code,
                        kernel_name=kernel_name,
                        input_shapes=input_shapes,
                        flop=flop,
                        dtype=dtype,
                        input_dtypes=input_dtypes,
                    )

                    logger.debug(
                        f"compare_kernels result: speedup={comparison.speedup}, "
                        f"orig_time={getattr(comparison, 'original_time_us', 'N/A')}, "
                        f"opt_time={getattr(comparison, 'optimized_time_us', 'N/A')}, "
                        f"correct={comparison.optimized_correct}, "
                        f"is_slower={comparison.is_slower}"
                    )

                    if not comparison.optimized_correct:
                        return comparison.feedback_message or "Optimized kernel failed."

                    if comparison.is_slower:
                        slowdown = (
                            1.0 / comparison.speedup if comparison.speedup > 0 else float("inf")
                        )
                        return (
                            f"PERFORMANCE REGRESSION: {slowdown:.2f}x SLOWER.\n"
                            f"Original: {comparison.original_time_us:.2f}μs\n"
                            f"Optimized: {comparison.optimized_time_us:.2f}μs"
                        )

                    logger.info(
                        f"Optimization verified: {comparison.speedup:.2f}x speedup "
                        f"({comparison.original_tflops or 0:.2f} -> {comparison.optimized_tflops or 0:.2f} TFLOPS)"
                    )
                    if best is not None:
                        best.offer(code, comparison.speedup, comparison)
                    return SUCCESS_MESSAGE

                except Exception as e:
                    return f"RUNTIME ERROR: {e!s}"

            logger.warning("No executor available - accepting based on static checks only")
            return SUCCESS_MESSAGE

        return compile_and_verify

    def optimize_stage(
        self,
        code: str,
        stage: OptimizationStage,
        analysis: KernelAnalysis,
        xpu_config: dict,
        kernel_name: str | None = None,
        input_shapes: list[tuple[int, ...]] | None = None,
        spec_dims: dict[str, int] | None = None,
        flop: float | None = None,
        dtype=None,
        pytorch_code: str = "",
        init_args: list | None = None,
        vtune_report: str = "",
        perf_context: dict | None = None,
        input_dtypes: list | None = None,
    ) -> StageResult:
        """
        Apply a single optimization stage using ReAct.

        The  agent iteratively generates and validates optimizations until
        the compile_and_verify tool returns SUCCESS_MESSAGE or max_iterations
        is reached.

        Args:
            code: Current Triton code
            stage: Stage to apply (e.g., DTYPE_FIX, BLOCK_POINTERS, DEVICE_SPECIFIC)
            analysis: Kernel analysis results with detected issues
            xpu_config: XPU configuration (num_warps, tile sizes, etc.)
            kernel_name: Kernel function name (optional for Model-based kernels)
            input_shapes: Input shapes for runtime testing
            flop: FLOP count for TFLOPS calculation
            dtype: Torch dtype for input tensors

        Returns:
            StageResult with optimized code and metrics
        """
        logger.info(f"Applying optimization stage: {stage.value}")
        logger.info(f"  input_shapes: {input_shapes}")
        logger.info(f"  flop: {flop}")
        logger.info(f"  dtype: {dtype}")

        original_code = code

        # Get relevant issues for this stage
        stage_issues = self._get_stage_issues(analysis, stage)
        if not stage_issues:
            logger.info(f"No issues for stage {stage.value}, skipping")
            return StageResult(
                stage=stage,
                success=True,
                input_code=code,
                output_code=code,
                changes_made=["No changes needed"],
                reasoning="No optimization opportunities found for this stage",
            )

        # Get knowledge patterns for this stage
        knowledge_patterns = self._get_stage_patterns(stage)

        from xe_forge.config import get_config
        from xe_forge.core.device_query import format_device_config_for_llm

        _cfg = get_config()
        xpu_config_text = format_device_config_for_llm(xpu_config, _cfg.device_config.device)

        # Create verification tool. `best` collects every candidate the tool proves
        # on hardware, so a stage that verifies a winner mid-loop and then answers
        # with something worse still returns the winner instead of its input.
        best = _BestCandidate()
        verify_tool = self._create_verify_tool(
            original_code=original_code,
            kernel_name=kernel_name,
            input_shapes=input_shapes,
            flop=flop,
            dtype=dtype,
            spec_dims=spec_dims,
            input_dtypes=input_dtypes,
            best=best,
        )

        # Create ReAct agent for this optimization
        if self.dsl == DSL.SYCL:
            sig = SyclOptimizationReActSignature
        elif self.dsl == DSL.MLIR:
            sig = MlirOptimizationReActSignature
        else:
            sig = OptimizationReActSignature
        react_agent = dspy.ReAct(
            signature=sig,
            tools=[verify_tool],
            max_iters=self.max_iterations,
        )

        try:
            logger.info(f"Starting ReAct optimization (max {self.max_iterations} iterations)")

            result = react_agent(
                original_code=original_code,
                current_code=code,
                stage=stage.value,
                issues=stage_issues,
                knowledge_patterns=knowledge_patterns,
                xpu_config=xpu_config_text,
            )

            # Extract optimized code from result
            if not hasattr(result, "optimized_code") or result.optimized_code is None:
                logger.error("Agent didn't return code")
                return self._keep_best_or_fail(
                    stage, original_code, best, "Agent failed to produce optimized code"
                )

            optimized_code: str = result.optimized_code.code

            trajectory = result.trajectory if hasattr(result, "trajectory") else {}

            # Verify the final code directly (don't rely on trajectory)

            success = False
            speedup = None
            metrics_before = None
            metrics_after = None
            last_error = None

            # Check syntax first
            if self.dsl not in (DSL.SYCL, DSL.MLIR) and not self._is_valid_python(optimized_code):
                last_error = "Final code has invalid Python syntax"
            elif not self._is_valid_kernel(optimized_code):
                last_error = "Final code is not a valid kernel"
            elif self.executor and (self.dsl in (DSL.SYCL, DSL.MLIR) or input_shapes):
                # Runtime verification
                try:
                    if self.dsl == DSL.SYCL:
                        _dims = spec_dims or dict(
                            zip(("M", "N", "K"), _extract_gemm_dims(input_shapes), strict=False)
                        )
                        comparison = self.executor.compare_kernels(
                            original_code=original_code,
                            optimized_code=optimized_code,
                            dims=_dims,
                        )
                    elif self.dsl == DSL.MLIR:
                        comparison = self.executor.compare_kernels(
                            original_code=original_code,
                            optimized_code=optimized_code,
                            flop=flop,
                        )
                    else:
                        comparison = self.executor.compare_kernels(
                            original_code=original_code,
                            optimized_code=optimized_code,
                            kernel_name=kernel_name,
                            input_shapes=input_shapes,
                            flop=flop,
                            dtype=dtype,
                            input_dtypes=input_dtypes,
                        )

                    if not comparison.optimized_correct:
                        last_error = "Optimized kernel produces incorrect results"
                    elif comparison.is_slower:
                        slowdown = (
                            1.0 / comparison.speedup if comparison.speedup > 0 else float("inf")
                        )
                        last_error = f"Optimized kernel is {slowdown:.2f}x slower"
                    else:
                        # Success!
                        success = True
                        speedup = comparison.speedup

                        # Build metrics dicts
                        metrics_before = None
                        metrics_after = None

                        # Only create metrics if we have all values
                        if (
                            comparison.original_time_us is not None
                            and comparison.original_tflops is not None
                        ):
                            metrics_before = {
                                "time_us": comparison.original_time_us,
                                "tflops": comparison.original_tflops,
                            }

                        if (
                            comparison.optimized_time_us is not None
                            and comparison.optimized_tflops is not None
                        ):
                            metrics_after = {
                                "time_us": comparison.optimized_time_us,
                                "tflops": comparison.optimized_tflops,
                            }
                except Exception as e:
                    last_error = f"Runtime verification failed: {e}"
            else:
                # No executor - accept if syntax valid
                success = True
                logger.warning("No executor available - accepting based on syntax check only")

            # The agent's final answer is only one of the candidates this stage
            # measured. If some earlier iteration proved a FASTER kernel on the GPU,
            # keep that instead of the answer (or instead of the untouched input when
            # the answer failed) — it is a hardware-verified result either way.
            if best.code is not None and (not success or (speedup or 1.0) < best.speedup):
                if success:
                    logger.info(
                        "Stage %s: final answer %.3fx is worse than an earlier verified "
                        "candidate %.3fx — keeping the earlier one",
                        stage.value,
                        speedup or 1.0,
                        best.speedup,
                    )
                    note = (
                        f"Kept a GPU-verified intermediate candidate ({best.speedup:.3f}x) "
                        f"because the agent's final answer was slower "
                        f"({(speedup or 1.0):.3f}x)."
                    )
                else:
                    logger.info(
                        "Stage %s: final answer rejected (%s), but an earlier iteration "
                        "verified %.3fx — keeping that candidate",
                        stage.value,
                        last_error,
                        best.speedup,
                    )
                    note = (
                        f"Kept a GPU-verified intermediate candidate ({best.speedup:.3f}x) "
                        f"because the agent's final answer was rejected: {last_error}"
                    )
                    self._dump_kernel(stage, optimized_code)

                metrics_before, metrics_after = _metrics_from_comparison(best.comparison)
                return StageResult(
                    stage=stage,
                    success=True,
                    input_code=original_code,
                    output_code=best.code,
                    changes_made=[*self._extract_changes_from_trajectory(trajectory), note],
                    reasoning=self._extract_reasoning_from_trajectory(trajectory),
                    speedup=best.speedup,
                    metrics_before=metrics_before,
                    metrics_after=metrics_after,
                )

            if success:
                logger.info(f"Stage {stage.value} completed successfully")
                if speedup:
                    logger.info(f"  Speedup: {speedup:.2f}x")

                return StageResult(
                    stage=stage,
                    success=True,
                    input_code=original_code,
                    output_code=optimized_code,
                    changes_made=self._extract_changes_from_trajectory(trajectory),
                    reasoning=self._extract_reasoning_from_trajectory(trajectory),
                    speedup=speedup,
                    metrics_before=metrics_before,
                    metrics_after=metrics_after,
                )
            else:
                # Optimization failed
                logger.warning(f"Stage {stage.value} failed after {self.max_iterations} iterations")
                logger.warning(f"Last error: {last_error}")

                # Dump failed kernel for debugging
                self._dump_kernel(stage, optimized_code)

                return StageResult(
                    stage=stage,
                    success=False,
                    input_code=original_code,
                    output_code=original_code,  # Keep original on failure
                    error_message=f"Failed after {self.max_iterations} iterations: {last_error}",
                )

        except Exception as e:
            logger.error(f"ReAct optimization failed with exception: {e}")
            import traceback

            logger.debug(traceback.format_exc())

            return self._keep_best_or_fail(stage, original_code, best, str(e))

    def _keep_best_or_fail(
        self,
        stage: OptimizationStage,
        original_code: str,
        best: _BestCandidate,
        error: str,
    ) -> StageResult:
        """Fall back to a GPU-verified candidate, or report failure if there is none.

        Used on the paths where there is no final answer to check at all (the agent
        returned nothing, or the ReAct call raised) — an earlier iteration may still
        have proved a faster kernel on hardware, and that result is worth keeping.
        """
        if best.code is None:
            return StageResult(
                stage=stage,
                success=False,
                input_code=original_code,
                output_code=original_code,
                error_message=error,
            )

        logger.info(
            "Stage %s: no usable final answer (%s), keeping a GPU-verified candidate (%.3fx)",
            stage.value,
            error,
            best.speedup,
        )
        metrics_before, metrics_after = _metrics_from_comparison(best.comparison)
        return StageResult(
            stage=stage,
            success=True,
            input_code=original_code,
            output_code=best.code,
            changes_made=[
                f"Kept a GPU-verified intermediate candidate ({best.speedup:.3f}x); "
                f"the agent produced no usable final answer: {error}"
            ],
            speedup=best.speedup,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
        )

    def _dump_kernel(self, stage: OptimizationStage, code: str) -> None:
        """Dump failed kernel code to file for debugging."""
        import os
        from datetime import datetime

        dump_dir = os.environ.get("TRITON_OPT_DUMP_DIR", "./outputs/kernels")
        os.makedirs(dump_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dump_dir}/{stage.value}_failed_{timestamp}.py"

        try:
            with open(filename, "w") as f:
                f.write(f"# Stage: {stage.value}\n")
                f.write("# Status: FAILED\n")
                f.write(f"# Timestamp: {timestamp}\n\n")
                f.write(code)
            logger.info(f"Dumped failed kernel to: {filename}")
        except Exception as e:
            logger.warning(f"Failed to dump kernel: {e}")

    def _is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.debug(f"Syntax error at line {e.lineno}: {e.msg}")
            return False

    def _is_valid_kernel(self, code: str) -> bool:
        """Check if code looks like a valid kernel for the current DSL."""
        if self.dsl == DSL.SYCL:
            return "#include" in code and ("sycl" in code.lower() or "cutlass" in code.lower())
        if self.dsl == DSL.MLIR:
            return "gpu.launch_func" in code and "func.func @main" in code
        has_triton_import = "import triton" in code or "from triton" in code
        has_kernel = "@triton.jit" in code or "class Model" in code
        return has_triton_import and has_kernel

    def _extract_changes_from_trajectory(self, trajectory: dict) -> list[str]:
        """Extract changes made from the agent's trajectory thoughts."""
        changes = []
        for key, value in sorted(trajectory.items()):
            if key.startswith("thought_"):
                thought = str(value).strip()
                # Extract meaningful parts of thoughts
                if any(
                    word in thought.lower()
                    for word in [
                        "applied",
                        "changed",
                        "replaced",
                        "added",
                        "removed",
                        "optimized",
                        "fixed",
                        "converted",
                        "updated",
                    ]
                ):
                    # Truncate very long thoughts
                    if len(thought) > 500:
                        thought = thought[:500] + "..."
                    changes.append(thought)

        return changes if changes else ["Optimization applied via ReAct"]

    def _extract_reasoning_from_trajectory(self, trajectory: dict) -> str:
        """Extract reasoning from the agent's trajectory thoughts."""
        thoughts = []
        for key, value in sorted(trajectory.items()):
            if key.startswith("thought_"):
                thought = str(value).strip()
                if len(thought) > 100:
                    thought = thought[:100] + "..."
                thoughts.append(thought)

        if not thoughts:
            return "ReAct optimization completed"

        return " → ".join(thoughts)

    def _get_stage_issues(
        self, analysis: KernelAnalysis, stage: OptimizationStage
    ) -> list[DetectedIssue]:
        """Get issues relevant to this stage."""
        return [
            issue
            for issue in analysis.detected_issues
            if get_stage_for_issue(issue.issue_type) == stage
        ]

    def _get_stage_patterns(self, stage: OptimizationStage) -> str:
        """Get knowledge patterns for a stage."""
        if self.knowledge_base:
            return self.knowledge_base.format_for_stage(stage)
        else:
            return (
                f"No specific patterns available for {stage.value}. Apply general best practices."
            )
