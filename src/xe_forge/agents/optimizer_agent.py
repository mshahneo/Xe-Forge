"""
Optimizer Agent - Uses CoVeR for iterative kernel optimization with tool-based verification.
Relies on LLM built-in knowledge instead of local YAML knowledge base.
The pipeline still builds the list of detected issues and passes them to each stage.
"""

import ast
import json
import logging
import re
from pathlib import Path

import dspy

from xe_forge.agents.base import Optimizer
from xe_forge.agents.cover import CoVeR
from xe_forge.knowledge.loader import KnowledgeBase
from xe_forge.models import (
    DSL,
    OptimizationStage,
    StageResult,
)

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


def _verify_sycl(code, original_code, executor, input_shapes, spec_dims=None):
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
            return SUCCESS_MESSAGE
        except Exception as e:
            return f"RUNTIME ERROR: {e!s}"

    logger.warning("No executor - accepting SYCL code based on static checks only")
    return SUCCESS_MESSAGE


SUCCESS_MESSAGE = "Success! Optimization verified and kernel is faster."


def _verify_mlir(code, original_code, executor, flop=None):
    """Verify an XeGPU WG-level MLIR kernel: structure check + runtime compare.

    The kernel file is self-contained (host @main + embedded CPU reference),
    so correctness is the optimized kernel's own [ALLCLOSE: TRUE]; speedup uses
    rtclock timing when the harness prints it.
    """
    # Cheap structural pre-checks before paying for lowering+execution.
    if "gpu.launch_func" not in code:
        return "MISSING: module must keep the gpu.launch_func kernel launch."
    if "func.func @main" not in code:
        return "MISSING: module must keep the @main host harness (correctness oracle)."
    if "[ALLCLOSE" not in original_code or "[ALLCLOSE" not in code:
        # Original may verify via printMemref instead; only warn for optimized.
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
            return SUCCESS_MESSAGE
        except Exception as e:
            return f"RUNTIME ERROR: {e!s}"

    logger.warning("No executor - accepting MLIR code based on static checks only")
    return SUCCESS_MESSAGE


class OptimizationSignature(dspy.Signature):
    """Apply optimization transformation to Triton kernel.

    You are an expert Triton kernel optimizer for Intel XPU with deep knowledge
    of GPU programming, numerical linear algebra, and high-performance computing.

    Optimize the kernel for maximum performance while producing numerically
    equivalent outputs. You may change the algorithm if outputs are equivalent.
    Maintain the same Model class signature including weights shapes and names.

    === STAGE-SPECIFIC GUIDANCE ===
    ALGORITHMIC: mathematical simplifications, CSE, loop-invariant hoisting,
      caching intermediates, reorder associative ops, tree reductions,
      exploit GEMM structure (symmetric, triangular, low-rank).
    DTYPE_FIX: float64->float32, proper accumulator precision, remove
      unnecessary type conversions.
    FUSION: fuse kernel launches, elementwise chains, reduction+elementwise.
    MEMORY_ACCESS: fix uncoalesced access, remove transposes from inner loops,
      add boundary checks, reduce register pressure.
    BLOCK_POINTERS: use tl.make_block_ptr(), boundary_check=(0,1) tuple format,
      tl.advance() for pointer updates.
    XPU_SPECIFIC: BLOCK_M=256, BLOCK_N=256, BLOCK_K=32, num_warps=32,
      GROUP_SIZE_M swizzling.
      GRF MODE: grf_mode is a compiler option, NOT a triton.Config() kwarg.
      Declare it as tl.constexpr in the kernel signature:
        grf_mode: tl.constexpr  (values: "default", "128", "256", "auto")
      Use "auto" — it automatically selects 256-GRF when register spill > 1000 bytes.
      256-GRF requires num_warps <= 32 (halved thread occupancy).
    PERSISTENT_KERNEL: persistent kernel pattern, tune NUM_PROGS.
    DISCOVERY: apply the open-ended optimization described in the issues field.
      This is a novel optimization not covered by standard stages. Follow the
      proposal exactly, preserving all numerical equivalences.

    === CODE REQUIREMENTS ===
    - Include ALL imports, @triton.jit decorator, kernel function, Model class
    - num_warps must be power of 2; block sizes must be powers of 2
    - NEVER replace @triton.jit kernels with torch.matmul, torch.mm, torch.bmm,
      or any vendor library (oneDNN, cuBLAS, MKL). Keep all original Triton kernels.
    """

    original_code: str = dspy.InputField(desc="Original Triton kernel code for reference")
    current_code: str = dspy.InputField(desc="Current Triton kernel code to optimize")
    stage: str = dspy.InputField(desc="Optimization stage to apply")
    issues: str = dspy.InputField(desc="Specific issues to fix in this stage")
    xpu_config: str = dspy.InputField(desc="Intel XPU configuration parameters")
    problem_context: str = dspy.InputField(
        desc="Problem context: input tensor shapes, dtype, Model init args, and FLOP count. "
        "Use this to choose appropriate tile sizes, understand memory footprint, "
        "and reason about whether the kernel is compute-bound or memory-bound."
    )
    performance_context: str = dspy.InputField(
        desc="Current execution performance: original baseline time, current time after "
        "previous stages, and speedup achieved so far. Use this to understand how much "
        "headroom remains and whether the kernel is close to hardware peak. "
        "Empty string if not yet measured."
    )
    vtune_report: str = dspy.InputField(
        desc="VTune profiling report (Markdown). Empty string if not available. "
        "Use hotspot and memory-access data to guide which optimizations matter most."
    )
    knowledge_base_context: str = dspy.InputField(
        desc="Relevant optimization patterns, constraints, and examples from the knowledge base "
        "for this stage. Empty string if KB is disabled. "
        "IMPORTANT: follow the patterns and constraints listed here precisely — "
        "they are validated optimizations for Intel XPU."
    )
    optimized_code: dspy.Code["python"] = dspy.OutputField(
        desc="Complete optimized Triton kernel code with all imports, decorators, kernel, and Model class."
    )


class AlgorithmicOptimizationSignature(dspy.Signature):
    """Apply algorithmic / mathematical optimization to a Triton kernel.

    You are an expert in numerical linear algebra, compiler optimizations, and
    high-performance GPU kernel design.

    Transform the kernel to perform FEWER FLOPs and/or FEWER memory accesses
    while producing numerically equivalent results.

    Think about:
    1. Matrix structure exploitation (symmetric, triangular, diagonal, low-rank, sparse)
    2. Associative / distributive law rewrites to reduce FLOPs
    3. Common sub-expression elimination
    4. Loop-invariant code hoisting
    5. Caching intermediates in registers vs recomputing
    6. Tree reductions vs serial reductions
    7. Algebraic simplification of fused computations

    Maintain the Model class signature. Produce equivalent outputs.

    === CODE REQUIREMENTS ===
    - Include ALL imports, @triton.jit decorator, kernel function, Model class
    - NEVER replace @triton.jit kernels with torch.matmul, torch.mm, or any vendor library.
    """

    original_code: str = dspy.InputField(desc="Original Triton kernel code for reference")
    current_code: str = dspy.InputField(desc="Current Triton kernel code to optimize")
    pytorch_code: str = dspy.InputField(
        desc="Original PyTorch implementation for context (may be empty)"
    )
    issues: str = dspy.InputField(desc="Specific algorithmic issues identified by analysis")
    xpu_config: str = dspy.InputField(desc="Intel XPU configuration parameters")
    problem_context: str = dspy.InputField(
        desc="Problem context: input tensor shapes, dtype, Model init args, and FLOP count. "
        "Use this to understand problem scale, memory footprint, and compute intensity."
    )
    performance_context: str = dspy.InputField(
        desc="Current execution performance: original baseline time and speedup so far. "
        "Empty string if not yet measured."
    )
    knowledge_base_context: str = dspy.InputField(
        desc="Relevant algorithmic patterns and examples from the knowledge base. "
        "Empty string if KB is disabled. Follow these patterns precisely."
    )
    optimized_code: dspy.Code["python"] = dspy.OutputField(
        desc="Complete optimized Triton kernel with algorithmic improvements."
    )


class AutotuneSignature(dspy.Signature):
    """Add or improve @triton.autotune configuration for a Triton kernel.

    You are an expert in Triton kernel autotuning for Intel XPU.

    Your task: Add or improve the @triton.autotune decorator so the kernel
    automatically selects the best configuration at runtime.

    You will receive:
    - The current kernel code
    - Hardware information (compute units, memory, capabilities)
    - Problem shapes (M, N, K dimensions)
    - A set of suggested autotune configurations generated from hardware analysis

    Your job:
    1. Add @triton.autotune decorator with a good set of configs to search.
    2. Use the suggested configs as a starting point but ADD more configs
       based on your knowledge of what works well for this kernel type.
    3. Include the key= argument so configs are re-evaluated when shapes change.
    4. Ensure num_warps and num_stages are included in each config.
    5. Ensure BLOCK sizes are powers of 2 and appropriate for the hardware.
    6. For Intel XPU, always include at least one config with num_warps=32
       and large tile sizes (256x256).
    7. Remove any hardcoded meta-parameters that are now covered by autotune.
    8. Keep the kernel functionally equivalent.

    Tips for good autotune configs:
    - Vary BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K across powers of 2
    - Include both small tiles (64x64) for small problems and large tiles
      (256x256) for large problems
    - Vary num_warps: try 4, 8, 16, 32
    - Vary num_stages: try 2, 3, 4
    - Include GROUP_SIZE_M for L2 cache swizzling
    - Use key= with the shape arguments that affect tiling
    - Do NOT put grf_mode in triton.Config() — it causes TypeError at runtime.
      grf_mode is a compiler option: declare it as tl.constexpr in the kernel
      signature. Use grf_mode="auto" (auto-selects 256-GRF if spill > 1000 bytes)
      or grf_mode="256" for large register file. Requires num_warps <= 32.

    === CODE REQUIREMENTS ===
    - Include ALL imports (torch, triton, triton.language as tl)
    - Include @triton.autotune with configs list and key
    - Include @triton.jit on the kernel
    - Include the Model class with forward() method
    """

    original_code: str = dspy.InputField(desc="Original Triton kernel code for reference")
    current_code: str = dspy.InputField(desc="Current Triton kernel code to add autotune to")
    issues: str = dspy.InputField(desc="Specific autotuning issues identified by analysis")
    xpu_config: str = dspy.InputField(desc="Intel XPU hardware info and recommended parameters")
    suggested_autotune_configs: str = dspy.InputField(
        desc="Suggested autotune configurations from hardware/shape analysis (use as starting point)"
    )
    problem_shapes: str = dspy.InputField(
        desc="Problem dimensions (M, N, K) and input shapes for key= argument"
    )
    problem_context: str = dspy.InputField(
        desc="Problem context: input tensor shapes, dtype, Model init args, and FLOP count. "
        "Use this to choose config search space breadth and understand compute intensity."
    )
    performance_context: str = dspy.InputField(
        desc="Current execution performance: original baseline time and speedup so far. "
        "Empty string if not yet measured."
    )
    knowledge_base_context: str = dspy.InputField(
        desc="Relevant autotuning patterns and constraints from the knowledge base. "
        "Empty string if KB is disabled. Follow these patterns precisely."
    )
    optimized_code: dspy.Code["python"] = dspy.OutputField(
        desc="Complete Triton kernel with @triton.autotune. Must include all imports, autotune decorator with configs and key, kernel, and Model class."
    )


class SyclOptimizationSignature(dspy.Signature):
    """Optimize a SYCL/CUTLASS C++ kernel for Intel XPU.

    You are an expert in SYCL, CUTLASS/XeTLA, Intel XPU GPU architecture,
    and high-performance C++ kernel optimization.

    Optimize the kernel for maximum performance while producing numerically
    equivalent outputs. You may change template parameters, dispatch policies,
    data types, and memory layouts.

    === SYCL/CUTLASS OPTIMIZATION KNOBS ===
    - TileShape: Shape<_M, _N, _K> — try 256x256x32, 128x128x64, 128x256x32
    - PipelineStages: 2, 3, or 4 — more prefetching vs register pressure
    - MMA Atom: XE_DPAS_TT<SubgroupSize, AccumType, InputType> — SubgroupSize 4 or 8
    - Dispatch Policy: MainloopXeL1Staged (L1 cached), MainloopXeL0Staged (uncached)
    - Data types: bfloat16_t/half_t inputs, float/bfloat16_t accumulators
    - Memory layout: RowMajor vs ColumnMajor for A, B, C, D
    - Epilogue: LinearCombination, bias, activation via FusionCallbacks
    - GmemTiledCopy: void (auto) or explicit copy atoms

    === STAGE-SPECIFIC GUIDANCE ===
    ALGORITHMIC: mathematical simplifications, CSE, loop-invariant hoisting,
      exploit GEMM structure (symmetric, triangular, low-rank).
    DTYPE_FIX: use bfloat16_t/half_t inputs, float accumulators, avoid double.
    FUSION: fuse into CUTLASS epilogue callbacks — LinearCombination, bias, activation.
    MEMORY_ACCESS: fix layout mismatch (RowMajor vs ColumnMajor), increase PipelineStages
      for better prefetching, reduce register pressure.
    DEVICE_SPECIFIC: TileShape 256x256x32 or 128x128x64, PipelineStages=2-3,
      XE_DPAS_TT<8, float, bfloat16_t>, MainloopXeL1Staged dispatch policy.
    DISCOVERY: apply the open-ended optimization described in the issues field.

    === CODE REQUIREMENTS ===
    - Must be complete, valid SYCL C++ with all #include directives
    - Must use cutlass namespace and CUTLASS template types
    - Must include ExampleRunner template and main() function
    - Must compile with icpx -fsycl
    - Keep the same output format (Cutlass GEMM Performance line)
    """

    original_code: str = dspy.InputField(desc="Original SYCL/CUTLASS C++ kernel for reference")
    current_code: str = dspy.InputField(desc="Current SYCL C++ kernel code to optimize")
    stage: str = dspy.InputField(desc="Optimization stage to apply")
    issues: str = dspy.InputField(desc="Specific issues to fix in this stage")
    xpu_config: str = dspy.InputField(desc="Intel XPU configuration parameters")
    problem_context: str = dspy.InputField(
        desc="Problem context: GEMM dimensions (M, N, K), FLOP count, compute intensity."
    )
    performance_context: str = dspy.InputField(
        desc="Current execution performance: baseline time, speedup so far. Empty if not measured."
    )
    vtune_report: str = dspy.InputField(
        desc="VTune profiling report. Empty string if not available."
    )
    knowledge_base_context: str = dspy.InputField(
        desc="Relevant optimization patterns from knowledge base. Empty if KB disabled."
    )
    optimized_code: dspy.Code["cpp"] = dspy.OutputField(
        desc="Complete optimized SYCL C++ kernel. Must include all #includes, templates, ExampleRunner, and main()."
    )


class SyclAlgorithmicOptimizationSignature(dspy.Signature):
    """Apply algorithmic / mathematical optimization to a SYCL/CUTLASS C++ kernel.

    You are an expert in numerical linear algebra, compiler optimizations, and
    high-performance GPU kernel design for Intel XPU.

    Transform the kernel to perform FEWER FLOPs and/or FEWER memory accesses
    while producing numerically equivalent results.

    Think about:
    1. Matrix structure exploitation (symmetric, triangular, diagonal, low-rank)
    2. Associative / distributive law rewrites to reduce FLOPs
    3. Common sub-expression elimination in template expressions
    4. Data layout optimization (RowMajor vs ColumnMajor)
    5. Batch dimension exploitation

    === CODE REQUIREMENTS ===
    - Must be complete, valid SYCL C++ with all #include directives
    - Keep CUTLASS GEMM structure (GemmUniversalAdapter, ExampleRunner, main)
    - Must compile with icpx -fsycl
    """

    original_code: str = dspy.InputField(desc="Original SYCL C++ kernel for reference")
    current_code: str = dspy.InputField(desc="Current SYCL C++ kernel to optimize")
    pytorch_code: str = dspy.InputField(desc="Reference description. May be empty.")
    issues: str = dspy.InputField(desc="Specific algorithmic issues identified")
    xpu_config: str = dspy.InputField(desc="Intel XPU configuration parameters")
    problem_context: str = dspy.InputField(desc="Problem context: dimensions, FLOP count.")
    performance_context: str = dspy.InputField(desc="Current performance. Empty if not measured.")
    knowledge_base_context: str = dspy.InputField(desc="KB patterns. Empty if disabled.")
    optimized_code: dspy.Code["cpp"] = dspy.OutputField(
        desc="Complete optimized SYCL C++ kernel with algorithmic improvements."
    )


class MlirOptimizationSignature(dspy.Signature):
    """Optimize an XeGPU workgroup-level MLIR kernel for Intel XPU.

    You are an expert in MLIR, the XeGPU dialect, Intel Xe GPU architecture
    (Xe-cores, subgroups, DPAS systolic array, 2D block loads), and
    high-performance kernel tuning.

    The input is a self-contained MLIR module: a host `func.func @main` that
    allocates inputs, launches a `gpu.module` kernel via `gpu.launch_func`, and
    verifies the result against an embedded CPU reference (printing
    `[ALLCLOSE: TRUE]`). Optimize for maximum performance while keeping the
    result numerically equivalent.

    === HARD CONSTRAINTS (do not break these) ===
    - Edit ONLY the `gpu.module` kernel body, the `#xegpu.layout` attributes,
      and the `gpu.launch_func` grid/block geometry.
    - DO NOT modify the `@main` harness, the CPU reference function, the input
      fill values, or the `[ALLCLOSE]` check — they are the correctness oracle.
    - Keep the kernel entry symbol name and its `gpu.launch_func` reference
      consistent.
    - The module must stay valid MLIR that lowers cleanly through
      `--gpu-lower-to-xevm-pipeline=xegpu-op-level=workgroup`.
    - The `sg_layout` product (number of subgroups) must remain consistent with
      the threads-per-workgroup in `gpu.launch_func`.

    === XeGPU WG-LEVEL OPTIMIZATION KNOBS ===
    - `#xegpu.layout<sg_layout=[..], sg_data=[..], inst_data=[..]>`: distribute
      the workgroup tile across subgroups. Tune sg_layout / sg_data so each
      subgroup's tile matches DPAS-friendly shapes; inst_data should match the
      DPAS instruction shape (e.g. [8,16], [16,16]).
    - Prefetch: insert `xegpu.prefetch_nd` with `l1_hint/l2_hint/l3_hint =
      #xegpu.cache_hint<cached>` ahead of `load_nd` to hide memory latency.
    - K-loop tiling: adjust the `scf.for` step / accumulator tile to trade
      register pressure against reuse.
    - Launch geometry: blocks = output tiles, threads = subgroups-per-WG * 16.

    === STAGE-SPECIFIC GUIDANCE ===
    DTYPE_FIX: f16/bf16 inputs with f32 DPAS accumulators; avoid f64.
    FUSION: fold elementwise epilogue ops into the kernel after the dpas loop.
    MEMORY_ACCESS: add/strengthen prefetch and cache hints; coalesce loads;
      pick sg_data tiles that map to contiguous 2D block loads.
    DEVICE_SPECIFIC: tune sg_layout / sg_data / inst_data and launch geometry
      for the Xe-core (DPAS shape, subgroup size 16). This stage ALSO owns the
      per-op float flags the Xe backend cares about: `fastmath<fast>` on hot-loop
      `math.exp` and on `arith.mulf/subf/addf/divf`, and `math.exp2` in place of
      `math.exp`. Those are one-token edits worth more than most layout changes —
      apply the knowledge-base patterns for them before retuning tiles.
    DISCOVERY: apply the open-ended optimization described in the issues field.

    === HOW TO SUBMIT A CHANGE ===
    Prefer `edits`: a JSON list of exact-anchor replacements. Copy `old` verbatim
    from the module, make it unique, and write `UNCHANGED` in `optimized_code`.
    Re-emitting the whole module corrupts unrelated syntax and wastes the attempt.
    Use `optimized_code` only for a restructuring too large to express as anchors.
    """

    original_code: str = dspy.InputField(desc="Original MLIR module for reference")
    current_code: str = dspy.InputField(desc="Current MLIR module to optimize")
    stage: str = dspy.InputField(desc="Optimization stage to apply")
    issues: str = dspy.InputField(desc="Specific issues to fix in this stage")
    xpu_config: str = dspy.InputField(desc="Intel XPU configuration parameters")
    problem_context: str = dspy.InputField(
        desc="Problem context: tile shapes, dimensions, FLOP count, compute intensity."
    )
    performance_context: str = dspy.InputField(
        desc="Current execution performance: baseline time, speedup so far. Empty if not measured."
    )
    vtune_report: str = dspy.InputField(desc="VTune profiling report. Empty if not available.")
    knowledge_base_context: str = dspy.InputField(
        desc="Relevant optimization patterns from knowledge base. Empty if KB disabled."
    )
    edits: str = dspy.OutputField(
        desc="PREFERRED. JSON list of exact-anchor replacements: "
        '[{"old": "<text copied verbatim from the module>", "new": "<replacement>"}]. '
        "Each `old` must appear EXACTLY ONCE — include indentation and enough "
        "surrounding text to be unique. Write NONE only if you truly need to "
        "rewrite the whole module."
    )
    optimized_code: dspy.Code["mlir"] = dspy.OutputField(
        desc="Write the single word UNCHANGED when you supplied `edits`. Otherwise the "
        "complete optimized MLIR module, with the @main harness and reference "
        "unchanged; edit only the gpu.module kernel, #xegpu.layout attrs, and "
        "launch geometry."
    )


class MlirAlgorithmicOptimizationSignature(dspy.Signature):
    """Apply algorithmic / mathematical optimization to an XeGPU MLIR kernel.

    You are an expert in numerical linear algebra and MLIR XeGPU kernel design
    for Intel XPU.

    Transform the `gpu.module` kernel to perform FEWER FLOPs and/or FEWER memory
    accesses while producing numerically equivalent results that still pass the
    embedded `[ALLCLOSE: TRUE]` check.

    Think about:
    1. Matrix structure exploitation (symmetric, triangular, diagonal).
    2. Associative / distributive rewrites that reduce the K-loop work.
    3. Common sub-expression elimination across subgroup tiles.
    4. Eliminating redundant loads / recomputation inside the scf.for loop.

    === HARD CONSTRAINTS ===
    - Edit ONLY the `gpu.module` kernel and its `#xegpu.layout` attrs.
    - DO NOT touch the `@main` harness, fill values, or CPU reference.
    - Must stay valid MLIR that lowers through the workgroup XeVM pipeline.

    === HOW TO SUBMIT A CHANGE ===
    Prefer `edits`: a JSON list of exact-anchor replacements. Copy `old` verbatim
    from the module, make it unique, and write `UNCHANGED` in `optimized_code`.
    Re-emitting the whole module corrupts unrelated syntax and wastes the attempt.
    """

    original_code: str = dspy.InputField(desc="Original MLIR module for reference")
    current_code: str = dspy.InputField(desc="Current MLIR module to optimize")
    pytorch_code: str = dspy.InputField(desc="Reference description. May be empty.")
    issues: str = dspy.InputField(desc="Specific algorithmic issues identified")
    xpu_config: str = dspy.InputField(desc="Intel XPU configuration parameters")
    problem_context: str = dspy.InputField(desc="Problem context: dimensions, FLOP count.")
    performance_context: str = dspy.InputField(desc="Current performance. Empty if not measured.")
    knowledge_base_context: str = dspy.InputField(desc="KB patterns. Empty if disabled.")
    edits: str = dspy.OutputField(
        desc="PREFERRED. JSON list of exact-anchor replacements: "
        '[{"old": "<text copied verbatim from the module>", "new": "<replacement>"}]. '
        "Each `old` must appear EXACTLY ONCE. Write NONE only if you truly need to "
        "rewrite the whole module."
    )
    optimized_code: dspy.Code["mlir"] = dspy.OutputField(
        desc="Write the single word UNCHANGED when you supplied `edits`. Otherwise the "
        "complete optimized MLIR module with algorithmic improvements; harness unchanged."
    )


def _build_performance_context(perf_context: dict | None) -> str:
    """Format perf_context dict into a human-readable string for the LLM prompt."""
    if not perf_context:
        return ""
    orig_ms = perf_context.get("original_ms")
    orig_tf = perf_context.get("original_tflops")
    curr_ms = perf_context.get("current_ms")
    so_far = perf_context.get("speedup_so_far")

    lines = ["=== Performance Context ==="]
    if orig_ms:
        lines.append(
            f"Original baseline:  {orig_ms:.3f} ms"
            + (f"  ({orig_tf:.2f} TFLOPS)" if orig_tf else "")
        )
    if curr_ms and curr_ms != orig_ms:
        lines.append(f"Current (after previous stages): {curr_ms:.3f} ms")
    if so_far and so_far != 1.0:
        lines.append(f"Speedup so far: {so_far:.2f}x")
        if so_far < 1.0:
            lines.append("  WARNING: previous stages made the kernel slower — be conservative.")
        elif so_far >= 2.0:
            lines.append("  Good progress. Focus on remaining bottlenecks.")
        else:
            lines.append("  Moderate progress. Significant headroom likely remains.")
    elif orig_ms and (not so_far or so_far == 1.0):
        lines.append("No speedup from previous stages yet.")
    stage_best = perf_context.get("stage_best_so_far")
    if stage_best:
        lines.append(
            f"Best achieved this stage so far: {stage_best:.3f}x — "
            f"your next attempt must beat this to be accepted."
        )
    return "\n".join(lines)


def _has_cpu_return(code: str) -> bool:
    """Return True if kernel_function/forward returns a CPU tensor.
    .cpu() in a return is always wrong — output must stay on XPU.
    Note: .xpu() IS valid (moves to XPU) but should not be needed in a return.
    """
    if re.search(r"return\s+\S+\.cpu\(\)", code):
        return True
    if re.search(r"return\s+\S+\.to\(.[Cc][Pp][Uu].", code):
        return True
    return False


#: What the LLM writes in `optimized_code` when it wants the `edits` path instead.
EDITS_SENTINEL = "UNCHANGED"

#: A `%`, digits, then a letter/underscore/`$` — e.g. `%35_f32`. Illegal in MLIR: a
#: numeric SSA id must be ALL digits, and a named one must start with a non-digit.
_DIGIT_LED_SSA = re.compile(r"%(\d+)([A-Za-z_$][\w$]*)")

#: Every SSA name already present, so a repair cannot collide with one.
_ANY_SSA = re.compile(r"%[\w$.\-]+")


def _repair_digit_led_ssa_names(code: str) -> tuple[str, str]:
    """Rename `%35_f32`-style values to `%v35_f32`. Returns `(code, note)`.

    Why this exists: converting an accumulator to f32 means introducing values derived
    from existing ones, and our kernels number their values. The natural name for a
    derived value is then `%<n>_f32` — which does not parse, because MLIR reads `%35`
    as a numeric id and then wants `=`. Two stages on 4k flash attention lost 5 of 10
    attempts to exactly this, three times in a row with identical text, while the
    conversion around it was correct.

    A KB constraint spelling out the rule did not stop it: the model followed the rest
    of the recipe (separate f32 constants, updated dpas signature) and still wrote
    `%34_f32`. So repair it here instead of asking. The rename is purely syntactic and
    behaviour-preserving — every occurrence of a given name maps to one new name — so
    it cannot change what the kernel computes.

    `note` is empty when nothing needed repair.
    """
    found = {m.group(0) for m in _DIGIT_LED_SSA.finditer(code)}
    if not found:
        return code, ""

    taken = set(_ANY_SSA.findall(code))
    mapping: dict[str, str] = {}
    for old in sorted(found):
        digits, suffix = _DIGIT_LED_SSA.match(old).groups()
        new = f"%v{digits}{suffix}"
        n = 0
        while new in taken or new in mapping.values():
            n += 1
            new = f"%v{digits}{suffix}_{n}"
        mapping[old] = new
        taken.add(new)

    # One pass over the maximal-identifier matches, so each name is replaced whole and
    # a short name can never clobber part of a longer one.
    repaired = _DIGIT_LED_SSA.sub(lambda m: mapping[m.group(0)], code)
    sample = ", ".join(f"{o} -> {mapping[o]}" for o in sorted(mapping)[:3])
    return repaired, f"renamed {len(mapping)} digit-led SSA name(s): {sample}"


def _apply_anchored_edits(code: str, edits_str: str) -> tuple[str | None, str]:
    """Apply a JSON list of exact-anchor replacements to `code`.

    Returns `(new_code, "")` on success, or `(None, reason)` on failure. The
    reason is written for the LLM to read and retry against.

    A whole 22KB MLIR module does not survive being re-emitted by an LLM: a
    one-attribute change arrives with unrelated syntax corrupted somewhere else.
    So small changes travel as anchors instead. Every anchor must appear exactly
    once, which makes a stale or ambiguous anchor a hard error rather than a
    silent edit in the wrong place.
    """
    raw = (edits_str or "").strip()
    if not raw or raw.upper() == "NONE":
        return None, "no edits supplied"

    if "```" in raw:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"EDITS NOT VALID JSON at char {e.pos}: {e.msg}"

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        return None, 'EDITS must be a non-empty JSON list of {"old": ..., "new": ...}'

    out = code
    for i, ed in enumerate(parsed, 1):
        if not isinstance(ed, dict) or "old" not in ed or "new" not in ed:
            return None, f'edit {i}: needs both an "old" and a "new" key'
        old, new = str(ed["old"]), str(ed["new"])
        if not old:
            return None, f'edit {i}: "old" is empty — it must be text copied from the module'
        n = out.count(old)
        if n == 0:
            return None, (
                f'edit {i}: ANCHOR NOT FOUND. Copy "old" verbatim from the module, '
                f"including indentation. Got: {old[:120]!r}"
            )
        if n > 1:
            return None, (
                f"edit {i}: ANCHOR AMBIGUOUS ({n} matches). Add surrounding lines "
                f"to make it unique. Got: {old[:120]!r}"
            )
        if old == new:
            return None, f'edit {i}: "old" and "new" are identical — that is a no-op'
        out = out.replace(old, new, 1)

    if out == code:
        return None, "edits applied but the module is unchanged"
    return out, ""


def _extract_code_from_response(code_str):
    if code_str is None:
        return ""
    code = str(code_str)
    if "```python" in code:
        m = re.search(r"```python\s*(.*?)\s*```", code, re.DOTALL)
        if m:
            code = m.group(1)
    elif "```cpp" in code or "```c++" in code:
        m = re.search(r"```(?:cpp|c\+\+)\s*(.*?)\s*```", code, re.DOTALL)
        if m:
            code = m.group(1)
    elif "```" in code:
        m = re.search(r"```\s*(.*?)\s*```", code, re.DOTALL)
        if m:
            code = m.group(1)
    return code.strip()


class OptimizerAgent(Optimizer):
    """Applies optimization transformations using CoVeR with LLM knowledge."""

    def __init__(
        self,
        knowledge_base: "KnowledgeBase | None" = None,
        executor=None,
        validator=None,
        max_iterations=5,
        dsl: DSL | str = DSL.TRITON,
    ):
        self.executor = executor
        self.validator = validator
        self.max_iterations = max_iterations
        self.knowledge_base: KnowledgeBase | None = knowledge_base
        self.dsl = DSL(dsl) if isinstance(dsl, str) else dsl
        #: Where rejected attempts get written. Set by the pipeline; when it is
        #: None only the log line is emitted.
        self.attempts_dir: Path | None = None
        if not executor:
            logger.warning("No executor provided - kernels will NOT be verified at runtime!")

    def _attempt_log(self, stage, n, via, verdict, speedup, code: str | None = None) -> None:
        """Record one verify attempt, rejected ones included.

        A stage that burns its whole budget used to report only "no valid result in
        budget", so there was no way to tell a syntax failure from a slower kernel
        without re-running the stage for ~35 minutes. Every attempt now leaves a
        verdict line, and every rejected module is kept.
        """
        tag = getattr(stage, "value", str(stage))
        lines = (verdict or "").strip().splitlines()
        head = lines[0][:200] if lines else "(no verdict)"
        # A compiler verdict's first line is only the headline —
        # "LOWERING FAILED (imex-opt):". The reason is on a later line. Without it
        # the log read the same for five different failures.
        detail = next((ln.strip() for ln in lines[1:] if "error:" in ln), "")
        if not detail:
            detail = next((ln.strip() for ln in lines[1:] if ln.strip()), "")
        logger.info(
            "  attempt %d [%s] via %s: %s%s%s",
            n,
            tag,
            via,
            head,
            f" {detail[:200]}" if detail else "",
            f" ({speedup:.3f}x)" if speedup else "",
        )
        if not self.attempts_dir or code is None:
            return
        try:
            self.attempts_dir.mkdir(parents=True, exist_ok=True)
            ok = (verdict or "").strip() == SUCCESS_MESSAGE
            stem = f"{tag}_a{n}_{'ok' if ok else 'rejected'}"
            (self.attempts_dir / f"{stem}.mlir").write_text(code)
            (self.attempts_dir / f"{stem}.verdict.txt").write_text(
                f"stage: {tag}\nattempt: {n}\nvia: {via}\nspeedup: {speedup}\nverdict:\n{verdict}\n"
            )
        except OSError as e:
            logger.debug("could not write attempt artifact: %s", e)

    def _create_verify_tool(
        self,
        original_code,
        kernel_name,
        input_shapes,
        flop,
        dtype=None,
        init_args=None,
        skip_speedup_check=False,
        stage=None,
        baseline_ms: float | None = None,
        spec_dims=None,
        input_dtypes=None,
    ):
        executor = self.executor
        dsl = self.dsl
        # "base" is what anchors resolve against. It advances as the stage improves
        # the module, so a later edit anchors on the text the LLM was actually shown
        # and edits stack instead of reverting the previous win.
        last_accepted = {"comparison": None, "code": None, "base": original_code}
        attempt_log = self._attempt_log

        _verify_call_count = [0]

        def compile_and_verify(optimized_code: dspy.Code["python"], edits: str = "") -> str:
            _verify_call_count[0] += 1
            logger.debug("compile_and_verify call #%d", _verify_call_count[0])
            code = _extract_code_from_response(
                optimized_code.code if hasattr(optimized_code, "code") else str(optimized_code)
            )

            # Anchored edits win when supplied: they are the reliable way to change
            # a large module. `optimized_code` is only read when there are none.
            via = "whole-module"
            if dsl == DSL.MLIR:
                patched, why = _apply_anchored_edits(last_accepted["base"], edits)
                if patched is not None:
                    code, via = patched, "edits"
                elif why != "no edits supplied":
                    attempt_log(stage, _verify_call_count[0], "edits", why, None)
                    return f"EDITS REJECTED: {why}"
                elif code.strip().upper() == EDITS_SENTINEL:
                    msg = (
                        f"You wrote {EDITS_SENTINEL} in optimized_code but supplied no "
                        "edits. Provide the edits, or the full module instead."
                    )
                    attempt_log(stage, _verify_call_count[0], "edits", msg, None)
                    return msg

            if dsl == DSL.SYCL:
                result = _verify_sycl(code, original_code, executor, input_shapes, spec_dims)
                if result == SUCCESS_MESSAGE and executor:
                    _dims = spec_dims or dict(
                        zip(("M", "N", "K"), _extract_gemm_dims(input_shapes), strict=False)
                    )
                    try:
                        c = executor.compare_kernels(
                            original_code=original_code,
                            optimized_code=code,
                            dims=_dims,
                        )
                        last_accepted["comparison"] = c
                    except Exception:
                        pass
                return result

            if dsl == DSL.MLIR:
                # Repair before parsing, not after failing. An f32 retype is otherwise
                # correct and still dies on the name it gave a derived value.
                code, _repair = _repair_digit_led_ssa_names(code)
                if _repair:
                    logger.info("  repaired attempt %d: %s", _verify_call_count[0], _repair)
                result = _verify_mlir(code, original_code, executor, flop=flop)
                _spd = None
                if result == SUCCESS_MESSAGE and executor:
                    try:
                        c = executor.compare_kernels(
                            original_code=original_code,
                            optimized_code=code,
                            flop=flop,
                        )
                        last_accepted["comparison"] = c
                        _spd = c.speedup
                    except Exception:
                        pass
                if result == SUCCESS_MESSAGE:
                    # The edits path leaves `optimized_code` a sentinel, so the
                    # stage cannot recover the module from the prediction. Hand it
                    # over here instead.
                    last_accepted["code"] = code
                attempt_log(stage, _verify_call_count[0], via, result, _spd, code=code)
                return result

            # --- Triton path ---
            try:
                ast.parse(code)
            except SyntaxError as e:
                return f"SYNTAX ERROR at line {e.lineno}: {e.msg}"

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
                nw = int(warps_match.group(1))
                if nw <= 0 or (nw & (nw - 1)) != 0:
                    return f"INVALID num_warps={nw}: Must be power of 2."

            for bn in [
                "BLOCK_M",
                "BLOCK_N",
                "BLOCK_K",
                "BLOCK_SIZE_M",
                "BLOCK_SIZE_N",
                "BLOCK_SIZE_K",
            ]:
                bm = re.search(rf"{bn}\s*[=:]\s*(\d+)", code)
                if bm:
                    bs = int(bm.group(1))
                    if bs <= 0 or (bs & (bs - 1)) != 0:
                        return f"INVALID {bn}={bs}: Must be power of 2."

            if re.search(r"triton\.Config\s*\([^)]*grf_mode", code):
                return (
                    "INVALID: grf_mode cannot be passed to triton.Config(). "
                    "It is a compiler-level option, not a kernel meta-parameter. "
                    "To use large GRF: declare grf_mode: tl.constexpr in the kernel "
                    "signature (values: 'default', '128', '256', 'auto'). "
                    "'auto' is recommended — it recompiles with 256-GRF only if "
                    "register spill > 1000 bytes. Remove grf_mode from triton.Config()."
                )

            if stage is not None and stage.value == "fusion":
                for _vp in [
                    r"torch\.matmul",
                    r"torch\.mm\b",
                    r"torch\.bmm",
                    r"F\.linear\b",
                    r"torch\.nn\.functional\.linear",
                ]:
                    if re.search(_vp, code):
                        return (
                            "INVALID: fusion replaced a Triton kernel with a vendor call. "
                            "Keep all @triton.jit kernels. "
                            "Do not replace Triton kernels with torch.matmul/mm/bmm."
                        )

            if _has_cpu_return(code):
                return (
                    "INVALID: kernel_function/forward must NOT return a CPU tensor. "
                    "Remove .cpu() from all return statements. "
                    "The output tensor must stay on XPU."
                )

            original_kernels = set(re.findall(r"def\s+(\w+)\s*\(", original_code))
            original_jit_kernels = {
                name
                for name in original_kernels
                if "@triton.jit" in original_code
                and re.search(
                    rf"@triton\.jit[\s\S]{{0,200}}def\s+{re.escape(name)}\s*\(", original_code
                )
            }
            if original_jit_kernels:
                missing = [k for k in original_jit_kernels if f"def {k}" not in code]
                if missing:
                    return (
                        f"INVALID: Triton kernel(s) {missing} were removed from the optimized code. "
                        "You must keep all original @triton.jit kernels. "
                        "Do NOT replace Triton kernels with torch.matmul, torch.mm, "
                        "oneDNN, or any other vendor library call."
                    )

            if executor and input_shapes:
                try:
                    logger.info(
                        "Measuring: original vs optimized (call #%d)", _verify_call_count[0]
                    )
                    comparison = executor.compare_kernels(
                        original_code=original_code,
                        optimized_code=code,
                        kernel_name=kernel_name,
                        input_shapes=input_shapes,
                        flop=flop,
                        dtype=dtype,
                        init_args=init_args,
                        input_dtypes=input_dtypes,
                    )
                    if comparison.original_time_us:
                        logger.info("  original : %.2f µs", comparison.original_time_us)
                    if comparison.optimized_time_us:
                        logger.info("  optimized: %.2f µs", comparison.optimized_time_us)
                    if not comparison.optimized_correct:
                        return comparison.feedback_message or "Optimized kernel failed."
                    # Use pipeline baseline for regression check (avoids JIT warmup noise)
                    if not skip_speedup_check:
                        if baseline_ms and comparison.optimized_time_us:
                            true_spd = (baseline_ms * 1000) / comparison.optimized_time_us
                        else:
                            true_spd = comparison.speedup
                        if true_spd < 1.0:
                            sd = 1.0 / true_spd if true_spd > 0 else float("inf")
                            return (
                                f"PERFORMANCE REGRESSION: {sd:.2f}x SLOWER. Try different approach."
                            )
                    # Compute speedup relative to true baseline if available
                    # (avoids JIT warmup / thermal noise in re-measured original)
                    if baseline_ms and comparison.optimized_time_us:
                        true_speedup = (baseline_ms * 1000) / comparison.optimized_time_us
                        logger.info(
                            f"Optimization verified: {true_speedup:.2f}x speedup "
                            f"(vs true baseline {baseline_ms:.3f}ms)"
                        )
                    else:
                        logger.info(f"Optimization verified: {comparison.speedup:.2f}x speedup")
                    last_accepted["comparison"] = comparison
                    last_accepted["baseline_ms"] = baseline_ms
                    return SUCCESS_MESSAGE
                except Exception as e:
                    return f"RUNTIME ERROR: {e}"
            return SUCCESS_MESSAGE

        tool = dspy.Tool(
            func=compile_and_verify,
            name="compile_and_verify",
            desc=(
                f'Compiles and verifies optimized kernel. Returns "{SUCCESS_MESSAGE}" on '
                "success. Pass `edits` (a JSON list of exact-anchor replacements) to change "
                "a large module; pass `optimized_code` only for a whole rewrite."
            ),
        )
        return tool, last_accepted

    def optimize_stage(
        self,
        code,
        stage,
        analysis,
        xpu_config,
        kernel_name=None,
        input_shapes=None,
        spec_dims=None,
        flop=None,
        dtype=None,
        pytorch_code="",
        init_args=None,
        vtune_report="",
        perf_context: dict | None = None,
        input_dtypes=None,
    ):
        logger.info(f"Applying optimization stage: {stage.value}")
        original_code = code

        stage_issues = self._get_stage_issues(analysis, stage)
        if not stage_issues:
            return StageResult(
                stage=stage,
                success=True,
                input_code=code,
                output_code=code,
                changes_made=["No changes needed"],
            )

        issues_text = "\n".join(
            [
                f"- {i.issue_type.value}: {i.description}\n  Fix: {i.suggested_fix}\n  Speedup: {i.estimated_speedup or 'Unknown'}"
                + (
                    f"\n  Proposal: {i.open_ended_proposal}"
                    if hasattr(i, "open_ended_proposal") and i.open_ended_proposal
                    else ""
                )
                for i in stage_issues
            ]
        )

        from xe_forge.config import get_config
        from xe_forge.core.device_query import format_device_config_for_llm

        _cfg = get_config()
        xpu_text = format_device_config_for_llm(xpu_config, _cfg.device_config.device)

        CORRECTNESS_ONLY_STAGES = {}
        skip_speedup = stage in CORRECTNESS_ONLY_STAGES

        _baseline_ms = perf_context.get("original_ms") if perf_context else None

        verify_tool, last_accepted = self._create_verify_tool(
            original_code,
            kernel_name,
            input_shapes,
            flop,
            dtype,
            init_args=init_args,
            skip_speedup_check=skip_speedup,
            stage=stage,
            baseline_ms=_baseline_ms,
            spec_dims=spec_dims,
            input_dtypes=input_dtypes,
        )

        problem_ctx = self._build_problem_context(input_shapes, dtype, init_args, flop)
        perf_ctx = _build_performance_context(perf_context)

        kb_context = self._get_stage_patterns(stage)
        if kb_context:
            logger.info(
                "KB context for %s: %d chars, %d patterns/constraints",
                stage.value,
                len(kb_context),
                kb_context.count("###") + kb_context.count("CONSTRAINT"),
            )
        else:
            logger.debug("No KB context for stage %s (KB disabled or empty)", stage.value)

        if self.dsl == DSL.SYCL:
            if stage == OptimizationStage.ALGORITHMIC:
                sig = SyclAlgorithmicOptimizationSignature
                kwargs = {
                    "original_code": original_code,
                    "current_code": code,
                    "pytorch_code": pytorch_code or "",
                    "issues": issues_text,
                    "xpu_config": xpu_text,
                    "problem_context": problem_ctx,
                    "performance_context": perf_ctx,
                    "knowledge_base_context": kb_context,
                }
            else:
                sig = SyclOptimizationSignature
                kwargs = {
                    "original_code": original_code,
                    "current_code": code,
                    "stage": stage.value,
                    "issues": issues_text,
                    "xpu_config": xpu_text,
                    "problem_context": problem_ctx,
                    "performance_context": perf_ctx,
                    "vtune_report": vtune_report or "",
                    "knowledge_base_context": kb_context,
                }
        elif self.dsl == DSL.MLIR:
            if stage == OptimizationStage.ALGORITHMIC:
                sig = MlirAlgorithmicOptimizationSignature
                kwargs = {
                    "original_code": original_code,
                    "current_code": code,
                    "pytorch_code": pytorch_code or "",
                    "issues": issues_text,
                    "xpu_config": xpu_text,
                    "problem_context": problem_ctx,
                    "performance_context": perf_ctx,
                    "knowledge_base_context": kb_context,
                }
            else:
                sig = MlirOptimizationSignature
                kwargs = {
                    "original_code": original_code,
                    "current_code": code,
                    "stage": stage.value,
                    "issues": issues_text,
                    "xpu_config": xpu_text,
                    "problem_context": problem_ctx,
                    "performance_context": perf_ctx,
                    "vtune_report": vtune_report or "",
                    "knowledge_base_context": kb_context,
                }
        elif stage == OptimizationStage.ALGORITHMIC:
            sig = AlgorithmicOptimizationSignature
            kwargs = {
                "original_code": original_code,
                "current_code": code,
                "pytorch_code": pytorch_code or "",
                "issues": issues_text,
                "xpu_config": xpu_text,
                "problem_context": problem_ctx,
                "performance_context": perf_ctx,
                "knowledge_base_context": kb_context,
            }
        elif stage == OptimizationStage.AUTOTUNING:
            sig = AutotuneSignature
            suggested_configs = self._build_autotune_configs(xpu_config, input_shapes)
            problem_shapes = self._build_problem_shapes(input_shapes)
            kwargs = {
                "original_code": original_code,
                "current_code": code,
                "issues": issues_text,
                "xpu_config": xpu_text,
                "suggested_autotune_configs": suggested_configs,
                "problem_shapes": problem_shapes,
                "problem_context": problem_ctx,
                "performance_context": perf_ctx,
                "knowledge_base_context": kb_context,
            }
        else:
            sig = OptimizationSignature
            kwargs = {
                "original_code": original_code,
                "current_code": code,
                "stage": stage.value,
                "issues": issues_text,
                "xpu_config": xpu_text,
                "problem_context": problem_ctx,
                "performance_context": perf_ctx,
                "vtune_report": vtune_report or "",
                "knowledge_base_context": kb_context,
            }

        cover = CoVeR(
            signature=sig,
            tools=[verify_tool],
            success=SUCCESS_MESSAGE,
            max_iters=self.max_iterations,
            use_raw_fixer_output=True,
        )

        best_code = None
        best_spd = None
        best_mb = best_ma = best_traj = None
        iters_used = 0
        attempt_history: list[str] = []  # what each run tried and achieved

        current_code_for_run = code

        try:
            while iters_used < self.max_iterations:
                remaining = self.max_iterations - iters_used
                cover.max_iters = remaining

                run_kwargs = {**kwargs, "current_code": current_code_for_run}

                # Anchors resolve against exactly what the LLM is shown this run, and
                # a module accepted by an earlier run must not be picked up as if it
                # came from this one. The cached timing goes too: it belongs to the
                # previous run's candidate, which may have been rejected.
                last_accepted["base"] = current_code_for_run
                last_accepted["code"] = None
                last_accepted["comparison"] = None

                result = cover(**run_kwargs)

                traj = result.trajectory if hasattr(result, "trajectory") else {}
                thoughts_this_run = sum(1 for k in traj if k.startswith("thought_"))
                iters_used += max(1, thoughts_this_run)

                if not hasattr(result, "optimized_code") or result.optimized_code is None:
                    break

                code_obj = result.optimized_code
                candidate = _extract_code_from_response(
                    code_obj.code if hasattr(code_obj, "code") else str(code_obj)
                )

                # On the edits path `optimized_code` is only the sentinel, so the
                # real module is whatever the verify tool last accepted.
                if self.dsl == DSL.MLIR:
                    _patched = last_accepted.get("code")
                    if _patched and (not candidate or candidate.strip().upper() == EDITS_SENTINEL):
                        candidate = _patched
                    elif not candidate:
                        break

                ok, spd, mb, ma, err = self._final_verify(
                    original_code,
                    candidate,
                    kernel_name,
                    input_shapes,
                    flop,
                    dtype,
                    init_args=init_args,
                    skip_speedup_check=skip_speedup,
                    cached_comparison=last_accepted["comparison"],
                    baseline_ms=_baseline_ms,
                    spec_dims=spec_dims,
                )

                # Record this attempt for feedback to the next run
                _attempt_thoughts = self._reasoning(traj) if traj else ""
                _attempt_summary = (
                    f"Attempt {len(attempt_history) + 1}: "
                    + (f"achieved {spd:.3f}x" if ok and spd else f"failed ({err})")
                    + (f" | approach: {_attempt_thoughts[:150]}" if _attempt_thoughts else "")
                )
                attempt_history.append(_attempt_summary)

                if not ok:
                    logger.debug(f"Run failed ({err}), stopping best-of loop")
                    break

                # Require 2% improvement on ALL runs (first and subsequent)
                # This eliminates noise-based false improvements from timing variance
                _MIN_IMPROVEMENT = 1.02
                _is_improvement = (
                    spd is not None and spd > _MIN_IMPROVEMENT
                    if best_spd is None
                    else spd is not None and spd > best_spd * _MIN_IMPROVEMENT
                )
                # Also stop if code is identical to previous best (LLM stuck)
                _code_identical = best_code is not None and candidate == best_code
                if _code_identical:
                    logger.info(f"Stage {stage.value}: LLM produced identical code — stopping")
                    break
                if _is_improvement:
                    logger.info(
                        f"Stage {stage.value} new best: {spd:.2f}x"
                        + (f" (was {best_spd:.2f}x)" if best_spd is not None else "")
                    )
                    best_code, best_spd, best_mb, best_ma, best_traj = (
                        candidate,
                        spd,
                        mb,
                        ma,
                        traj,
                    )
                    current_code_for_run = candidate
                    # Rebuild performance_context with updated speedup
                    # so the next CoVeR iteration knows where it stands
                    if perf_context:
                        updated_perf = dict(perf_context)
                    else:
                        updated_perf = {}
                    _orig_ms = updated_perf.get("original_ms")
                    if spd and _orig_ms:
                        updated_perf["current_ms"] = _orig_ms / spd
                    updated_perf["speedup_so_far"] = spd
                    updated_perf["stage_best_so_far"] = spd
                    perf_ctx = _build_performance_context(updated_perf)
                else:
                    # A measured miss is not a reason to quit with budget left. The
                    # first run often spends itself on the knob the issue text names;
                    # on flash-attention that was a layout tweak at 0.99x, and the
                    # stage stopped with 3 of 5 iterations unspent, so the float
                    # flags worth 1.83x were never tried. Keep the old base — a
                    # slower candidate must not become what the next run edits — and
                    # let the loop condition end the stage.
                    _best_str = f"{best_spd:.2f}x" if best_spd is not None else "none"
                    _spd_str = f"{spd:.2f}x" if spd is not None else "N/A"
                    logger.info(
                        f"Stage {stage.value} no improvement ({_spd_str} vs best "
                        f"{_best_str}); {self.max_iterations - iters_used} iterations left"
                    )

                # Feed the attempt back either way, or the next run re-tries the knob
                # that was just measured. Only the improving branch used to do this,
                # so a miss taught the LLM nothing.
                history_text = "\n".join(attempt_history[-3:])  # last 3 attempts
                _issues_with_history = issues_text + (
                    f"\n\n=== Previous attempts this stage ===\n{history_text}\n"
                    "Try a DIFFERENT approach to beat the current best. Do not repeat "
                    "an approach already measured above."
                    if attempt_history
                    else ""
                )
                kwargs = {
                    **kwargs,
                    "performance_context": perf_ctx,
                    "issues": _issues_with_history,
                }

            if best_code is not None:
                logger.info(
                    f"Stage {stage.value} OK — best {best_spd:.2f}x"
                    f" ({self.max_iterations - (self.max_iterations - iters_used)} iters used)"
                )
                return StageResult(
                    stage=stage,
                    success=True,
                    input_code=original_code,
                    output_code=best_code,
                    changes_made=self._changes(best_traj or {}),
                    reasoning=self._reasoning(best_traj or {}),
                    speedup=best_spd,
                    metrics_before=best_mb,
                    metrics_after=best_ma,
                )
            else:
                logger.warning(f"Stage {stage.value} failed: no valid result in budget")
                return StageResult(
                    stage=stage,
                    success=False,
                    input_code=original_code,
                    output_code=original_code,
                    error_message="No valid optimization found within iteration budget",
                )

        except Exception as e:
            logger.error(f"CoVeR failed: {e}")
            return StageResult(
                stage=stage,
                success=False,
                input_code=original_code,
                output_code=original_code,
                error_message=str(e),
            )

    @staticmethod
    def _extract_example_code(code: str, max_chars: int = 2500) -> str:
        """Extract key patterns from example code: header comments, __init__, cache method, forward, kernel_function."""
        import re

        sections = []

        # File-level optimization summary
        header = re.match(r"((?:#[^\n]*\n)+)", code)
        if header:
            key = [
                line
                for line in header.group(1).split("\n")
                if any(
                    kw in line.lower()
                    for kw in [
                        "fix",
                        "key",
                        "optim",
                        "cache",
                        "pack",
                        "fuse",
                        "speedup",
                        "important",
                        "fp16",
                        "fp32",
                        "1)",
                        "2)",
                        "3)",
                    ]
                )
            ]
            if key:
                sections.append("# Key optimizations:\n" + "\n".join(key[:10]))

        # Model.__init__
        m = re.search(
            r"(    def __init__\(self[^)]*\):.*?)(?=\n    def |\nclass |\Z)", code, re.DOTALL
        )
        if m:
            sections.append("# __init__:\n" + m.group(1)[:500])

        # Parameter cache/pack method
        m = re.search(
            r"(    def (?:_move_params_once|_ensure_cache|_ensure_device|_build_cache)[^:]*:.*?)"
            r"(?=\n    def |\nclass |\Z)",
            code,
            re.DOTALL,
        )
        if m:
            sections.append("# Parameter caching:\n" + m.group(1)[:900])

        # forward()
        m = re.search(
            r"(    def forward\(self[^)]*\):.*?)(?=\n    def |\nclass |\Z)", code, re.DOTALL
        )
        if m:
            sections.append("# forward():\n" + m.group(1)[:350])

        # kernel_function
        m = re.search(
            r"(def kernel_function\([^)]*\):.*?)(?=\nclass |\ndef (?!kernel)|\Z)", code, re.DOTALL
        )
        if m:
            sections.append("# kernel_function:\n" + m.group(1)[:450])

        # If nothing matched (e.g. pure kernel file), fall back to first max_chars
        if not sections:
            return code[:max_chars]

        result = "\n\n".join(s for s in sections if s.strip())
        return result[:max_chars]

    def _get_stage_patterns(self, stage: OptimizationStage) -> str:
        """Return KB context: constraints + patterns + compact example summaries."""
        if self.knowledge_base is None:
            return ""
        try:
            # 1. Get constraints + patterns (no full code) from format_for_stage
            full = self.knowledge_base.format_for_stage(stage)
            if not full:
                return ""
            # Strip the full code section — replace with compact example summaries
            split_marker = "FULL CODE EXAMPLES FOR"
            if split_marker in full:
                full = full[: full.index(split_marker)].rstrip()

            # 2. Append compact example summaries (name + optimizations, no code)
            examples = self.knowledge_base.examples_for_stage(stage)
            if examples:
                ex_lines = [f"\n\nRELEVANT EXAMPLES FOR {stage.value.upper()}:"]
                ex_lines.append("(these are real optimized kernels — apply the same patterns)")
                for ex in examples[:4]:  # cap at 4 examples
                    ex_lines.append(f"\n## {ex.name}")
                    ex_lines.append(f"Description: {ex.description.strip()[:150]}")
                    if ex.optimizations_applied:
                        ex_lines.append("Optimizations applied:")
                        for opt in ex.optimizations_applied[:8]:
                            ex_lines.append(f"  - {opt}")
                    if ex.expected_speedup:
                        ex_lines.append(f"Expected speedup: {ex.expected_speedup}")
                    # Include code: full if small, smart-extracted if large
                    if ex.optimized_code:
                        code_to_show = self._extract_example_code(ex.optimized_code)
                        if code_to_show:
                            ex_lines.append("Key patterns from optimized code:")
                            ex_lines.append("```python")
                            ex_lines.append(code_to_show)
                            ex_lines.append("```")
                full += "\n".join(ex_lines)

            # Hard cap at 14000 chars (~3500 tokens)
            if len(full) > 14000:
                full = full[:14000] + "\n... [KB content truncated]"
            logger.debug("KB patterns for %s: %d chars", stage.value, len(full))
            return full
        except Exception as e:
            logger.debug("KB context retrieval failed: %s", e)
            return ""

    def _build_problem_context(self, input_shapes, dtype, init_args, flop):
        lines = ["=== Problem Context ==="]

        if input_shapes:
            lines.append(f"Input tensors ({len(input_shapes)}):")
            for i, shape in enumerate(input_shapes):
                numel = 1
                for d in shape:
                    numel *= d
                bytes_per_elem = 2 if dtype and "16" in str(dtype) else 4
                mem_mb = numel * bytes_per_elem / (1024 * 1024)
                lines.append(f"  Input {i}: shape={shape}, elements={numel:,}, ~{mem_mb:.1f} MB")

            total_mem = 0
            bytes_per_elem = 2 if dtype and "16" in str(dtype) else 4
            for shape in input_shapes:
                n = 1
                for d in shape:
                    n *= d
                total_mem += n * bytes_per_elem
            lines.append(f"  Total input memory: ~{total_mem / (1024 * 1024):.1f} MB")
        else:
            lines.append("Input tensors: not available")

        if dtype:
            lines.append(f"Data type: {dtype}")

        if init_args:
            lines.append(f"Model init args: {init_args}")
            if len(init_args) == 1:
                lines.append(f"  (likely head_dim or hidden_dim = {init_args[0]})")

        if flop:
            lines.append(f"FLOP count: {flop:,.0f}")
            if flop > 1e12:
                lines.append(f"  = {flop / 1e12:.2f} TFLOP")
            elif flop > 1e9:
                lines.append(f"  = {flop / 1e9:.2f} GFLOP")

            if input_shapes:
                total_bytes = 0
                bytes_per_elem = 2 if dtype and "16" in str(dtype) else 4
                for shape in input_shapes:
                    n = 1
                    for d in shape:
                        n *= d
                    total_bytes += n * bytes_per_elem
                if total_bytes > 0:
                    ai = flop / total_bytes
                    lines.append(f"  Arithmetic intensity: {ai:.1f} FLOPs/byte")
                    if ai > 100:
                        lines.append(
                            "  -> Compute-bound: focus on algorithmic and compute optimizations"
                        )
                    elif ai > 10:
                        lines.append("  -> Balanced: both compute and memory optimizations matter")
                    else:
                        lines.append(
                            "  -> Memory-bound: focus on memory access patterns and data reuse"
                        )

        return "\n".join(lines)

    def _build_autotune_configs(self, xpu_config, input_shapes):
        try:
            from xe_forge.core.xpu_query import (
                extract_mnk_from_shapes,
                get_autotune_configs,
            )

            if input_shapes and len(input_shapes) >= 1:
                M, N, K = extract_mnk_from_shapes(input_shapes)
                if M and N and K:
                    configs = get_autotune_configs(M, N, K)
                    lines = [f"Suggested configs for M={M}, N={N}, K={K}:"]
                    for i, cfg in enumerate(configs):
                        lines.append(f"  Config {i + 1}: {cfg}")
                    return "\n".join(lines)
        except Exception as e:
            logger.debug(f"Could not generate autotune configs: {e}")

        lines = ["No shape-specific configs available. Suggested search space:"]
        bm = xpu_config.get("BLOCK_SIZE_M", 256)
        bn = xpu_config.get("BLOCK_SIZE_N", 256)
        bk = xpu_config.get("BLOCK_SIZE_K", 32)
        nw = xpu_config.get("num_warps", 32)
        lines.append(f"  Base: BLOCK_M={bm}, BLOCK_N={bn}, BLOCK_K={bk}, num_warps={nw}")
        lines.append("  Also try: BLOCK_M/N in [64, 128, 256], BLOCK_K in [32, 64]")
        lines.append("  Also try: num_warps in [4, 8, 16, 32], num_stages in [2, 3, 4]")
        return "\n".join(lines)

    def _build_problem_shapes(self, input_shapes):
        if not input_shapes:
            return "No input shapes available. Use appropriate key= based on kernel arguments."
        try:
            from xe_forge.core.xpu_query import extract_mnk_from_shapes

            M, N, K = extract_mnk_from_shapes(input_shapes)
            lines = [f"Input shapes: {input_shapes}"]
            if M and N and K:
                lines.append(f"Extracted dimensions: M={M}, N={N}, K={K}")
                lines.append(
                    "Use key= with the stride/shape args that correspond "
                    "to M, N, K so autotune re-runs when problem size changes."
                )
            return "\n".join(lines)
        except Exception:
            return f"Input shapes: {input_shapes}"

    def _final_verify(
        self,
        orig,
        opt,
        kn,
        shapes,
        flop,
        dtype,
        init_args=None,
        skip_speedup_check=False,
        cached_comparison=None,
        baseline_ms: float | None = None,
        spec_dims=None,
    ):
        if self.dsl == DSL.SYCL:
            if "#include" not in opt:
                return False, None, None, None, "Not valid SYCL C++"
        elif self.dsl == DSL.MLIR:
            if "gpu.launch_func" not in opt or "func.func @main" not in opt:
                return False, None, None, None, "Not a valid self-contained MLIR kernel"
        else:
            if not self._valid_py(opt):
                return False, None, None, None, "Invalid Python syntax"
            if not self._valid_triton(opt):
                return False, None, None, None, "Not valid Triton"
        if self.executor and (self.dsl in (DSL.SYCL, DSL.MLIR) or shapes):
            try:
                if cached_comparison is not None:
                    c = cached_comparison
                elif self.dsl == DSL.MLIR:
                    c = self.executor.compare_kernels(
                        original_code=orig,
                        optimized_code=opt,
                        flop=flop,
                    )
                elif self.dsl == DSL.SYCL:
                    _dims = spec_dims or dict(
                        zip(("M", "N", "K"), _extract_gemm_dims(shapes), strict=False)
                    )
                    c = self.executor.compare_kernels(
                        original_code=orig,
                        optimized_code=opt,
                        dims=_dims,
                    )
                else:
                    c = self.executor.compare_kernels(
                        original_code=orig,
                        optimized_code=opt,
                        kernel_name=kn,
                        input_shapes=shapes,
                        flop=flop,
                        dtype=dtype,
                        init_args=init_args,
                    )
                if not c.optimized_correct:
                    return False, None, None, None, "Incorrect results"
                if getattr(c, "lowered_identical", False):
                    # No-op edit: lowers to identical IR. Reject regardless of the
                    # (noisy) measured time so a phantom speedup can't be recorded
                    # via the baseline_ms path below.
                    return False, None, None, None, "no-op (lowers to identical IR)"
                if c.is_slower and not skip_speedup_check:
                    sd = 1.0 / c.speedup if c.speedup > 0 else float("inf")
                    return False, None, None, None, f"{sd:.2f}x slower"
                mb = None
                if c.original_time_us and c.original_tflops:
                    mb = {"time_us": c.original_time_us, "tflops": c.original_tflops}
                ma = None
                if c.optimized_time_us and c.optimized_tflops:
                    ma = {"time_us": c.optimized_time_us, "tflops": c.optimized_tflops}
                if baseline_ms and c.optimized_time_us:
                    spd = (baseline_ms * 1000) / c.optimized_time_us
                else:
                    spd = c.speedup
                return True, spd, mb, ma, None
            except Exception as e:
                return False, None, None, None, f"Verify failed: {e}"
        return True, None, None, None, None

    def _get_stage_issues(self, analysis, stage):
        from xe_forge.knowledge.patterns import get_stage_for_issue

        seen_types: set = set()
        result = []
        for i in analysis.detected_issues:
            if get_stage_for_issue(i.issue_type) != stage:
                continue
            # Deduplicate by issue_type — keep the first (highest severity) occurrence
            if i.issue_type not in seen_types:
                seen_types.add(i.issue_type)
                result.append(i)
            else:
                logger.debug("Skipping duplicate issue %s in stage %s", i.issue_type, stage.value)
        return result

    def _dump_kernel(self, stage, code):
        import os
        from datetime import datetime

        d = os.environ.get("TRITON_OPT_DUMP_DIR", "./outputs/kernels")
        os.makedirs(d, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            with open(f"{d}/{stage.value}_failed_{ts}.py", "w") as f:
                f.write(f"# Stage: {stage.value}\n# FAILED\n\n{code}")
        except Exception:
            pass

    def _valid_py(self, code):
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _valid_triton(self, code):
        has_triton = "import triton" in code or "from triton" in code
        has_kernel = "@triton.jit" in code or "class Model" in code
        return has_triton and has_kernel

    def _changes(self, traj):
        cs = []
        keywords = [
            "applied",
            "changed",
            "replaced",
            "added",
            "removed",
            "optimized",
            "fixed",
            "simplified",
            "cached",
            "hoisted",
            "fused",
            "reordered",
        ]
        for k, v in sorted(traj.items()):
            if k.startswith("thought_"):
                t = str(v).strip()
                if any(w in t.lower() for w in keywords):
                    cs.append(t[:500])
        return cs or ["Optimization applied via CoVeR"]

    def _reasoning(self, traj):
        ts = [str(v).strip()[:100] for k, v in sorted(traj.items()) if k.startswith("thought_")]
        return " -> ".join(ts) if ts else "CoVeR optimization completed"
