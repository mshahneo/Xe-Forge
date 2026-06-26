"""
Core components for kernel execution and validation

Provides KernelBench-style testing with accurate GPU timing including:
- L2 cache flushing between runs
- Hardware event-based timing
- Proper warmup and synchronization
- Comparison tools for CoVeR agent feedback
- YAML spec loading for test configurations
- Device hardware query for optimal kernel parameters (XPU, CUDA)
- Configurable correctness validation (via REQUIRE_CORRECTNESS, CORRECTNESS_RTOL, CORRECTNESS_ATOL)

Imports are LAZY (PEP 562): symbols resolve to their submodule only on first
access. This keeps `import xe_forge.core` (and any submodule import that runs
this package __init__) free of heavy/optional dependencies like torch — so
DSL backends that don't need torch (e.g. the MLIR/XeGPU executor) can be used
without a torch install present.
"""

from importlib import import_module
from typing import TYPE_CHECKING

# Map exported symbol -> submodule that defines it. Resolved on demand.
_SYMBOL_MODULES: dict[str, str] = {
    # device_query
    "CUDADeviceInfo": "device_query",
    "DeviceInfo": "device_query",
    "format_device_config_for_llm": "device_query",
    "get_device_config_for_pipeline": "device_query",
    "query_cuda_via_torch": "device_query",
    "query_device": "device_query",
    # executor (pulls torch)
    "ComparisonResult": "executor",
    "KernelBenchExecutor": "executor",
    "KernelExecutor": "executor",
    "create_executor_tool": "executor",
    # kernel_analyzer
    "AnalysisResult": "kernel_analyzer",
    "KernelAnalyzer": "kernel_analyzer",
    "format_analysis": "kernel_analyzer",
    # profiler
    "ProfileMetrics": "profiler",
    "ProfileResult": "profiler",
    "Recommendation": "profiler",
    "XPUProfiler": "profiler",
    # spec_loader
    "InputSpec": "spec_loader",
    "KernelSpec": "spec_loader",
    "VariantSpec": "spec_loader",
    "get_test_config_from_spec": "spec_loader",
    "load_spec": "spec_loader",
    "load_spec_from_string": "spec_loader",
    "parse_spec": "spec_loader",
    # sycl_executor
    "KernelType": "sycl_executor",
    "SyclComparisonResult": "sycl_executor",
    "SyclExecutor": "sycl_executor",
    # mlir_executor (torch-free)
    "MlirComparisonResult": "mlir_executor",
    "MlirExecutor": "mlir_executor",
    # trial_manager
    "TrialManager": "trial_manager",
    # validator
    "KernelValidator": "validator",
    "ValidationIssue": "validator",
    "format_issues": "validator",
    # xpu_query
    "XPUDeviceInfo": "xpu_query",
    "extract_mnk_from_shapes": "xpu_query",
    "format_xpu_config_for_llm": "xpu_query",
    "get_autotune_configs": "xpu_query",
    "get_optimal_params": "xpu_query",
    "get_xpu_config": "xpu_query",
    "get_xpu_config_dict": "xpu_query",
    "get_xpu_config_for_pipeline": "xpu_query",
    "print_xpu_info": "xpu_query",
}

__all__ = [
    "AnalysisResult",
    "CUDADeviceInfo",
    "ComparisonResult",
    "DeviceInfo",
    "InputSpec",
    "KernelAnalyzer",
    "KernelBenchExecutor",
    "KernelExecutor",
    "KernelSpec",
    "KernelType",
    "KernelValidator",
    "MlirComparisonResult",
    "MlirExecutor",
    "ProfileMetrics",
    "ProfileResult",
    "Recommendation",
    "SyclComparisonResult",
    "SyclExecutor",
    "TrialManager",
    "ValidationIssue",
    "VariantSpec",
    "XPUDeviceInfo",
    "XPUProfiler",
    "create_executor_from_config",
    "create_executor_tool",
    "extract_mnk_from_shapes",
    "format_analysis",
    "format_device_config_for_llm",
    "format_issues",
    "format_xpu_config_for_llm",
    "get_autotune_configs",
    "get_device_config_for_pipeline",
    "get_optimal_params",
    "get_test_config_from_spec",
    "get_xpu_config",
    "get_xpu_config_dict",
    "get_xpu_config_for_pipeline",
    "load_spec",
    "load_spec_from_string",
    "parse_spec",
    "print_xpu_info",
    "query_cuda_via_torch",
    "query_device",
]


def __getattr__(name: str):
    """Lazily import and cache an exported symbol from its submodule."""
    module_name = _SYMBOL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value  # cache for subsequent access
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:  # static type-checkers still see the real symbols
    from xe_forge.core.device_query import (
        CUDADeviceInfo,
        DeviceInfo,
        format_device_config_for_llm,
        get_device_config_for_pipeline,
        query_cuda_via_torch,
        query_device,
    )
    from xe_forge.core.executor import (
        ComparisonResult,
        KernelBenchExecutor,
        KernelExecutor,
        create_executor_tool,
    )
    from xe_forge.core.kernel_analyzer import AnalysisResult, KernelAnalyzer, format_analysis
    from xe_forge.core.mlir_executor import MlirComparisonResult, MlirExecutor
    from xe_forge.core.profiler import (
        ProfileMetrics,
        ProfileResult,
        Recommendation,
        XPUProfiler,
    )
    from xe_forge.core.spec_loader import (
        InputSpec,
        KernelSpec,
        VariantSpec,
        get_test_config_from_spec,
        load_spec,
        load_spec_from_string,
        parse_spec,
    )
    from xe_forge.core.sycl_executor import KernelType, SyclComparisonResult, SyclExecutor
    from xe_forge.core.trial_manager import TrialManager
    from xe_forge.core.validator import KernelValidator, ValidationIssue, format_issues
    from xe_forge.core.xpu_query import (
        XPUDeviceInfo,
        extract_mnk_from_shapes,
        format_xpu_config_for_llm,
        get_autotune_configs,
        get_optimal_params,
        get_xpu_config,
        get_xpu_config_dict,
        get_xpu_config_for_pipeline,
        print_xpu_info,
    )


def create_executor_from_config(
    config,
    kernel_type=None,
):
    """
    Create an executor with settings from Config.

    Returns SyclExecutor when dsl=sycl, MlirExecutor when dsl=mlir,
    KernelBenchExecutor otherwise. Imports are local so this stays lazy.
    """
    from xe_forge.models import DSL

    if config.device_config.dsl == DSL.SYCL:
        from xe_forge.core.sycl_executor import KernelType, SyclExecutor

        return SyclExecutor(
            verify=config.optimization.require_correctness,
            kernel_type=kernel_type or KernelType.GEMM,
        )
    if config.device_config.dsl == DSL.MLIR:
        from xe_forge.core.mlir_executor import MlirExecutor

        return MlirExecutor(
            require_correctness=config.optimization.require_correctness,
        )
    from xe_forge.core.executor import KernelBenchExecutor

    return KernelBenchExecutor(
        device=config.device_config.device,
        require_correctness=config.optimization.require_correctness,
        rtol=config.optimization.correctness_rtol,
        atol=config.optimization.correctness_atol,
    )
