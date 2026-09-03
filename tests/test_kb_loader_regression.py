"""Regression guard for the knowledge-base loader.

The loader is shared by every DSL. When the MLIR knowledge base is
reorganized, the Triton / SYCL / Gluon views must keep loading *exactly* the
same content, because their stage context strings go straight into LLM prompts —
a one-character change there can move a run's behaviour.

This test pins, for each (dsl, device) view, the number of patterns /
constraints / examples and a hash of every per-stage context string. Baselines
live in ``tests/data/kb_loader_baseline.json``.

If a change is *meant* to alter a non-MLIR view, regenerate deliberately:

    python -m tests.test_kb_loader_regression --update

and review the diff in the baseline file as part of the change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from xe_forge.knowledge.loader import load_knowledge_base
from xe_forge.models import OptimizationStage

REPO_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = REPO_ROOT / "knowledge_base"
BASELINE = Path(__file__).parent / "data" / "kb_loader_baseline.json"

# Every view the pipeline actually loads. mlir/linalg is a distinct view: the
# lowering agent loads it separately (see XeForgePipeline._lowering_kb).
VIEWS = [
    ("triton", "xpu"),
    ("triton", "cuda"),
    ("sycl", "xpu"),
    ("gluon", "xpu"),
    ("mlir", "xpu"),
    ("mlir", "linalg"),
]

# Views that must stay byte-identical while the MLIR KB is being reworked.
FROZEN_VIEWS = [v for v in VIEWS if v[0] != "mlir"]


def _snapshot() -> dict:
    snap: dict = {}
    for dsl, device in VIEWS:
        kb = load_knowledge_base(KB_DIR, dsl=dsl, device_type=device)
        snap[f"{dsl}/{device}"] = {
            "patterns": kb.entry_count,
            "constraints": kb.constraint_count,
            "examples": kb.example_count,
            "stage_pattern_counts": {
                s.value: len(kb.get_by_stage(s)) for s in OptimizationStage if kb.get_by_stage(s)
            },
            "stage_context_sha256": {
                s.value: hashlib.sha256(kb.format_for_stage(s).encode()).hexdigest()
                for s in OptimizationStage
            },
        }
    return snap


@pytest.fixture(scope="module")
def baseline() -> dict:
    if not BASELINE.exists():
        pytest.skip(f"no baseline at {BASELINE}; run with --update to create it")
    return json.loads(BASELINE.read_text())


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return _snapshot()


@pytest.mark.parametrize(("dsl", "device"), FROZEN_VIEWS)
def test_non_mlir_view_context_unchanged(dsl, device, snapshot, baseline):
    """A non-MLIR view's per-stage prompt context must be byte-identical."""
    key = f"{dsl}/{device}"
    want = baseline[key]["stage_context_sha256"]
    got = snapshot[key]["stage_context_sha256"]
    drifted = sorted(s for s in want if want[s] != got.get(s))
    assert not drifted, f"{key} stage context changed for: {drifted}"


@pytest.mark.parametrize(("dsl", "device"), FROZEN_VIEWS)
def test_non_mlir_view_counts_unchanged(dsl, device, snapshot, baseline):
    key = f"{dsl}/{device}"
    for field in ("patterns", "constraints", "examples", "stage_pattern_counts"):
        assert snapshot[key][field] == baseline[key][field], f"{key}.{field} changed"


def test_every_view_loads_something(snapshot):
    for key, v in snapshot.items():
        assert v["patterns"] or v["constraints"], f"{key} loaded an empty KB"


def test_mlir_views_are_not_empty(snapshot):
    """The MLIR KB may be reorganized, but it must never silently vanish."""
    for key in ("mlir/xpu", "mlir/linalg"):
        assert snapshot[key]["constraints"] > 0, f"{key} has no constraints"


if __name__ == "__main__":
    import sys

    if "--update" in sys.argv:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(_snapshot(), indent=1, sort_keys=True) + "\n")
        print(f"wrote {BASELINE}")
    else:
        print(json.dumps(_snapshot(), indent=1, sort_keys=True))
