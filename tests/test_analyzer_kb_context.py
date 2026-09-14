"""Tests for the constraint list the analyzer is shown.

Why it exists: the analyzer only ever sees KB *constraints*, never patterns. A
pattern can fire only if the analyzer first flags an issue that routes to the
pattern's stage. So a constraint that is silently cut from this list makes its
pattern unreachable.

That happened. The list was built in `OptimizationStage` enum order and cut at 6000
chars, which dropped 10 of 24 constraints on mlir/xpu. `algorithmic` sorts early and
kept 7; `device_specific` sorts late and kept 2 of its 10. One of the 8 it lost was
the trigger for `fastmath<fast>` — the biggest measured flash-attention lever at
1.834x. The analyzer never saw the trigger, so it only ever reported
`suboptimal_tile_size`, and the lever was never tried.
"""

import logging

from xe_forge.agents.analyzer_agent import AnalyzerAgent
from xe_forge.knowledge.loader import load_knowledge_base
from xe_forge.models import OptimizationStage


def _agent(dsl="mlir", device="xpu"):
    agent = AnalyzerAgent.__new__(AnalyzerAgent)  # no LLM, no dspy setup
    agent.knowledge_base = load_knowledge_base("knowledge_base", dsl=dsl, device_type=device)
    return agent


def _buckets(kb):
    """Per-stage constraint buckets, deduped by id exactly as `_get_kb_context` does.

    A constraint can declare several stages. The first stage in enum order claims it,
    so the buckets are disjoint and match what the round-robin actually walks.
    """
    out = []
    seen = set()
    for stage in OptimizationStage:
        if stage == OptimizationStage.ANALYSIS:
            continue
        bucket = []
        for c in kb.constraints_for_stage(stage):
            if c.id in seen or c.severity not in ("critical", "warning"):
                continue
            seen.add(c.id)
            bucket.append(c)
        if bucket:
            out.append(bucket)
    return out


def _eligible_ids(kb):
    return [c.id for bucket in _buckets(kb) for c in bucket]


def test_the_whole_mlir_constraint_set_fits():
    # 24 constraints, ~9.8 KB. The budget must cover the real KB, or the fix only
    # holds until someone adds a pattern.
    agent = _agent()
    ctx = agent._get_kb_context()
    shown = ctx.count("[CRITICAL]") + ctx.count("[WARNING]")
    assert shown == len(_eligible_ids(agent.knowledge_base))
    assert "not shown]" not in ctx


def test_the_fastmath_trigger_reaches_the_analyzer():
    # The specific regression: this is the constraint that makes
    # `xegpu_softmax_reduction_fastmath` reachable.
    ctx = _agent()._get_kb_context()
    assert "fastmath" in ctx.lower()


def test_over_budget_thins_every_stage_instead_of_starving_late_ones():
    # Round-robin, not enum order. With a budget that fits only a few lines, each
    # stage present must still contribute one before any stage contributes two.
    agent = _agent()
    buckets = _buckets(agent.knowledge_base)
    assert len(buckets) >= 4, "test needs several stages to be meaningful"

    agent._KB_CONTEXT_BUDGET = 2600  # room for ~6 lines of ~400 chars
    ctx = agent._get_kb_context()
    shown = ctx.count("[CRITICAL]") + ctx.count("[WARNING]")
    assert 0 < shown < len(_eligible_ids(agent.knowledge_base))

    # Every stage that could fit contributed its first constraint before any stage
    # contributed a second. Enum order would have given all of these to `algorithmic`.
    for bucket in buckets[:shown]:
        assert bucket[0].name in ctx, f"stage starved: {bucket[0].id} missing at budget 2600"


def test_over_budget_says_so(caplog):
    # The only old tell was a log line reading exactly "6015 chars".
    agent = _agent()
    agent._KB_CONTEXT_BUDGET = 2600
    with caplog.at_level(logging.WARNING):
        ctx = agent._get_kb_context()
    assert "constraints dropped" in caplog.text
    assert "more constraints not shown]" in ctx


def test_no_knowledge_base_is_empty_not_an_error():
    agent = AnalyzerAgent.__new__(AnalyzerAgent)
    agent.knowledge_base = None
    assert agent._get_kb_context() == ""
