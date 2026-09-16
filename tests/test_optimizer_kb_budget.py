"""Tests for the KB context the optimizer is shown.

Why it exists: the optimizer used to build this context by rendering the whole
stage with `format_for_stage()` and cutting the string at 14000 chars. That
string renders EVERY constraint before the FIRST pattern. On mlir/xpu the
algorithmic constraints alone came to 24802 chars, so the cut landed inside them
and the optimizer received zero patterns for that stage. It had been that way
for every stage and every run.

The cost was measured. The `xegpu_row_per_subgroup_single_read_softmax` pattern
sat at char 30762 of 43124, along with its 1.982x figure and its "zero spills at
128 GRF" rebuttal. Two 4096x4096 softmax runs (2026-09-10 and 2026-09-16) both
detected the redundant second pass, ranked it top severity, and then declined to
fix it as too risky. The evidence that it was not risky was in the part of the
prompt that got cut. Both runs finished near 1.17x; the hand-written one-pass
kernel measures 2.32x.

Two properties are load-bearing here:
  1. Entries are selected WHOLE. A character cut on a rendered entry can leave a
     trigger or a fix line dangling mid-sentence.
  2. Constraints and patterns are offered to the budget ALTERNATELY, so verbose
     constraints cannot starve every pattern for the stage.
"""

from xe_forge.agents.optimizer_agent import OptimizerAgent
from xe_forge.knowledge.loader import load_knowledge_base
from xe_forge.models import OptimizationStage

# Stages that actually carry patterns on mlir/xpu.
_PATTERN_STAGES = [
    OptimizationStage.ALGORITHMIC,
    OptimizationStage.MEMORY_ACCESS,
    OptimizationStage.DEVICE_SPECIFIC,
    OptimizationStage.DTYPE_FIX,
]


def _agent(dsl="mlir", device="xpu"):
    agent = OptimizerAgent.__new__(OptimizerAgent)  # no LLM, no dspy setup
    agent.knowledge_base = load_knowledge_base("knowledge_base", dsl=dsl, device_type=device)
    return agent


def test_every_stage_with_patterns_gets_at_least_one_pattern():
    """The starvation bug. Constraints must not consume the whole budget."""
    agent = _agent()
    for stage in _PATTERN_STAGES:
        available = len(agent.knowledge_base.get_by_stage(stage))
        assert available, f"{stage.value} has no patterns to test with"
        agent._get_stage_patterns(stage)
        _, _, kept_patterns, _ = agent._last_kb_counts
        assert kept_patterns > 0, (
            f"{stage.value}: 0 of {available} patterns reached the optimizer. "
            "Constraints ate the budget — check the alternating selection."
        )


def test_context_stays_within_budget():
    for dsl in ("mlir", "triton"):
        agent = _agent(dsl=dsl)
        for stage in OptimizationStage:
            context = agent._get_stage_patterns(stage)
            assert len(context) <= agent.KB_CONTEXT_BUDGET, (
                f"{dsl}/{stage.value}: {len(context)} chars over {agent.KB_CONTEXT_BUDGET} budget"
            )


def test_entries_are_never_cut_mid_entry():
    """No selected entry may be a prefix of itself.

    Every rendered constraint and pattern that appears at all must appear whole.
    This is what stops a trigger line or a fix line from being severed.
    """
    agent = _agent()
    kb = agent.knowledge_base
    for stage in _PATTERN_STAGES:
        context = agent._get_stage_patterns(stage)
        for constraint in kb.constraints_for_stage(stage):
            rendered = kb.render_constraint(constraint)
            if f"### {constraint.name}" in context:
                assert rendered in context, f"{constraint.id} was cut mid-entry"
        for pattern in kb.get_by_stage(stage):
            rendered = kb.render_pattern(pattern)
            if f"## {pattern.name}" in context:
                assert rendered in context, f"{pattern.id} was cut mid-entry"


def test_softmax_single_read_fix_reaches_the_optimizer():
    """The regression this file was written for.

    The optimizer cannot be asked to make the one-pass softmax edit if the
    pattern describing it, its measured speedup, and the rebuttal to the
    register-pressure objection are all absent from its prompt.
    """
    agent = _agent()
    context = agent._get_stage_patterns(OptimizationStage.ALGORITHMIC)
    for needle in (
        "Give ONE ROW to the whole WORKGROUP",  # the fix pattern
        "Two-loop online softmax reads the input twice",  # the trigger constraint
        "1.982x",  # what the fix is worth
        "no spills at 128 GRF",  # why it is not the register-pressure risk it looks like
    ):
        assert needle in context, f"missing from algorithmic KB context: {needle!r}"


def test_critical_constraints_are_offered_before_warnings():
    """A dropped critical constraint can break the kernel; a dropped warning costs perf."""
    agent = _agent()
    kb = agent.knowledge_base
    stage = OptimizationStage.ALGORITHMIC
    context = agent._get_stage_patterns(stage)
    constraints = kb.constraints_for_stage(stage)
    critical = [c for c in constraints if str(c.severity).lower() == "critical"]
    assert critical, "no critical constraints on algorithmic to test with"
    for constraint in critical:
        assert f"### {constraint.name}" in context, (
            f"critical constraint {constraint.id} was dropped while "
            "lower-severity entries may have been kept"
        )


def test_no_knowledge_base_returns_empty():
    agent = OptimizerAgent.__new__(OptimizerAgent)
    agent.knowledge_base = None
    assert agent._get_stage_patterns(OptimizationStage.ALGORITHMIC) == ""
