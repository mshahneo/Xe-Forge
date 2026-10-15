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
        "reads it twice; one row per workgroup, held in registers, reads it once",  # the trigger constraint
        "1.982x",  # what the fix is worth
        "no spills at 128 GRF",  # why it is not the register-pressure risk it looks like
    ):
        assert needle in context, f"missing from algorithmic KB context: {needle!r}"


def test_grf_thread_ceiling_reaches_every_stage_that_edits_the_geometry():
    """The GRF mode sets the thread ceiling, so both must be in the same prompt.

    On Xe2 the default GRF allows 64 subgroups (1024 work-items) and large GRF
    only 32 (512). A stage told it may rewrite `threads in (...)` but not told
    the ceiling cannot pick a number. Layer-norm 2026-09-17 is the case: its
    route was algorithmic + memory_access, the geometry rule was scoped to
    device_specific + autotuning, and all four shapes kept `threads = 128` — an
    eighth of the available budget.
    """
    agent = _agent()
    for stage in (
        OptimizationStage.ALGORITHMIC,
        OptimizationStage.MEMORY_ACCESS,
        OptimizationStage.DEVICE_SPECIFIC,
        OptimizationStage.AUTOTUNING,
    ):
        context = agent._get_stage_patterns(stage)
        for needle in (
            "THE GRF MODE SETS THE CEILING",  # the rule itself
            "| default (128) | 64 | 1024 |",  # the numbers
            "| large (256)   | 32 |  512 |",
            "RE-DECIDE IT",  # it is a procedure, not just a warning
        ):
            assert needle in context, f"{stage.value}: missing from KB context: {needle!r}"


def test_both_halves_of_the_one_pass_rewrite_fit_together():
    """The geometry rule and the register-residency rule must not evict each other.

    Adding the GRF/thread ceiling to the algorithmic stage at the old 28000-char
    budget pushed out
    xegpu_online_softmax_still_reads_twice_unless_the_row_is_register_resident,
    which is the entry that says the row has to stay in registers for the
    rewrite to pay. Two halves of one fix, and only one fit.
    """
    agent = _agent()
    context = agent._get_stage_patterns(OptimizationStage.ALGORITHMIC)
    _, total_constraints, kept_patterns, total_patterns = agent._last_kb_counts
    assert "THE GRF MODE SETS THE CEILING" in context
    assert "reads it twice; one row per workgroup, held in registers, reads it once" in context
    assert kept_patterns == total_patterns, (
        f"algorithmic dropped {total_patterns - kept_patterns} of {total_patterns} "
        "patterns; the stage that owns the launch-geometry rewrite should see all of them"
    )


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


# Over budget, the entries tied to the routed issue go first. The cut used to keep
# severity and file order only, and dropped the very trigger the analyzer had routed
# to the stage: xegpu_dpas_feed_load_uses_dpas_inst_data (memory_access) and
# xegpu_tile_rows_walked_in_groups (device_specific) on the 2026-10-15 SwiGLU runs.
def _names(agent, stage, ids):
    kb = agent.knowledge_base
    entries = list(kb.constraints_for_stage(stage)) + list(kb.get_by_stage(stage))
    return {e.id: e.name for e in entries if e.id in ids}


def test_the_routed_trigger_survives_the_memory_access_cut():
    agent = _agent()
    stage = OptimizationStage.MEMORY_ACCESS
    name = _names(agent, stage, {"xegpu_dpas_feed_load_uses_dpas_inst_data"})[
        "xegpu_dpas_feed_load_uses_dpas_inst_data"
    ]
    assert name not in agent._get_stage_patterns(stage), "test needs the entry to be cut without a hint"
    assert name in agent._get_stage_patterns(stage, issue_types=["uncoalesced_access"])


def test_a_pattern_tied_to_the_trigger_is_kept_under_a_tight_budget():
    # The pattern names its trigger constraint by id; that tie is what keeps it.
    agent = _agent()
    agent.KB_CONTEXT_BUDGET = 9000
    stage = OptimizationStage.DEVICE_SPECIFIC
    names = _names(agent, stage, {"xegpu_tile_rows_walked_in_groups", "xegpu_grouped_tile_launch_order"})
    plain = agent._get_stage_patterns(stage)
    tied = agent._get_stage_patterns(stage, issue_types=["no_swizzling"])
    assert not all(n in plain for n in names.values()), "budget too loose for the test to mean anything"
    for n in names.values():
        assert n in tied, n
    assert len(tied) <= agent.KB_CONTEXT_BUDGET


def test_no_issue_types_keeps_the_old_order():
    agent = _agent()
    for stage in _PATTERN_STAGES:
        assert agent._get_stage_patterns(stage) == agent._get_stage_patterns(stage, issue_types=[])
