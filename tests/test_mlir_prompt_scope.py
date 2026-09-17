"""The MLIR prompts must agree on what the model is allowed to edit.

Why this file exists. The one-pass softmax rewrite needs a new launch geometry:
`blocks in` 64 -> 4096 and one row per workgroup. Three separate prompt blocks
had something to say about that, and they did not agree.

* `MlirOptimizationSignature` said BOTH "you may edit the `gpu.launch_func`
  grid/block geometry" AND "DO NOT modify the `@main` harness". The launch sits
  inside `@main`, so the model resolved the contradiction conservatively and
  refused the fix with "outside the permitted edit scope" (4K run, 2026-09-16).
* `react_agent.py` carried the identical two lines.
* `MlirAlgorithmicOptimizationSignature` did not mention the launch geometry at
  all, so its scope list read as forbidding it outright.

Changing the launch geometry is a primary optimization lever, not an edge case.
It re-decomposes the problem, it changes the layout the tile is distributed
with, and it trades register budget against thread count. So these tests pin
that every MLIR prompt permits it and states the rules that come with it.

These assert on prompt text. That is deliberate: the defect WAS the text, and no
behavioural test catches it without a live LLM.
"""

import pytest

from xe_forge.agents.optimizer_agent import (
    MlirAlgorithmicOptimizationSignature,
    MlirOptimizationSignature,
)


def _mlir_prompts():
    """Every prompt that tells an LLM what it may edit in an MLIR module."""
    from xe_forge.agents import react_agent

    prompts = {
        "MlirOptimizationSignature": MlirOptimizationSignature.__doc__,
        "MlirAlgorithmicOptimizationSignature": MlirAlgorithmicOptimizationSignature.__doc__,
    }
    # react_agent's MLIR signature, whatever it is named.
    for name in dir(react_agent):
        obj = getattr(react_agent, name)
        doc = getattr(obj, "__doc__", None)
        if (
            isinstance(doc, str)
            and "gpu.launch_func" in doc
            and "HARD CONSTRAINTS" in doc
            and name.startswith("Mlir")
        ):
            prompts[f"react_agent.{name}"] = doc
    return prompts


ALL_PROMPTS = _mlir_prompts()


def test_the_react_mlir_prompt_was_found():
    # Guards the loop above: if react_agent's signature is renamed, the scope
    # checks below must not silently stop covering it.
    assert any(k.startswith("react_agent.") for k in ALL_PROMPTS), sorted(ALL_PROMPTS)


@pytest.mark.parametrize("name", sorted(ALL_PROMPTS))
def test_the_launch_geometry_is_editable(name):
    doc = ALL_PROMPTS[name]
    assert "gpu.launch_func" in doc, f"{name} never mentions the launch"
    # The scope sentence must name the launch geometry as editable.
    assert "grid/block geometry" in doc or "blocks in" in doc, (
        f"{name} does not say the launch geometry is editable"
    )


@pytest.mark.parametrize("name", sorted(ALL_PROMPTS))
def test_no_blanket_ban_on_editing_main(name):
    """A bare "do not touch @main" contradicts the launch-geometry permission.

    The launch lives inside `@main`. Any ban has to carve it out, or the model
    reads the two rules together and declines the edit.
    """
    doc = ALL_PROMPTS[name]
    banned = [
        "DO NOT touch the `@main` harness, fill values, or CPU reference.",
        "DO NOT modify the `@main` harness",
    ]
    for phrase in banned:
        assert phrase not in doc, f"{name} still carries the blanket ban: {phrase!r}"
    # It must instead say the launch is inside @main and still editable.
    assert "INSIDE `func.func @main`" in doc, f"{name} does not place the launch inside @main"


@pytest.mark.parametrize("name", sorted(ALL_PROMPTS))
def test_what_inside_main_stays_off_limits_is_still_named(name):
    # Permitting the launch edit must not be read as permitting harness edits.
    doc = ALL_PROMPTS[name]
    for oracle in ("ALLCLOSE", "reference"):
        assert oracle in doc, f"{name} stopped protecting the {oracle} oracle"


@pytest.mark.parametrize("name", sorted(ALL_PROMPTS))
def test_the_lockstep_rule_travels_with_the_permission(name):
    """A new launch geometry silently miscompiles unless these agree.

    `known_block_size` / `known_grid_size` on the `gpu.func` must match the
    launch, or the workgroup pass distributes against the wrong shape.
    """
    doc = ALL_PROMPTS[name]
    assert "known_block_size" in doc, f"{name} permits the edit without the lockstep rule"
    assert "known_grid_size" in doc, f"{name} permits the edit without the lockstep rule"


@pytest.mark.parametrize("name", sorted(ALL_PROMPTS))
def test_sg_layout_must_track_the_thread_count(name):
    # Changing threads without changing sg_layout is the other way to break it.
    doc = ALL_PROMPTS[name]
    assert "sg_layout" in doc, f"{name} does not tie sg_layout to the thread count"


def test_the_algorithmic_prompt_gives_the_register_tradeoff():
    """Why the launch edit matters, not just that it is allowed.

    Smaller per-thread slices with more threads is how a register-resident
    rewrite avoids spilling. Without this the model cites "register pressure" as
    a reason to refuse — which it did on the 09-10 run, wrongly: the winning
    kernel spills zero at 128 GRF.
    """
    doc = MlirAlgorithmicOptimizationSignature.__doc__
    assert "spill" in doc
    assert "thread count" in doc
