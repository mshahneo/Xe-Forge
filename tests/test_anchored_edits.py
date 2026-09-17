"""Tests for the anchored-edit path on large MLIR modules.

Why it exists: a 22 KB MLIR module does not survive being re-emitted by an LLM. A
one-attribute change comes back with unrelated syntax corrupted somewhere else, the
attempt is rejected, and the stage burns an iteration. That is how the biggest
measured flash-attention lever (``fastmath<fast>``, 1.834x) went unapplied for a
whole run. So small changes travel as anchors instead: exact text copied from the
module, plus its replacement.

Every failure mode below returns a reason string instead of a patched module. The
reason is written to be read by the LLM and retried against, so these tests also pin
that an ambiguous anchor never becomes a silent edit in the wrong place.

Two later rules are pinned here too, both from the 4096x4096 softmax run of
2026-09-16:

* **One bad anchor skips only its own edit.** The batch used to fail whole, so a
  single stale anchor threw away every good edit beside it.
* **An ambiguous anchor has a way out: ``occurrence`` and ``all``.** Without them
  the guard blocked exactly the fixes that remove duplicated code, because the
  duplication is what makes the anchor non-unique. That is how the 2.3x one-pass
  softmax rewrite was lost: the model tried to retype the input descriptor, the
  line existed twice (once per input pass), and the edit was thrown away.
"""

import json
import logging
from pathlib import Path

import dspy

from xe_forge.agents.optimizer_agent import (
    EDITS_SENTINEL,
    SUCCESS_MESSAGE,
    OptimizerAgent,
    _apply_anchored_edits,
)

MODULE = """\
gpu.func @payload_kernel() kernel {
  %0 = math.exp %a : vector<128xf32>
  %1 = arith.mulf %b, %c : vector<128xf32>
  %2 = math.exp %d : vector<128xf32>
  gpu.return
}
"""


def _edits(*pairs):
    return json.dumps([{"old": o, "new": n} for o, n in pairs])


def test_single_edit_applies():
    new, why = _apply_anchored_edits(
        MODULE, _edits(("%0 = math.exp %a :", "%0 = math.exp %a fastmath<fast> :"))
    )
    assert why == ""
    assert "%0 = math.exp %a fastmath<fast> :" in new
    # The second exp is untouched: only the anchored site changes.
    assert "%2 = math.exp %d : vector<128xf32>" in new


def test_multiple_edits_apply_in_order():
    new, why = _apply_anchored_edits(
        MODULE,
        _edits(
            ("%0 = math.exp %a :", "%0 = math.exp %a fastmath<fast> :"),
            ("%2 = math.exp %d :", "%2 = math.exp %d fastmath<fast> :"),
        ),
    )
    assert why == ""
    assert new.count("fastmath<fast>") == 2


def test_a_dict_is_accepted_as_one_edit():
    new, why = _apply_anchored_edits(
        MODULE, json.dumps({"old": "arith.mulf %b, %c :", "new": "arith.mulf %b, %c fast :"})
    )
    assert why == ""
    assert "arith.mulf %b, %c fast :" in new


def test_fenced_json_is_unwrapped():
    body = _edits(("gpu.return", "gpu.return // done"))
    new, why = _apply_anchored_edits(MODULE, f"```json\n{body}\n```")
    assert why == ""
    assert "gpu.return // done" in new


def test_missing_anchor_is_rejected():
    new, why = _apply_anchored_edits(MODULE, _edits(("math.exp %zzz :", "whatever")))
    assert new is None
    assert "ANCHOR NOT FOUND" in why


def test_ambiguous_anchor_is_rejected():
    # "vector<128xf32>" appears 3 times. Applying to the first would be a silent
    # edit in a place the LLM did not choose.
    new, why = _apply_anchored_edits(MODULE, _edits(("vector<128xf32>", "vector<128xf16>")))
    assert new is None
    assert "ANCHOR AMBIGUOUS (3 matches)" in why


def test_the_ambiguous_message_offers_both_ways_out():
    # The model cannot use `occurrence` or `all` unless the rejection names them.
    _new, why = _apply_anchored_edits(MODULE, _edits(("vector<128xf32>", "vector<128xf16>")))
    assert '"occurrence": 1..3' in why
    assert '"all": true' in why


# --- one bad anchor must not kill the batch (option A) ------------------------


def test_one_bad_anchor_skips_only_itself():
    new, why = _apply_anchored_edits(
        MODULE,
        _edits(
            ("%0 = math.exp %a :", "%0 = math.exp %a fastmath<fast> :"),
            ("math.exp %zzz :", "whatever"),
            ("%2 = math.exp %d :", "%2 = math.exp %d fastmath<fast> :"),
        ),
    )
    # The two good edits land. Previously all three were discarded.
    assert new is not None
    assert new.count("fastmath<fast>") == 2
    # And the skip is reported, not swallowed.
    assert "2 edit(s) applied, 1 skipped" in why
    assert "ANCHOR NOT FOUND" in why


def test_a_batch_of_only_bad_anchors_still_fails():
    new, why = _apply_anchored_edits(
        MODULE, _edits(("math.exp %zzz :", "a"), ("math.exp %qqq :", "b"))
    )
    assert new is None
    # Both reasons come back, so the model can fix both at once.
    assert why.count("ANCHOR NOT FOUND") == 2


def test_an_ambiguous_anchor_beside_a_good_one_keeps_the_good_one():
    new, why = _apply_anchored_edits(
        MODULE,
        _edits(
            ("vector<128xf32>", "vector<128xf16>"),  # 3 matches, no occurrence/all
            ("gpu.return", "gpu.return // done"),
        ),
    )
    assert new is not None
    assert "gpu.return // done" in new
    # The ambiguous one changed nothing.
    assert new.count("vector<128xf32>") == 3
    assert "ANCHOR AMBIGUOUS" in why


def test_the_note_is_empty_when_every_edit_applies():
    new, why = _apply_anchored_edits(MODULE, _edits(("gpu.return", "gpu.return // done")))
    assert new is not None
    assert why == ""


# --- occurrence and all (option C) -------------------------------------------


def test_all_true_changes_every_match():
    new, why = _apply_anchored_edits(
        MODULE, json.dumps([{"old": "vector<128xf32>", "new": "vector<128xf16>", "all": True}])
    )
    assert why == ""
    assert new.count("vector<128xf16>") == 3
    assert "vector<128xf32>" not in new


def test_occurrence_picks_exactly_that_match():
    new, why = _apply_anchored_edits(
        MODULE,
        json.dumps([{"old": "vector<128xf32>", "new": "vector<128xf16>", "occurrence": 2}]),
    )
    assert why == ""
    # The middle line changed; the first and third did not.
    assert "%0 = math.exp %a : vector<128xf32>" in new
    assert "%1 = arith.mulf %b, %c : vector<128xf16>" in new
    assert "%2 = math.exp %d : vector<128xf32>" in new


def test_occurrence_one_is_the_first_match():
    new, _why = _apply_anchored_edits(
        MODULE,
        json.dumps([{"old": "vector<128xf32>", "new": "vector<128xf16>", "occurrence": 1}]),
    )
    assert "%0 = math.exp %a : vector<128xf16>" in new
    assert new.count("vector<128xf32>") == 2


def test_occurrence_past_the_end_is_rejected():
    new, why = _apply_anchored_edits(
        MODULE,
        json.dumps([{"old": "vector<128xf32>", "new": "vector<128xf16>", "occurrence": 4}]),
    )
    assert new is None
    assert "matches 3 time(s)" in why


def test_occurrence_must_be_a_positive_int():
    for bad in (0, -1, 1.5, "2", True):
        new, why = _apply_anchored_edits(
            MODULE, json.dumps([{"old": "vector<128xf32>", "new": "x", "occurrence": bad}])
        )
        assert new is None, f"{bad!r} was accepted"
        assert "must be an integer >= 1" in why


def test_all_and_occurrence_together_is_rejected():
    new, why = _apply_anchored_edits(
        MODULE,
        json.dumps([{"old": "vector<128xf32>", "new": "x", "all": True, "occurrence": 1}]),
    )
    assert new is None
    assert "not both" in why


def test_all_on_a_unique_anchor_behaves_like_a_plain_edit():
    new, why = _apply_anchored_edits(
        MODULE, json.dumps([{"old": "gpu.return", "new": "gpu.return // done", "all": True}])
    )
    assert why == ""
    assert new.count("gpu.return // done") == 1


# --- the regression this was all for -----------------------------------------

# A two-pass softmax in miniature: the input descriptor is built once per pass, so
# the two lines are byte-identical. Retyping it is the core of the one-pass rewrite.
TWO_PASS = """\
gpu.func @payload_kernel(%arg0: memref<4096x4096xf32>) kernel {
  scf.for %i = %c0 to %c256 step %c1 {
    %2 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf32> -> !xegpu.tensor_desc<64x16xf32>
    %3 = xegpu.load_nd %2[%i, %c0] : !xegpu.tensor_desc<64x16xf32> -> vector<64x16xf32>
  }
  scf.for %j = %c0 to %c256 step %c1 {
    %2 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf32> -> !xegpu.tensor_desc<64x16xf32>
    %3 = xegpu.load_nd %2[%j, %c0] : !xegpu.tensor_desc<64x16xf32> -> vector<64x16xf32>
  }
  gpu.return
}
"""

DESC_LINE = (
    "    %2 = xegpu.create_nd_tdesc %arg0 : memref<4096x4096xf32> "
    "-> !xegpu.tensor_desc<64x16xf32>\n"
)


def test_the_duplicated_descriptor_retype_used_to_be_impossible():
    # Guards against a vacuous test below: without `all`/`occurrence` this is the
    # exact rejection that lost the 2.3x rewrite on the 4K run.
    assert TWO_PASS.count(DESC_LINE) == 2
    new, why = _apply_anchored_edits(
        TWO_PASS, _edits((DESC_LINE, DESC_LINE.replace("64x16", "1x4096")))
    )
    assert new is None
    assert "ANCHOR AMBIGUOUS (2 matches)" in why


def test_the_duplicated_descriptor_retype_transports_with_all():
    new, why = _apply_anchored_edits(
        TWO_PASS,
        json.dumps([{"old": DESC_LINE, "new": DESC_LINE.replace("64x16", "1x4096"), "all": True}]),
    )
    assert why == ""
    assert new.count("!xegpu.tensor_desc<1x4096xf32>") == 2
    assert "!xegpu.tensor_desc<64x16xf32>" in new  # the load types are untouched


def test_only_the_second_pass_can_be_targeted():
    # Deleting the second read is the other shape the fix takes.
    new, why = _apply_anchored_edits(
        TWO_PASS, json.dumps([{"old": DESC_LINE, "new": "", "occurrence": 2}])
    )
    assert why == ""
    assert new.count(DESC_LINE) == 1
    # The surviving one is the first pass, not the second.
    assert new.index("scf.for %i") < new.index(DESC_LINE.strip())
    assert DESC_LINE.strip() not in new.split("scf.for %j")[1]


def test_invalid_json_is_rejected_with_a_position():
    new, why = _apply_anchored_edits(MODULE, '[{"old": "gpu.return", "new":}]')
    assert new is None
    assert "EDITS NOT VALID JSON" in why


def test_missing_key_is_rejected():
    new, why = _apply_anchored_edits(MODULE, json.dumps([{"old": "gpu.return"}]))
    assert new is None
    assert "needs both" in why


def test_no_op_edit_is_rejected():
    new, why = _apply_anchored_edits(MODULE, _edits(("gpu.return", "gpu.return")))
    assert new is None
    assert "no-op" in why


def test_empty_and_none_mean_no_edits_supplied():
    # This exact reason is what makes `compile_and_verify` fall back to reading
    # `optimized_code`, so the whole-module path still works.
    for raw in ("", "   ", "NONE", "none"):
        new, why = _apply_anchored_edits(MODULE, raw)
        assert new is None
        assert why == "no edits supplied"


def test_empty_list_is_rejected():
    new, why = _apply_anchored_edits(MODULE, "[]")
    assert new is None
    assert "non-empty JSON list" in why


def test_sentinel_is_a_plain_word():
    # The verify tool compares `optimized_code.strip().upper()` against it, so it
    # must not carry punctuation or whitespace.
    assert EDITS_SENTINEL == EDITS_SENTINEL.strip().upper()
    assert EDITS_SENTINEL.isalpha()


# --- the verify tool end of the path (no executor, no GPU) -------------------

# Minimal module that clears _verify_mlir's structural pre-checks.
WG_MODULE = """\
gpu.module @k {
  gpu.func @payload_kernel() kernel {
    %0 = math.exp %a : vector<128xf32>
    gpu.return
  }
}
func.func @main() {
  gpu.launch_func @k::@payload_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)
  call @printAllclose() : () -> ()
  return
}
"""


def _mlir_tool(tmp_path=None):
    from xe_forge.models import DSL, OptimizationStage

    agent = OptimizerAgent(dsl=DSL.MLIR)
    if tmp_path is not None:
        agent.attempts_dir = tmp_path
    tool, last_accepted = agent._create_verify_tool(
        WG_MODULE, "k", None, None, stage=OptimizationStage.DEVICE_SPECIFIC
    )
    return tool, last_accepted


def test_verify_tool_applies_edits_and_hands_the_module_back():
    tool, last_accepted = _mlir_tool()
    out = tool.func(
        optimized_code=EDITS_SENTINEL,
        edits=_edits(("math.exp %a :", "math.exp %a fastmath<fast> :")),
    )
    assert out == SUCCESS_MESSAGE
    # The stage cannot read the module out of `optimized_code` on this path, so the
    # tool must leave it in `last_accepted["code"]`.
    assert "fastmath<fast>" in last_accepted["code"]


def test_verify_tool_reports_a_bad_anchor_instead_of_falling_back():
    # Falling back to `optimized_code` here would silently verify the unchanged
    # module and bank a 1.00x "win".
    tool, last_accepted = _mlir_tool()
    out = tool.func(optimized_code=WG_MODULE, edits=_edits(("math.exp %nope :", "x")))
    assert out.startswith("EDITS REJECTED")
    assert "ANCHOR NOT FOUND" in out
    assert last_accepted["code"] is None


def test_verify_tool_rejects_the_sentinel_with_no_edits():
    tool, _ = _mlir_tool()
    out = tool.func(optimized_code=EDITS_SENTINEL, edits="")
    assert EDITS_SENTINEL in out and "supplied no" in out


def test_verify_tool_still_takes_a_whole_module():
    tool, last_accepted = _mlir_tool()
    whole = WG_MODULE.replace("math.exp %a :", "math.exp %a fastmath<fast> :")
    assert tool.func(optimized_code=whole, edits="NONE") == SUCCESS_MESSAGE
    assert "fastmath<fast>" in last_accepted["code"]


def test_anchors_resolve_against_the_advancing_base():
    # After a stage banks an improvement the LLM is shown the improved module, so
    # anchors must resolve against that — not the stage input.
    tool, last_accepted = _mlir_tool()
    improved = WG_MODULE.replace("math.exp %a :", "math.exp %a fastmath<fast> :")
    last_accepted["base"] = improved
    out = tool.func(
        optimized_code=EDITS_SENTINEL,
        edits=_edits(("math.exp %a fastmath<fast> :", "math.exp2 %a fastmath<fast> :")),
    )
    assert out == SUCCESS_MESSAGE
    # Both edits are present: the second stacked on the first instead of reverting it.
    assert "math.exp2 %a fastmath<fast> :" in last_accepted["code"]


def test_rejected_attempts_are_written_to_disk(tmp_path):
    # A stage that fails 5/5 used to report only "no valid result in budget".
    tool, _ = _mlir_tool(tmp_path)
    tool.func(optimized_code="not mlir at all", edits="NONE")
    rejected = list(tmp_path.glob("*_rejected.mlir"))
    assert len(rejected) == 1
    assert rejected[0].read_text() == "not mlir at all"
    verdict = rejected[0].with_suffix("").with_suffix(".verdict.txt")
    assert "MISSING" in verdict.read_text()


def test_triton_path_ignores_edits():
    from xe_forge.models import DSL

    agent = OptimizerAgent(dsl=DSL.TRITON)
    tool, _ = agent._create_verify_tool("x = 1", "k", None, None)
    # Valid Python with edits set: the edits must not be consulted at all.
    out = tool.func(optimized_code="def f():\n    return 1\n", edits=_edits(("x", "y")))
    assert "ANCHOR" not in out and "EDITS REJECTED" not in out


# --- the stage must spend its whole budget ------------------------------------


def _stage_run(monkeypatch, speedups, max_iterations=5):
    """Drive `optimize_stage` with canned per-run speedups; return runs made.

    CoVeR and `_final_verify` are both stubbed, so no LLM and no GPU. Each entry
    in `speedups` is what that run's candidate measures.
    """
    import xe_forge.agents.optimizer_agent as oa
    from xe_forge.models import DSL, DetectedIssue, IssueType, KernelAnalysis, OptimizationStage

    runs = []

    class FakeCoVeR:
        def __init__(self, **kw):
            self.max_iters = kw.get("max_iters", 1)

        def __call__(self, **kwargs):
            i = len(runs)
            runs.append(kwargs)
            # A distinct module per run, so the "identical code" guard never fires.
            return dspy.Prediction(
                trajectory={"thought_0": "t"},
                optimized_code=WG_MODULE.replace("gpu.return", f"gpu.return // v{i}"),
                edits="NONE",
            )

    monkeypatch.setattr(oa, "CoVeR", FakeCoVeR)

    agent = oa.OptimizerAgent(dsl=DSL.MLIR, max_iterations=max_iterations)
    monkeypatch.setattr(
        agent,
        "_final_verify",
        lambda *a, **k: (True, speedups[len(runs) - 1], {"time_ms": 1.0}, {"time_ms": 1.0}, None),
    )

    analysis = KernelAnalysis(
        kernel_name="k",
        detected_issues=[
            DetectedIssue(
                issue_type=IssueType.SUBOPTIMAL_TILE_SIZE,
                severity=5,
                description="tiles",
                suggested_fix="use bigger tiles",
                location="kernel",
            )
        ],
    )
    res = agent.optimize_stage(
        code=WG_MODULE,
        stage=OptimizationStage.DEVICE_SPECIFIC,
        analysis=analysis,
        xpu_config={},
        kernel_name="k",
    )
    return res, runs


def test_a_missed_run_does_not_end_the_stage(monkeypatch):
    # The real failure: run 1 measured 0.99x on the knob the issue text named, the
    # stage stopped, and the 1.83x float flags were never tried.
    res, runs = _stage_run(monkeypatch, [0.99, 0.99, 1.50, 1.0, 1.0])
    assert len(runs) == 5, f"stage quit after {len(runs)} of 5 runs"
    assert res.success and res.speedup == 1.50


def test_a_missed_run_is_reported_back_to_the_next_run(monkeypatch):
    _res, runs = _stage_run(monkeypatch, [0.99, 0.99, 0.99, 0.99, 0.99])
    assert len(runs) == 5
    # Run 2 onwards must see what run 1 already measured.
    assert "Previous attempts this stage" in runs[1]["issues"]
    assert "0.990x" in runs[1]["issues"]


def test_a_missed_run_does_not_become_the_next_base(monkeypatch):
    # A slower candidate must not be what the next run edits.
    _res, runs = _stage_run(monkeypatch, [0.99, 0.99, 0.99, 0.99, 0.99])
    assert all(r["current_code"] == WG_MODULE for r in runs)


def test_a_win_does_become_the_next_base(monkeypatch):
    _res, runs = _stage_run(monkeypatch, [1.50, 0.99, 0.99, 0.99, 0.99])
    assert runs[1]["current_code"] != WG_MODULE
    assert "// v0" in runs[1]["current_code"]


# --- a failure must not read the same as four other failures ------------------


def _log_one_attempt(caplog, verdict):
    """Return the `attempt N` log line `_attempt_log` writes for `verdict`."""
    from xe_forge.models import DSL, OptimizationStage

    agent = OptimizerAgent(dsl=DSL.MLIR)
    with caplog.at_level(logging.INFO, logger="xe_forge.agents.optimizer_agent"):
        agent._attempt_log(OptimizationStage.DEVICE_SPECIFIC, 4, "edits", verdict, None)
    lines = [r.getMessage() for r in caplog.records if "attempt 4" in r.getMessage()]
    assert lines, [r.getMessage() for r in caplog.records]
    return lines[0]


def test_the_log_line_carries_the_compiler_error(caplog):
    # Five different lowering failures all logged the same headline, so the log said
    # nothing about why. The reason is always on a later line.
    line = _log_one_attempt(
        caplog,
        "FAILURE: Optimized kernel failed: LOWERING FAILED (imex-opt):\n"
        "/tmp/x.mlir:107:12: error: expected '=' after SSA name\n",
    )
    assert "error: expected '=' after SSA name" in line


def test_the_error_line_beats_an_earlier_note_line(caplog):
    # imex-opt prints `note:` lines too, sometimes before the error. The error is
    # what identifies the failure.
    line = _log_one_attempt(
        caplog,
        "FAILURE: LOWERING FAILED (imex-opt):\n"
        "/tmp/x.mlir:92:31: note: prior use here\n"
        "/tmp/x.mlir:92:31: error: use of value '%cst' expects different type\n",
    )
    assert "error: use of value" in line
    assert "note: prior use" not in line


def test_a_one_line_verdict_is_unchanged(caplog):
    # A success verdict has no second line and must not grow noise.
    line = _log_one_attempt(caplog, SUCCESS_MESSAGE)
    assert line.endswith(SUCCESS_MESSAGE)


# --- a digit-led SSA name is repaired, not rejected -----------------------------


def test_a_digit_led_name_is_renamed():
    from xe_forge.agents.optimizer_agent import _repair_digit_led_ssa_names

    code, note = _repair_digit_led_ssa_names(
        "%35_f32 = arith.extf %35 : vector<4xf16> to vector<4xf32>\n"
        "%36 = arith.mulf %35_f32, %35_f32 : vector<4xf32>\n"
    )
    # Every occurrence moves together, and the legal `%35` / `%36` are untouched.
    assert "%v35_f32 = arith.extf %35 :" in code
    assert "arith.mulf %v35_f32, %v35_f32" in code
    assert "%35_f32" not in code
    assert "renamed 1" in note


def test_legal_names_are_left_alone():
    from xe_forge.agents.optimizer_agent import _repair_digit_led_ssa_names

    src = "%0 = arith.addf %arg5, %cst_1 : f32\n%18:3 = scf.for %c0 = %c0 to %c4 step %c1\n"
    code, note = _repair_digit_led_ssa_names(src)
    assert code == src
    assert note == ""


def test_a_repair_never_collides_with_an_existing_name():
    from xe_forge.agents.optimizer_agent import _repair_digit_led_ssa_names

    # `%v35_f32` is already taken, so the repair must pick something else.
    code, _ = _repair_digit_led_ssa_names("%v35_f32 = f()\n%35_f32 = g()\n")
    assert "%v35_f32 = f()" in code
    assert "%v35_f32 = g()" not in code
    assert "%35_f32" not in code.replace("%v35_f32", "")


def test_the_real_rejected_module_loses_its_name_errors(tmp_path):
    """Repair clears every name error in the f32 retype from run 5.

    It is NOT enough to make that module parse, and the test says so: underneath the
    names sits a real error (the f16 `%cst` scale feeding an f32 multiply). The claim
    guarded here is only that repair removes the whole name class and surfaces the
    genuine problem instead of hiding behind it. Skips without a local mlir-opt.
    """
    import shutil
    import subprocess

    from xe_forge.agents.optimizer_agent import _repair_digit_led_ssa_names

    src = Path(__file__).parent / "data" / "f32_retype_digit_led_names.mlir"
    opt = shutil.which("mlir-opt") or "/home/gta/upstream/llvm-project/build-with-imex/bin/mlir-opt"
    if not src.exists() or not Path(opt).exists():
        import pytest

        pytest.skip("needs the sample module and a local mlir-opt")

    raw = src.read_text()
    before = subprocess.run([opt, str(src), "-o", "/dev/null"], capture_output=True, text=True)
    assert "expected '=' after SSA name" in before.stderr, "sample no longer reproduces"

    fixed, note = _repair_digit_led_ssa_names(raw)
    assert note
    p = tmp_path / "fixed.mlir"
    p.write_text(fixed)
    after = subprocess.run([opt, str(p), "-o", "/dev/null"], capture_output=True, text=True)

    assert "expected '=' after SSA name" not in after.stderr, after.stderr
    # What is left is the real defect the names were masking: the f16 scale constant
    # now feeds an f32 multiply. A different error class, and a useful one.
    assert "expects different type" in after.stderr
