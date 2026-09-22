"""Local dry run of `run_arm2_sweep_jetson.py` (the Jetson-direct driver) with MOCK backends -- no GPU,
no weights, no network. Complements `test_colab_sweep_mock.py`, which already covers the shared engine
(`colab_sweep.py`); this file covers what the DRIVER adds: stage sequencing, the reference-locked
session set + parity assertion, candidate splitting across separate invocations sharing one local
checkpoint (the multi-venv usage), a REAL process kill mid-sweep followed by resume, the environment
probe / static candidate feasibility rules, and the validation-gate record enforced across
invocations.

What this cannot show (needs the Jetson): that any real model loads, or its outputs.

Usage (from host_software/):  python ml_jetson_vla/deployment/test_run_arm2_sweep_jetson_mock.py
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from ml_jetson_vla.deployment import run_arm2_sweep_jetson as d  # noqa: E402

d._load_engine()
cs = d.cs
RESULTS: List[Tuple[str, bool, str]] = []
REFERENCE = d.DEFAULT_REFERENCE_JSON


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def run_main(argv: List[str]) -> Tuple[int, str]:
    """Runs the driver in-process, capturing its (long) stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = d.main(argv)
    return rc, buf.getvalue()


def base_args(results_dir: str, label: str) -> List[str]:
    return ["--use-mock", "--results-dir", results_dir, "--run-label", label]


def load_items(results_dir: str, label: str) -> Dict[str, Any]:
    with open(os.path.join(results_dir, f"checkpoint_{label}.json")) as fh:
        return json.load(fh)


def fake_info(**pkgs: Any) -> Dict[str, Any]:
    """A synthetic probe result: everything present at a benign version unless overridden with
    `name=None` (missing) or `name='x.y.z'`."""
    packages = {n: {"version": "1.0", "error": None} for n in d.PROBE_PACKAGES}
    packages["transformers"]["version"] = "4.57.0"
    for name, ver in pkgs.items():
        packages[name] = {"version": ver, "error": None if ver else "ModuleNotFoundError: simulated"}
    return {"python": "3.10", "executable": "x", "prefix": "x", "in_venv": True, "packages": packages,
            "cv2_aruco": True, "transformers_has_qwen2_5_vl": True, "transformers_has_paligemma": True,
            "jetson": {}}


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="arm2_jetson_drv_")
    try:
        print("== environment probe / static feasibility rules ==")
        ok_info = fake_info()
        check("healthy environment: no global blockers", d.global_blockers(ok_info) == [])
        check("missing PyAV is a global blocker", any("av" in b for b in d.global_blockers(fake_info(av=None))))
        no_aruco = fake_info()
        no_aruco["cv2_aruco"] = False
        check("cv2 without aruco is a global blocker", any("aruco" in b for b in d.global_blockers(no_aruco)))
        check("Moondream2 + transformers 5.x is BLOCKED",
              any(lv == "BLOCKED" for lv, _ in d.candidate_issues("moondream2", fake_info(transformers="5.17.0"))))
        check("Moondream2 + transformers 4.57 is fine",
              not any(lv == "BLOCKED" for lv, _ in d.candidate_issues("moondream2", ok_info)))
        check("missing accelerate blocks device_map candidates",
              any(lv == "BLOCKED" for lv, _ in d.candidate_issues("paligemma2_3b_mix", fake_info(accelerate=None))))
        check("Qwen without qwen_vl_utils is BLOCKED",
              any(lv == "BLOCKED" for lv, _ in d.candidate_issues("qwen2_5_vl_3b_instruct", fake_info(qwen_vl_utils=None))))
        check("missing timm / einops BLOCK InternVL2.5 (verified: its modeling files import them unconditionally)",
              [lv for lv, _ in d.candidate_issues("internvl2_5_4b", fake_info(timm=None))] == ["BLOCKED"]
              and [lv for lv, _ in d.candidate_issues("internvl2_5_4b", fake_info(einops=None))] == ["BLOCKED"])
        check("missing einops / torchvision BLOCK Moondream2 (its remote code imports them at module top level)",
              any(lv == "BLOCKED" for lv, _ in d.candidate_issues("moondream2", fake_info(einops=None)))
              and any(lv == "BLOCKED" for lv, _ in d.candidate_issues("moondream2", fake_info(torchvision=None))))
        rc, out = run_main(["--stage", "probe", "--results-dir", tmp])
        check("--stage probe runs and exits 0 in this (fully provisioned dev) environment", rc == 0 and "Per-candidate feasibility" in out)

        print("\n== preflight drops only the incompatible candidates, loudly ==")
        real_args = d.build_arg_parser().parse_args(["--results-dir", tmp])
        specs_real = cs.default_candidate_specs()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            kept = d.preflight(real_args, fake_info(transformers="5.17.0"), specs_real)
        kept_keys = [s.key for s in kept]
        check("under transformers 5: moondream2 specs dropped, the other four kept",
              kept_keys == ["qwen2_5_vl_3b", "internvl2_5_4b", "paligemma2_3b_mix:prompt", "paligemma2_3b_mix:detect"]
              and "SKIPPING moondream2:query" in buf.getvalue(), str(kept_keys))
        moon_only = [s for s in specs_real if s.backend == "moondream2"]
        raised = False
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                d.preflight(real_args, fake_info(transformers="5.17.0"), moon_only)
        except SystemExit:
            raised = True
        check("nothing runnable in this environment -> refuses to start", raised)

        print("\n== build_specs ==")
        a = d.build_arg_parser().parse_args(["--results-dir", tmp, "--candidates", "moondream2:point", "internvl3_5_4b_hf",
                                              "--qwen-model-dir", "/models/qwen"])
        sp = d.build_specs(a)
        check("--candidates picks exactly those keys, in order, incl. the optional InternVL3.5 fallback",
              [s.key for s in sp] == ["moondream2:point", "internvl3_5_4b_hf"], str([s.key for s in sp]))
        a2 = d.build_arg_parser().parse_args(["--results-dir", tmp, "--qwen-model-dir", "/models/qwen"])
        q = next(s for s in d.build_specs(a2) if s.key == "qwen2_5_vl_3b")
        check("--qwen-model-dir overrides the HF repo id; default set is the 6 standard specs (no fallback)",
              q.kwargs["model_dir"] == "/models/qwen" and len(d.build_specs(a2)) == 6)
        a3 = d.build_arg_parser().parse_args(["--results-dir", tmp, "--use-mock", "--candidates", "nope"])
        bad = False
        try:
            d.build_specs(a3)
        except SystemExit:
            bad = True
        check("unknown --candidates -> readable SystemExit", bad)

        print("\n== reference-locked frames + parity ==")
        rc, out = run_main(base_args(tmp, "lock") + ["--stage", "frames"])
        check("default session set is the reference run's (60 frames, parity identical)",
              rc == 0 and "reference-locked, 10 sessions" in out and "60 frames prepared" in out and "'identical': True" in out)
        orig_par = cs.sampling_parity
        cs.sampling_parity = lambda fr, ref: {"identical": False, "n_mine": 1, "n_reference": 60}
        raised = False
        try:
            run_main(base_args(tmp, "lock") + ["--stage", "frames"])
        except SystemExit:
            raised = True
        finally:
            cs.sampling_parity = orig_par
        check("a parity mismatch aborts the run", raised)

        print("\n== reference run: one uninterrupted mock invocation, all candidates ==")
        ref_dir = os.path.join(tmp, "ref")
        rc, out = run_main(base_args(ref_dir, "ref") + ["--stage", "all", "--smoke-frames", "1"])
        ref = load_items(ref_dir, "ref")
        expected_items = 60 * (3 * 4) + 60 * 2
        check("stage all completes; expected item count; validation gate skipped under --use-mock",
              rc == 0 and len(ref["items"]) == expected_items and "validation gate: skipped" in out,
              f"{len(ref['items'])} vs {expected_items}")
        check("combined results + aggregate CSV written",
              os.path.exists(os.path.join(ref_dir, "arm2_colab_sweep_results_ref.json"))
              and os.path.exists(os.path.join(ref_dir, "aggregate_table_ref.csv")))
        check("each candidate's stage record carries the producing environment",
              all("env" in ref["stages"][k] and "transformers" in ref["stages"][k]["env"]
                  for k in ("qwen2_5_vl_3b", "moondream2:point")))

        print("\n== multi-venv usage: two separate invocations, one shared local checkpoint ==")
        split_dir = os.path.join(tmp, "split")
        first = ["qwen2_5_vl_3b", "internvl2_5_4b"]
        second = ["paligemma2_3b_mix:prompt", "moondream2:query", "paligemma2_3b_mix:detect", "moondream2:point"]
        run_main(base_args(split_dir, "s") + ["--stage", "sweep", "--candidates", *first])
        n_first = len(load_items(split_dir, "s")["items"])
        rc, out2 = run_main(base_args(split_dir, "s") + ["--stage", "sweep", "--candidates", *second])
        both = load_items(split_dir, "s")
        check("second invocation adds its candidates without touching the first's",
              n_first == 60 * 3 * 2 and len(both["items"]) == expected_items and rc == 0, f"{n_first} then {len(both['items'])}")
        rc, out3 = run_main(base_args(split_dir, "s") + ["--stage", "sweep"])
        check("re-running everything is a no-op: every candidate 'nothing_to_do', nothing recomputed",
              "nothing left to do" in out3 and len(load_items(split_dir, "s")["items"]) == expected_items)
        check("split run's items are identical to the single uninterrupted run's",
              set(both["items"]) == set(ref["items"]) and all(
                  both["items"][k].get("raw_text") == ref["items"][k].get("raw_text")
                  and abs((both["items"][k].get("error_mm") or 0) - (ref["items"][k].get("error_mm") or 0)) < 1e-9
                  for k in both["items"]))
        rc, out4 = run_main(base_args(split_dir, "s") + ["--stage", "export"])
        check("--stage export from a third invocation sees everything (aggregate has no 'not run' rows)",
              rc == 0 and "not run" not in out4)

        print("\n== a REAL process kill mid-sweep, then resume ==")
        kill_dir = os.path.join(tmp, "kill")
        os.makedirs(kill_dir)
        sentinel = os.path.join(kill_dir, "reached_40_puts.flag")
        argv = base_args(kill_dir, "k") + ["--stage", "sweep"]
        code = f"""
import sys
sys.path.insert(0, r"{_HOST_SOFTWARE_DIR}"); sys.path.insert(0, r"{_REPO_ROOT_DIR}")
from ml_jetson_vla.deployment import run_arm2_sweep_jetson as d
d._load_engine()
cs = d.cs
_orig = cs.mock_candidate_specs
def _patched():
    specs = _orig()
    for s in specs:
        s.kwargs = dict(s.kwargs, sleep_s=0.03)   # non-identity kwarg: makes the run slow enough to kill
    return specs
cs.mock_candidate_specs = _patched
_n = {{"c": 0}}
_put = cs.CheckpointStore.put
def _counting_put(self, key, entry):
    _put(self, key, entry)
    _n["c"] += 1
    if _n["c"] == 40:
        open(r"{sentinel}", "w").write("x")
cs.CheckpointStore.put = _counting_put
sys.exit(d.main({argv!r}))
"""
        log = open(os.path.join(kill_dir, "child.log"), "w")
        proc = subprocess.Popen([sys.executable, "-c", code], stdout=log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while not os.path.exists(sentinel) and proc.poll() is None and time.time() - t0 < 180:
            time.sleep(0.25)
        reached = os.path.exists(sentinel)
        time.sleep(0.3)
        was_running = proc.poll() is None
        proc.kill()  # hard kill (TerminateProcess / SIGKILL): no cleanup handlers run
        proc.wait()
        log.close()
        check("child reached 40 finished calls and was still running when hard-killed", reached and was_running,
              f"reached={reached} running={was_running}")
        with open(os.path.join(kill_dir, "checkpoint_k.json")) as fh:
            partial = json.load(fh)  # must still be valid JSON after a hard kill (atomic os.replace)
        n_partial = len(partial["items"])
        check("checkpoint on disk is valid JSON and holds the finished calls (>=40, < total)",
              40 <= n_partial < expected_items, f"{n_partial} of {expected_items}")
        rc, out5 = run_main(argv)  # fresh process state, normal (fast) mock
        resumed = load_items(kill_dir, "k")
        check("resume completes the run", rc == 0 and len(resumed["items"]) == expected_items, f"{len(resumed['items'])}")
        already = sum(int(m) for m in re.findall(r"(\d+)/\d+ already done", out5))
        check("resume did not redo finished work (the per-candidate 'N/M already done' banners sum to the "
              "partial count found on disk)", already == n_partial, f"banners sum {already}, on disk {n_partial}")
        check("resumed run's items identical to the uninterrupted run's",
              set(resumed["items"]) == set(ref["items"]) and all(
                  resumed["items"][k].get("raw_text") == ref["items"][k].get("raw_text")
                  and abs((resumed["items"][k].get("error_mm") or 0) - (ref["items"][k].get("error_mm") or 0)) < 1e-9
                  for k in resumed["items"]))

        print("\n== validation gate: applicability, recorded outcome, enforcement across invocations ==")
        args_r = d.build_arg_parser().parse_args(["--results-dir", os.path.join(tmp, "gate"), "--run-label", "g"])
        mock_specs = cs.mock_candidate_specs()
        check("gate applicable with the real defaults + Qwen present", d.gate_applicable(args_r, mock_specs)[0])
        check("gate not applicable when Qwen is not in this invocation (the transformers<5 env's case)",
              not d.gate_applicable(args_r, [s for s in mock_specs if s.key != "qwen2_5_vl_3b"])[0])
        args_x = d.build_arg_parser().parse_args(["--results-dir", "x", "--max-new-tokens", "32"])
        check("gate not applicable when max-new-tokens differs from the reference run", not d.gate_applicable(args_x, mock_specs)[0])
        args_y = d.build_arg_parser().parse_args(["--results-dir", "x", "--all-matching-sessions"])
        check("gate not applicable with --all-matching-sessions", not d.gate_applicable(args_y, mock_specs)[0])

        gate_dir = os.path.join(tmp, "gate")
        os.makedirs(gate_dir)
        frames, _ = cs.prepare_frames(d.DEFAULT_BRONZE_DIR, 6, sessions=d.reference_sessions(REFERENCE), verbose=False)
        env = {"transformers": "test"}

        # (a) sweep with no gate record is refused; mock/skip flags bypass; a fake store shows the rule
        class _S:  # minimal store stand-in
            def __init__(self, stages: Dict[str, Any]) -> None:
                self.stages = stages

        refused = False
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                d.require_gate_record(args_r, _S({}))
        except SystemExit:
            refused = True
        check("sweep without a gate record is refused", refused)
        for label, stages, expect_ok in [
            ("passing record accepted", {d.GATE_STAGE_KEY: {"passed": True}}, True),
            ("knowingly-forced record accepted", {d.GATE_STAGE_KEY: {"passed": False, "forced": True}}, True),
            ("failed, unforced record refused", {d.GATE_STAGE_KEY: {"passed": False, "forced": False}}, False),
        ]:
            ok = True
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    d.require_gate_record(args_r, _S(stages))
            except SystemExit:
                ok = False
            check(label, ok == expect_ok)

        # (b) the gate FAILS on an oracle mock posing as Qwen and records the failure
        args_g = d.build_arg_parser().parse_args(["--results-dir", gate_dir, "--run-label", "g"])
        raised = False
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                d.stage_validate(args_g, mock_specs, frames, None, env)
        except cs.PipelineValidationError:
            raised = True
        rec = load_items(gate_dir, "g")["stages"].get(d.GATE_STAGE_KEY, {})
        check("gate FAILS on an oracle mock (not mistaken for the real local Qwen run) and records passed=False",
              raised and rec.get("passed") is False and rec.get("forced") is False and rec.get("failed_checks"))
        # (c) --force-continue: proceeds, but the record says forced
        args_f = d.build_arg_parser().parse_args(["--results-dir", gate_dir, "--run-label", "g", "--force-continue"])
        with contextlib.redirect_stdout(io.StringIO()):
            d.stage_validate(args_f, mock_specs, frames, None, env)
        rec = load_items(gate_dir, "g")["stages"][d.GATE_STAGE_KEY]
        check("--force-continue proceeds and records forced=True", rec["passed"] is False and rec["forced"] is True)
        # (d) the gate PASSES when fed the reference outputs themselves, and the record enables a sweep
        pass_dir = os.path.join(tmp, "gatepass")
        os.makedirs(pass_dir)
        store = cs.CheckpointStore(os.path.join(pass_dir, "checkpoint_p.json"))
        ref_doc = json.load(open(REFERENCE))
        qspec = next(s for s in mock_specs if s.key == "qwen2_5_vl_3b")
        for s in ref_doc["sessions"]:
            for v in ("baseline", "oriented"):
                for f in s["variants"][v]["frames"]:
                    fr = next(x for x in frames if x.session == s["session"] and x.frame_index == f["frame_index"])
                    store.items[cs.item_key(qspec, v, fr, 24)] = dict(f, candidate=qspec.key, variant=v, session=s["session"])
        store.flush()
        args_p = d.build_arg_parser().parse_args(["--results-dir", pass_dir, "--run-label", "p", "--skip-coord-calibration"])
        gate_out = io.StringIO()
        with contextlib.redirect_stdout(gate_out):
            d.stage_validate(args_p, mock_specs, frames, None, env)
        rec = load_items(pass_dir, "p")["stages"][d.GATE_STAGE_KEY]
        check("gate PASSES when fed the reference outputs, and records passed=True", rec["passed"] is True and rec["forced"] is False)
        gtxt = gate_out.getvalue()
        check("gate prints hits/mean next to the local reference (baseline 40/60 @ 27.9 mm, oriented 21/60 @ 43.7 mm, read from the file)",
              gtxt.count("40/60 hits, mean 27.9 mm") == 2 and gtxt.count("21/60 hits, mean 43.7 mm") == 2, gtxt[-600:])
        other = [s for s in mock_specs if s.key == "moondream2:point"]
        with contextlib.redirect_stdout(io.StringIO()):
            d.stage_sweep(args_p, other, frames, None, env)
        check("a later sweep (no Qwen in its candidates) is allowed once the gate record passed",
              any(k.startswith("moondream2:point|") for k in load_items(pass_dir, "p")["items"]))

        # (e) end-to-end through main() with a NON-mock path: a run with no gate record must be refused
        # BEFORE any model work (patch run_sweep to count calls; patch the probe so this dev machine's
        # real packages don't matter). Then --skip-validation-gate must let the same command through.
        early_dir = os.path.join(tmp, "early")
        real_probe, real_run = d.probe_environment, cs.run_sweep
        calls = {"n": 0}
        d.probe_environment = lambda: fake_info()
        cs.run_sweep = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or {})
        cmd = ["--results-dir", early_dir, "--run-label", "e", "--candidates", "moondream2:point", "--stage", "all"]
        try:
            refused = False
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    d.main(cmd)
            except SystemExit:
                refused = True
            check("--stage all in an env without Qwen and without a gate record is refused before any model work",
                  refused and calls["n"] == 0, f"run_sweep calls={calls['n']}")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = d.main(["--results-dir", early_dir, "--run-label", "e", "--candidates", "moondream2:point",
                             "--stage", "sweep", "--skip-validation-gate", "--skip-coord-calibration"])
            check("--skip-validation-gate (+ --skip-coord-calibration) lets the same sweep proceed (knowingly)",
                  rc == 0 and calls["n"] == 1, f"calls={calls['n']}")
            calls["n"] = 0
            with contextlib.redirect_stdout(io.StringIO()):
                rc2 = d.main(["--results-dir", early_dir, "--run-label", "e", "--candidates", "moondream2:point",
                              "--stage", "sweep", "--skip-validation-gate"])
            check("without a calibration record a real sweep is refused (exit 2, no model work), even with the gate skipped",
                  rc2 == 2 and calls["n"] == 0, f"rc={rc2} calls={calls['n']}")
        finally:
            d.probe_environment, cs.run_sweep = real_probe, real_run
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name, _ok, detail in failed:
        print(f"  FAILED: {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
