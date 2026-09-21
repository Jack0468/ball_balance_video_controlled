"""Local dry run of the whole Arm 2 Colab sweep pipeline with MOCK backends -- no GPU, no weights.

Runs the real code path (`colab_sweep.py`): real Track 4 video decode, real per-frame ArUco
homography, real prompt building + parser, real corrected scoring, real checkpoint/resume,
against mock backends that answer in each candidate's native output format (JSON, Qwen point_2d,
PaliGemma `<loc>` tokens, Moondream native points). What this can and cannot show is stated in
`docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md` -- in short: plumbing, not model behaviour.

Usage (from host_software/):  python ml_jetson_vla/deployment/test_colab_sweep_mock.py
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import List, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from ml_jetson_vla.core.mock_vlm_backends import SimulatedInterrupt  # noqa: E402
from ml_jetson_vla.deployment import colab_sweep as cs  # noqa: E402

BRONZE = os.path.join(_HOST_SOFTWARE_DIR, "data", "01_bronze")
REFERENCE = os.path.join(_THIS_DIR, "arm2_minimal_baseline_prompt_ab_scoring_20260918_RESCORED_v2.json")
RESULTS: List[Tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="arm2_mock_")
    ckpt = os.path.join(tmp, "ckpt.json")
    try:
        print("== frames ==")
        frames, skipped = cs.prepare_frames(BRONZE, frames_per_session=6, verbose=False)
        check("10 sessions x 6 frames prepared, none skipped", len(frames) == 60 and not skipped,
              f"{len(frames)} frames, skipped={skipped}")
        par = cs.sampling_parity(frames, REFERENCE)
        check("sampled (session, frame_index) set identical to the local 2026-09-18 run", par["identical"], str(par))

        variants = ["baseline", "oriented", "oriented_aruco"]
        specs = cs.mock_candidate_specs()

        print("\n== uninterrupted reference run ==")
        store_a = cs.CheckpointStore(os.path.join(tmp, "a.json"))
        st = cs.run_sweep(specs, frames, variants, store_a, log_every_call=False)
        check("all 6 mock candidates complete", all(s == "complete" for s in st.values()), str(st))
        expected_items = 60 * (3 * 4) + 60 * 2  # 4 prompt-sensitive x 3 variants + 2 native x 1 variant
        check("expected item count", len(store_a.items) == expected_items, f"{len(store_a.items)} vs {expected_items}")
        table = cs.aggregate(store_a, specs, frames, variants)
        print(table[["candidate", "variant", "n_done", "parse_rate", "hit@20mm", "mean_err_mm", "median_err_mm"]].to_string())

        # Oracle accuracy through the REAL scoring path: colour-command frames should land within a
        # few mm; go_black (the oracle answers the image centre on purpose) is excluded.
        colour = [e for e in store_a.items.values() if e["candidate"] == "qwen2_5_vl_3b"
                  and e["variant"] == "baseline" and e["instruction"] != "go_black" and e.get("parse_ok")]
        med = float(np.median([e["error_mm"] for e in colour]))
        legacy_med = float(np.median([e["error_mm_legacy"] for e in colour]))
        check("oracle through corrected scoring: median error on colour frames < 12mm", med < 12.0,
              f"median={med:.1f}mm (n={len(colour)})")
        check("same oracle outputs under the OLD wrong-frame comparison are far off (>60mm)", legacy_med > 60.0,
              f"legacy median={legacy_med:.1f}mm")
        check("model_input coordinate space handled (Qwen-style mock answers in resized 364x504 space)",
              all(e["coord_space"] == "model_input" and e["model_input_hw"] == [364, 504] for e in colour))
        alt = float(np.median([e["error_mm_alt_raw"] for e in colour]))
        check("...and reading those answers as raw pixels (the old bug) is clearly worse", alt > med + 15,
              f"alt_raw median={alt:.1f} vs {med:.1f}")
        for cand in ("internvl2_5_4b", "paligemma2_3b_mix:prompt", "moondream2:query",
                     "paligemma2_3b_mix:detect", "moondream2:point"):
            es = [e for e in store_a.items.values()
                  if e["candidate"] == cand and e["instruction"] != "go_black" and e.get("parse_ok")]
            m = float(np.median([e["error_mm"] for e in es])) if es else float("nan")
            check(f"native format parsed + scored for {cand}", len(es) > 30 and m < 12.0, f"n={len(es)} median={m:.1f}mm")
        reasons = {e["parse_reason"] for e in store_a.items.values()}
        check("all four native formats exercised by the parser",
              {"asked_json_contract", "qwen_point_2d", "paligemma2_loc_tokens", "native_points"} <= reasons, str(sorted(reasons)))

        print("\n== oriented_aruco prompt ==")
        from ml_jetson_vla.core.minimal_vlm_policy import build_prompt
        p_o = build_prompt("go_green", 640, 480, "oriented")[0]
        p_a = build_prompt("go_green", 640, 480, "oriented_aruco")[0]
        check("oriented_aruco differs from oriented and contains all 6 manifest marker IDs + coords + size",
              p_a != p_o and all(f"ID {i}" in p_a for i in range(6)) and "(175.5, 130.0)" in p_a and "22.5mm" in p_a)
        check("checkpoint keys differ per prompt variant",
              cs.item_key(specs[0], "oriented", frames[0], 24) != cs.item_key(specs[0], "oriented_aruco", frames[0], 24))

        print("\n== interrupt + resume ==")
        interrupted_specs = cs.mock_candidate_specs()
        interrupted_specs[1].kwargs = dict(interrupted_specs[1].kwargs, interrupt_after_calls=7)
        store_b = cs.CheckpointStore(ckpt)
        raised = False
        try:
            cs.run_sweep(interrupted_specs, frames, variants, store_b, log_every_call=False)
        except SimulatedInterrupt:
            raised = True
        check("simulated KeyboardInterrupt propagates out of run_sweep (not swallowed by isolation)", raised)
        n_at_interrupt = len(store_b.items)
        on_disk = len(json.load(open(ckpt))["items"])
        expected_partial = 60 * 3 + 7  # Qwen mock complete (180) + 7 finished InternVL-mock calls
        check("checkpoint on disk holds every finished call (nothing lost)",
              on_disk == n_at_interrupt == expected_partial,
              f"in-memory={n_at_interrupt} on-disk={on_disk} expected={expected_partial}")
        store_c = cs.CheckpointStore(ckpt)  # fresh "process": state comes only from disk
        loaded: List[str] = []
        orig_build = cs.build_backend

        def spy_build(spec: cs.CandidateSpec, mnt: int):
            loaded.append(spec.key)
            return orig_build(spec, mnt)

        cs.build_backend = spy_build
        try:
            st2 = cs.run_sweep(specs, frames, variants, store_c, log_every_call=False)
        finally:
            cs.build_backend = orig_build
        check("resume: completed candidate skipped without loading its model",
              st2["qwen2_5_vl_3b"] == "nothing_to_do" and "qwen2_5_vl_3b" not in loaded, str(st2))
        check("resume: remaining candidates complete", all(s in ("complete", "nothing_to_do") for s in st2.values()), str(st2))
        check("resumed run's items identical to the uninterrupted run's (deterministic mock)",
              set(store_c.items) == set(store_a.items) and all(
                  store_c.items[k].get("raw_text") == store_a.items[k].get("raw_text")
                  and abs((store_c.items[k].get("error_mm") or 0) - (store_a.items[k].get("error_mm") or 0)) < 1e-9
                  for k in store_c.items))

        print("\n== fault isolation ==")
        bad_specs = cs.mock_candidate_specs()
        bad_specs[1].kwargs = dict(bad_specs[1].kwargs, fail_on_load=True)                     # load failure
        bad_specs[2].kwargs = dict(bad_specs[2].kwargs, raise_on_call_indices=range(0, 50))     # every call fails
        bad_specs[3].hf_gated_repo = "google/definitely-not-accessible"                         # pretend gated
        store_d = cs.CheckpointStore(os.path.join(tmp, "d.json"))
        orig_check = cs.check_hf_access
        cs.check_hf_access = lambda repo, tok: (False, "simulated: license not accepted")
        try:
            st3 = cs.run_sweep(bad_specs, frames[:12], variants, store_d, log_every_call=False)
        finally:
            cs.check_hf_access = orig_check
        check("load failure isolated and recorded with a traceback",
              st3["internvl2_5_4b"] == "load_failed" and any(
                  e["stage"] == "load" and e["candidate"] == "internvl2_5_4b" and "Traceback" in e["traceback"]
                  for e in store_d.errors))
        check("candidate whose every call fails is aborted after 3 consecutive errors",
              st3["paligemma2_3b_mix:prompt"] == "aborted" and sum(
                  1 for e in store_d.errors if e["candidate"] == "paligemma2_3b_mix:prompt" and e["stage"] == "generate"
              ) == cs.MAX_CONSECUTIVE_CALL_ERRORS)
        check("gated candidate skipped gracefully (no load attempted)", st3["moondream2:query"] == "gated_skipped")
        check("the rest of the sweep still ran", st3["qwen2_5_vl_3b"] == "complete" and st3["moondream2:point"] == "complete", str(st3))
        check("failed calls are NOT stored as done (a re-run retries them)",
              not any(e["candidate"] == "paligemma2_3b_mix:prompt" for e in store_d.items.values()))

        print("\n== prompt overrides ==")
        cs.register_prompt_overrides({"my_variant": cs.PROMPT_VARIANTS["baseline"] + " Be careful."})
        k1 = cs.item_key(specs[0], "my_variant", frames[0], 24)
        cs.register_prompt_overrides({"my_variant": cs.PROMPT_VARIANTS["baseline"] + " Be VERY careful."})
        k2 = cs.item_key(specs[0], "my_variant", frames[0], 24)
        check("editing an override template changes the checkpoint key (no stale reuse)", k1 != k2)
        bad = False
        try:
            cs.register_prompt_overrides({"broken": 'answer as {"target_point_xy": [x, y]} for {target_label}'})
        except ValueError:
            bad = True
        check("un-doubled JSON braces in an override give a readable error", bad)

        print("\n== export / validation gate ==")
        out = cs.export_all(store_c, specs, frames, variants, os.path.join(tmp, "out"), run_label="mock")
        doc = json.load(open(out))
        one = doc["candidates"]["qwen2_5_vl_3b"]
        check("export is in the local scorer's format",
              {"backend", "prompt_variants", "tolerance_mm", "sessions"} <= set(one) and len(one["sessions"]) == 10
              and "frames" in one["sessions"][0]["variants"]["baseline"] and one["scoring_frame_version"] == 2)
        rep = cs.validate_against_reference(store_c, specs[0], frames, REFERENCE)
        cs.print_validation_report(rep)
        check("gate can FAIL: an oracle mock is not mistaken for the real local Qwen run", rep["passed"] is False)
        replay = cs.CheckpointStore(os.path.join(tmp, "replay.json"))
        ref = json.load(open(REFERENCE))
        for s in ref["sessions"]:
            for v in ("baseline", "oriented"):
                for f in s["variants"][v]["frames"]:
                    fr = next(x for x in frames if x.session == s["session"] and x.frame_index == f["frame_index"])
                    replay.items[cs.item_key(specs[0], v, fr, 24)] = dict(f, candidate=specs[0].key, variant=v, session=s["session"])
        rep2 = cs.validate_against_reference(replay, specs[0], frames, REFERENCE)
        check("gate can PASS: fed the reference outputs themselves it passes", rep2["passed"],
              str([c for c in rep2["checks"] if not c["ok"]]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name, _ok, detail in failed:
        print(f"  FAILED: {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
