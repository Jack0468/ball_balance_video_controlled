"""Local checks for the coordinate-space calibration probe (`deployment/coord_space_probe.py`) and its
wiring into the Jetson driver (`--stage calibrate` + the sweep-time refusal), using MOCK backends
that answer in a chosen coordinate space. No GPU, no weights, no network.

What the mocks prove: the probe's analysis identifies each hypothesised answer space correctly, flags
a backend whose declared `coord_space` disagrees with what it really answers, shows a letterbox offset
in the fitted intercept, warns (does not refuse) on a model that cannot localize, and that the driver
records/enforces/overrides the outcome. What only a real Jetson run can show: what the real models
actually answer on these frames.

Usage (from host_software/):  python ml_jetson_vla/deployment/test_coord_space_probe_mock.py
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from ml_jetson_vla.core.minimal_vlm_policy import COORD_SPACES, to_raw_px  # noqa: E402
from ml_jetson_vla.core.mock_vlm_backends import MockBackend  # noqa: E402
from ml_jetson_vla.deployment import coord_space_probe as csp  # noqa: E402
from ml_jetson_vla.deployment import run_arm2_sweep_jetson as d  # noqa: E402

d._load_engine()
cs = d.cs
RESULTS: List[Tuple[str, bool, str]] = []
MI_HW = (364, 504)


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def probe(**mock_kwargs: Any) -> Dict[str, Any]:
    """Runs the real probe path (MinimalVLMPolicy -> backend -> parser) against a mock and analyses it."""
    kw = dict(flavor="json", full_frame_search=True)
    kw.update(mock_kwargs)
    backend = MockBackend(**kw)
    obs = csp.run_probe_variant(backend, "baseline")
    return {"obs": obs, "analysis": csp.analyze(obs)}


def run_main(argv: List[str]) -> Tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = d.main(argv)
    return rc, buf.getvalue()


def load_ckpt(results_dir: str, label: str) -> Dict[str, Any]:
    with open(os.path.join(results_dir, f"checkpoint_{label}.json")) as fh:
        return json.load(fh)


class FakeStore:
    def __init__(self, stages: Dict[str, Any] | None = None) -> None:
        self.stages: Dict[str, Any] = stages or {}

    def set_stage(self, cand: str, **fields: Any) -> None:
        self.stages.setdefault(cand, {}).update(fields)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="arm2_coordprobe_")
    try:
        print("== probe design + synthetic frames ==")
        check("the shipped position set passes its own design validation", csp.validate_positions(csp.PROBE_POSITIONS) == [],
              str(csp.validate_positions(csp.PROBE_POSITIONS)))
        check(">= 6 positions, none point-symmetric/mirrored, reaching all four edges",
              len(csp.PROBE_POSITIONS) >= 6)
        sym = ((100, 100), (540, 380), (200, 300), (50, 60), (300, 50), (600, 400))  # first two are point-symmetric about (320,240)
        check("validate_positions flags a point-symmetric pair", any("point-symmetric" in p for p in csp.validate_positions(sym)))
        check("validate_positions flags too few positions and a disc that is not inside the frame",
              any("only 3" in p for p in csp.validate_positions(((50, 60), (300, 50), (600, 400))))
              and any("not fully inside" in p for p in csp.validate_positions(((5, 60),) + csp.PROBE_POSITIONS)))
        fr = csp.make_probe_frame(233, 41)
        red = np.all(fr == np.array(csp.DISC_BGR, np.uint8), axis=-1)
        ys, xs = np.nonzero(red)
        check("frame is 640x480, disc diameter ~30 px (area ~ pi r^2), centred on the requested position",
              fr.shape == (480, 640, 3) and abs(red.sum() - np.pi * 15 ** 2) < 30
              and abs(xs.mean() + 0.5 - 233) < 0.3 and abs(ys.mean() + 0.5 - 41) < 0.3, f"area={red.sum()}")
        check("background is neutral grey; frames are deterministic",
              np.all(fr[~red] == csp.BACKGROUND_GRAY) and csp.frame_sha1(fr) == csp.frame_sha1(csp.make_probe_frame(233, 41)))

        print("\n== to_raw_px: extended coordinate spaces, backward compatible ==")
        check("raw_image unchanged; model_input rescales; model_input without a size is unchanged (old behaviour)",
              to_raw_px(10, 20, "raw_image", None, (480, 640)) == (10.0, 20.0)
              and to_raw_px(252, 182, "model_input", (364, 504), (480, 640)) == (320.0, 240.0)
              and to_raw_px(7, 9, "model_input", None, (480, 640)) == (7.0, 9.0))
        check("norm1000 / norm1 map onto the raw frame; _yx swaps first",
              to_raw_px(500, 500, "norm1000", None, (480, 640)) == (320.0, 240.0)
              and to_raw_px(0.5, 0.5, "norm1", None, (480, 640)) == (320.0, 240.0)
              and to_raw_px(250, 500, "norm1000_yx", None, (480, 640)) == (320.0, 120.0))
        raised = False
        try:
            to_raw_px(1, 1, "norm100", None, (480, 640))
        except ValueError:
            raised = True
        check("an unknown coord_space raises instead of silently meaning raw pixels", raised)
        from ml_jetson_vla.deployment.score_minimal_baseline_offline import score_prediction
        H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        sp = score_prediction(H, 500.0, 500.0, 0.0, 0.0, 20.0, coord_space="raw_image", model_input_hw=MI_HW)
        check("score_prediction's error_mm_alt_* fields come from the same to_raw_px names (incl. new norm1)",
              all(k in sp for k in ("error_mm_alt_raw", "error_mm_alt_model_input", "error_mm_alt_norm1000", "error_mm_alt_norm1")))

        print("\n== every hypothesised answer space is identified correctly (declared == real) ==")
        for space in [f"{b}{sfx}" for b in COORD_SPACES for sfx in ("", "_yx")]:
            r = probe(answer_space=space, coord_space=space, model_input_hw=MI_HW)
            a = r["analysis"]
            check(f"answers in {space!r}: best hypothesis {space!r}, CONSISTENT, agrees with declared, error < 2 px",
                  a["best_hypothesis"] == space and a["verdict"] == csp.VERDICT_CONSISTENT and a["agrees_with_declared"]
                  and a["declared_mean_err_px"] < 2.0 and a["n_parsed"] == len(csp.PROBE_POSITIONS),
                  f"best={a['best_hypothesis']} verdict={a['verdict']} err={a['declared_mean_err_px']:.2f}")
        a = probe(answer_space="raw_image", coord_space="raw_image", model_input_hw=MI_HW)["analysis"]
        lf = a["linear_fit_vs_parsed"]
        check("pure resize/identity: fitted a ~ 1, b ~ 0 on both axes, tiny residuals",
              abs(lf["x"]["a"] - 1) < 0.01 and abs(lf["y"]["a"] - 1) < 0.01 and abs(lf["x"]["b"]) < 1 and abs(lf["y"]["b"]) < 1
              and lf["x"]["resid_rms_px"] < 1 and lf["valid"], str({k: lf[k] for k in ("x", "y")}))
        a = probe(answer_space="model_input", coord_space="model_input", model_input_hw=MI_HW)["analysis"]
        lf = a["linear_fit_vs_parsed"]
        check("model_input space: fitted a = raw/model_input scale (640/504, 480/364), b ~ 0",
              abs(lf["x"]["a"] - 640 / 504) < 0.01 and abs(lf["y"]["a"] - 480 / 364) < 0.01
              and abs(lf["x"]["b"]) < 1.5 and abs(lf["y"]["b"]) < 1.5)

        print("\n== the other output flavours, through the real parser ==")
        for flavor, extra in [("qwen_point2d", dict(answer_space="norm1000", coord_space="norm1000")),
                              ("paligemma_loc", {}), ("moondream_native", {})]:
            a = probe(flavor=flavor, **extra)["analysis"]
            check(f"flavor {flavor}: CONSISTENT with its declared space, error < 3 px",
                  a["verdict"] == csp.VERDICT_CONSISTENT and a["declared_mean_err_px"] < 3.0,
                  f"{a['verdict']} {a['declared_mean_err_px']:.2f} best={a['best_hypothesis']}")

        print("\n== a backend whose declared coord_space disagrees with what it answers is caught ==")
        for real, declared, kw in [("norm1000", "raw_image", {}), ("norm1", "raw_image", {}),
                                   ("model_input", "raw_image", {"model_input_hw": MI_HW}),
                                   ("raw_image_yx", "raw_image", {}), ("raw_image", "norm1000", {})]:
            a = probe(answer_space=real, coord_space=declared, **kw)["analysis"]
            check(f"answers {real!r} but declares {declared!r}: MISMATCH_NAMED, best={real!r}, agrees=False, enforce=True",
                  a["verdict"] == csp.VERDICT_MISMATCH_NAMED and a["best_hypothesis"] == real
                  and a["agrees_with_declared"] is False and a["enforce"] is True,
                  f"{a['verdict']} best={a['best_hypothesis']} declared_err={a['declared_mean_err_px']:.0f}")
        a = probe(answer_space="norm1000", coord_space="raw_image")["analysis"]
        rec = csp.recommendation("cand", "internvl2_5_4b", "baseline", a)
        check("the recommendation names the space to declare and the backend field to change",
              "coord_space='norm1000'" in rec and "InternVLBackend.generate()" in rec and "BackendOutput" in rec, rec[:200])

        print("\n== letterbox / crop: shows up in the fitted intercept, refused as unsupported ==")
        r = probe(answer_space="norm1000", coord_space="norm1000", letterbox_pad=(0, 80))
        a = r["analysis"]
        ly, lx = a["linear_fit_vs_parsed"]["y"], a["linear_fit_vs_parsed"]["x"]
        check("letterbox (80 px top/bottom): y intercept b ~ -80 px (x b ~ 0), a ~ 0.64, tight residual",
              abs(ly["b"] + 80) < 3 and abs(lx["b"]) < 3 and abs(ly["a"] - 0.64) < 0.01 and ly["resid_rms_px"] < 1.5,
              f"y: a={ly['a']} b={ly['b']}; x: a={lx['a']} b={lx['b']}")
        lfb = a["linear_fit_vs_best_hypothesis"]["y"]
        check("the offset is also visible relative to the best named hypothesis (b ~ -80 raw px)", abs(lfb["b"] + 80) < 3, str(lfb))
        check("no named hypothesis fits it, the linear map does -> MISMATCH_UNSUPPORTED (refusing)",
              a["verdict"] == csp.VERDICT_MISMATCH_UNSUPPORTED and a["enforce"] and a["explanation"] == "linear_fit"
              and a["best_hypothesis_mean_err_px"] > 15, f"{a['verdict']} named_best={a['best_hypothesis_mean_err_px']:.1f}")
        a2 = probe(flavor="paligemma_loc", loc_order="xy")["analysis"]
        check("X-before-Y <loc> tokens through the Y-first parser (mixed axis order) are caught as a refusing verdict",
              a2["verdict"] in csp.REFUSING_VERDICTS, f"{a2['verdict']}: {a2['detail'][:160]}")

        print("\n== a model that cannot localize: warn, never refuse ==")
        g = probe(garbage_seed=7)["analysis"]
        check("garbage answers -> NO_FIT, enforce=False, message states it is a capability finding, linear fit not valid",
              g["verdict"] == csp.VERDICT_NO_FIT and g["enforce"] is False and "CAPABILITY" in g["detail"]
              and not g["linear_fit_vs_parsed"]["valid"], f"{g['verdict']} best={g['explanation_mean_err_px']:.0f}px")
        obs_const = [dict(o, pred_xy=[320.0, 240.0]) for o in probe(answer_space="raw_image")["obs"]]
        gc = csp.analyze(obs_const)
        check("a constant centre answer is NO_FIT too", gc["verdict"] == csp.VERDICT_NO_FIT)
        allfail = probe(raise_on_call_indices=range(0, 20))
        af = allfail["analysis"]
        check("every call failing -> INSUFFICIENT (warn, not refuse); the run aborts after 3 consecutive failures",
              af["verdict"] == csp.VERDICT_INSUFFICIENT and af["enforce"] is False
              and sum(1 for o in allfail["obs"] if str(o.get("error", "")).startswith("not_run")) == len(csp.PROBE_POSITIONS) - 3)

        print("\n== hint-based model_input hypothesis (PaliGemma: 448 square) ==")
        true = np.array(csp.PROBE_POSITIONS, float)
        obs448 = [{"index": i, "true_xy": list(t), "parse_ok": True, "declared_space": "raw_image", "model_input_hw": None,
                   "pred_xy": [t[0] * 448 / 640, t[1] * 448 / 480]} for i, t in enumerate(true)]
        ah = csp.analyze(obs448, model_input_hw_hint=(448, 448))
        an = csp.analyze(obs448)
        check("with the 448x448 hint: MISMATCH_NAMED -> model_input (source recorded as hint); without: structured but unsupported",
              ah["verdict"] == csp.VERDICT_MISMATCH_NAMED and ah["best_hypothesis"] == "model_input"
              and ah["model_input_hw_source"] == "hint" and an["verdict"] == csp.VERDICT_MISMATCH_UNSUPPORTED,
              f"{ah['verdict']}/{an['verdict']}")
        sw = csp.fit_linear(true[:, ::-1].copy(), true)
        check("fit_linear detects an axis swap (sources y,x)", sw["x"]["source"] == "y" and sw["y"]["source"] == "x" and sw["valid"])

        print("\n== driver: --stage calibrate records + saves raw answers; the sweep refuses / warns / overrides ==")
        def mk(key: str, space: str, declared: str = "", **kw: Any) -> Any:
            kwargs = dict(flavor="json", full_frame_search=True, answer_space=space, coord_space=declared or space)
            kwargs.update(kw)
            return cs.CandidateSpec(key, "mock", kwargs)

        def specs_v1() -> List[Any]:
            return [mk("ok_raw", "raw_image"), mk("ok_qwenlike", "model_input", model_input_hw=MI_HW),
                    mk("liar", "norm1000", "raw_image"), mk("letterbox", "norm1000", letterbox_pad=(0, 80)),
                    mk("garbage", "raw_image", garbage_seed=3),
                    cs.CandidateSpec("ok_loc", "mock", dict(flavor="paligemma_loc", full_frame_search=True)),
                    cs.CandidateSpec("ok_native", "mock", dict(flavor="moondream_native", full_frame_search=True),
                                     variants=["baseline"], prompt_independent=True)]

        orig_specs = cs.mock_candidate_specs
        cs.mock_candidate_specs = specs_v1
        try:
            rdir = os.path.join(tmp, "drv")
            base = ["--use-mock", "--results-dir", rdir, "--run-label", "c"]
            rc, out = run_main(base + ["--stage", "calibrate"])
            stages = load_ckpt(rdir, "c")["stages"][d.CAL_STAGE_KEY]
            verdicts = {k: v["verdict"] for k, v in stages.items()}
            check("calibrate stage: exit 0 and the expected verdict per candidate, recorded in the checkpoint",
                  rc == 0 and verdicts == {"ok_raw": "CONSISTENT", "ok_qwenlike": "CONSISTENT", "liar": "MISMATCH_NAMED",
                                           "letterbox": "MISMATCH_UNSUPPORTED", "garbage": "NO_FIT", "ok_loc": "CONSISTENT",
                                           "ok_native": "CONSISTENT"}, str(verdicts))
            check("the refusing candidates carry the evidence + the field to change in their record",
                  "coord_space='norm1000'" in stages["liar"]["recommendations"][0] and "field to change" in stages["liar"]["recommendations"][0])
            pf = json.load(open(os.path.join(rdir, "coord_probe_c.json")))
            liar_obs = pf["candidates"]["liar"]["variants"]["baseline"]["observations"]
            check("raw answers saved: per candidate, every position with true xy, raw text, frame hash; frame spec recorded",
                  pf["format"] == csp.PROBE_FORMAT and len(liar_obs) == len(csp.PROBE_POSITIONS)
                  and all(o["raw_text"] and o["true_xy"] and o["frame_sha1"] for o in liar_obs)
                  and pf["frame_spec"]["instruction"] == "go_red" and len(pf["frame_spec"]["positions_xy"]) == len(csp.PROBE_POSITIONS)
                  and set(pf["candidates"]) == set(verdicts))
            check("the synthetic frames were written as PNGs for inspection",
                  len([f for f in os.listdir(os.path.join(rdir, "coord_probe_frames")) if f.endswith(".png")]) == len(csp.PROBE_POSITIONS))
            check("a native-API spec is probed once (baseline only)", list(pf["candidates"]["ok_native"]["variants"]) == ["baseline"])
            rc, out = run_main(base + ["--stage", "calibrate", "--candidates", "ok_raw", "liar"])
            check("re-running calibrate: non-refusing records are cached, the refusing one is re-run",
                  "cached calibration record" in out.split("--- liar")[0] and "cached calibration record" not in out.split("--- liar")[1])

            sweep = base + ["--stage", "sweep", "--frames-per-session", "1", "--prompt-variants", "baseline"]
            rc, out = run_main(sweep)
            ck = load_ckpt(rdir, "c")
            swept = {v["candidate"] for v in ck["items"].values()}
            check("sweep: liar and letterbox are REFUSED (exit 2, evidence printed), the rest run -- incl. the NO_FIT garbage one (warn only)",
                  rc == 2 and swept == {"ok_raw", "ok_qwenlike", "garbage", "ok_loc", "ok_native"}
                  and "REFUSED  liar" in out and "REFUSED  letterbox" in out and "field to change" in out
                  and "--allow-coord-mismatch liar" in out, f"rc={rc} swept={sorted(swept)}")
            check("refused candidates are marked in their stage record (status + reason); passing ones record their verdict",
                  ck["stages"]["liar"]["status"] == "calibration_refused"
                  and ck["stages"]["ok_raw"]["coord_calibration"] == {"verdict": "CONSISTENT", "override": False})
            rc, out = run_main(sweep + ["--allow-coord-mismatch", "liar", "letterbox"])
            ck = load_ckpt(rdir, "c")
            swept = {v["candidate"] for v in ck["items"].values()}
            check("--allow-coord-mismatch overrides knowingly: both are swept, exit 0, override recorded",
                  rc == 0 and {"liar", "letterbox"} <= swept and ck["stages"]["liar"]["coord_calibration"]["override"] is True
                  and "OVERRIDE liar" in out)

            print("\n-- fix the backend's declared space, recalibrate, and it passes --")
            def specs_v2() -> List[Any]:
                return [mk("liar", "norm1000", "norm1000")]
            rdir2 = os.path.join(tmp, "drv2")
            b2 = ["--use-mock", "--results-dir", rdir2, "--run-label", "f"]
            cs.mock_candidate_specs = lambda: [mk("liar", "norm1000", "raw_image")]
            run_main(b2 + ["--stage", "calibrate"])
            rc_bad, _ = run_main(b2 + ["--stage", "sweep", "--frames-per-session", "1", "--prompt-variants", "baseline"])
            cs.mock_candidate_specs = specs_v2  # the "fix": declared coord_space now matches (a changed config)
            rc_stale, out_stale = run_main(b2 + ["--stage", "sweep", "--frames-per-session", "1", "--prompt-variants", "baseline"])
            check("under --use-mock a stale record is tolerated only when it is not a refusal; a refusal of the OLD config no "
                  "longer applies to the fixed config", rc_bad == 2 and rc_stale == 0, f"{rc_bad}/{rc_stale}")
            run_main(b2 + ["--stage", "calibrate"])
            check("recalibrating the fixed backend gives CONSISTENT",
                  load_ckpt(rdir2, "f")["stages"][d.CAL_STAGE_KEY]["liar"]["verdict"] == "CONSISTENT")
        finally:
            cs.mock_candidate_specs = orig_specs

        print("\n== enforcement rules (no models): missing / stale records, skip flag, config identity ==")
        args = d.build_arg_parser().parse_args(["--results-dir", tmp])
        s_ok, s_liar = mk("a", "raw_image"), mk("b", "norm1000", "raw_image")
        with contextlib.redirect_stdout(io.StringIO()):
            keep, refused = d.enforce_calibration(args, FakeStore(), [s_ok, s_liar])
        check("real (non-mock) run with no calibration record: every candidate is refused", keep == [] and len(refused) == 2)
        stale = {d.CAL_STAGE_KEY: {"a": {"config_hash": "deadbeef", "verdict": "CONSISTENT"}}}
        with contextlib.redirect_stdout(io.StringIO()):
            keep, refused = d.enforce_calibration(args, FakeStore(stale), [s_ok])
        check("a record for a different configuration (stale hash) does not count", keep == [] and "stale" in refused[0]["reason"])
        fresh = {d.CAL_STAGE_KEY: {"a": {"config_hash": d.spec_config_hash(s_ok), "verdict": "CONSISTENT"},
                                   "b": {"config_hash": d.spec_config_hash(s_liar), "verdict": "MISMATCH_NAMED",
                                         "recommendations": ["evidence"]}}}
        with contextlib.redirect_stdout(io.StringIO()):
            keep, refused = d.enforce_calibration(args, FakeStore(fresh), [s_ok, s_liar])
        check("a fresh CONSISTENT record passes, a fresh MISMATCH record is refused",
              [s.key for s in keep] == ["a"] and [r["key"] for r in refused] == ["b"])
        with contextlib.redirect_stdout(io.StringIO()):
            keep, _ = d.enforce_calibration(d.build_arg_parser().parse_args(["--results-dir", tmp, "--allow-coord-mismatch", "b"]),
                                            FakeStore(fresh), [s_ok, s_liar])
            keep2, _ = d.enforce_calibration(d.build_arg_parser().parse_args(["--results-dir", tmp, "--skip-coord-calibration"]),
                                             FakeStore(), [s_ok, s_liar])
        check("--allow-coord-mismatch KEY and --skip-coord-calibration both let the candidate through", len(keep) == 2 and len(keep2) == 2)
        check("config hash changes when the backend configuration changes (answer space) and ignores fault-injection switches",
              d.spec_config_hash(mk("a", "raw_image")) != d.spec_config_hash(mk("a", "norm1000"))
              and d.spec_config_hash(mk("a", "raw_image")) == d.spec_config_hash(mk("a", "raw_image", sleep_s=0.5)))
        a_ = d.build_arg_parser().parse_args(["--results-dir", tmp, "--stage", "calibrate"])
        check("'calibrate' is a --stage choice; defaults: baseline-only variants", a_.stage == "calibrate" and a_.calibrate_variants == ["baseline"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name, _ok, detail in failed:
        print(f"  FAILED: {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
