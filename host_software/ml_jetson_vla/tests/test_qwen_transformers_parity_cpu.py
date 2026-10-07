"""CPU-only tests for deployment/qwen_transformers_parity.py (transformers 4.57.6 vs 5.17.0 Qwen parity).

Covers the pure comparison functions (exact match, pixel difference, verdict thresholds, summary), the resume
and refuse-to-overwrite rules, the frame list taken from the reference JSON, and the import-time guarantee that
transformers and torch are not loaded. No CUDA, no model weights, no transformers or torch import.

Run (repo convention: standalone, stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_jetson_vla/tests/test_qwen_transformers_parity_cpu.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Dict, List

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_jetson_vla.deployment import qwen_transformers_parity as qtp  # noqa: E402

REFERENCE_JSON: str = qtp.DEFAULT_REFERENCE


def _row(diff: float | None, exact: bool = True, parse_new: bool = True) -> Dict[str, Any]:
    """Minimal row shaped like build_row()'s output, for verdict/summary tests."""
    return {"exact_match": exact, "pixel_abs_diff": diff, "parse_ok_new": parse_new,
            "parse_ok_reference": True}


def _ref(session: str = "s1", frame_index: int = 3, parse_ok: bool = True) -> Dict[str, Any]:
    return {"session": session, "frame_index": frame_index, "instruction": "go_green",
            "raw_text": '{"target_point_xy": [10, 20]}', "parse_ok": parse_ok,
            "target_point_px": (10.0, 20.0) if parse_ok else None}


class ExactMatchAndDiffTests(unittest.TestCase):
    def test_exact_match_requires_both_strings_and_equality(self) -> None:
        self.assertTrue(qtp.exact_match("abc", "abc"))
        self.assertFalse(qtp.exact_match("abc", "abd"))
        self.assertFalse(qtp.exact_match(None, "abc"))
        self.assertFalse(qtp.exact_match("abc", None))
        self.assertFalse(qtp.exact_match(None, None))

    def test_exact_match_is_not_whitespace_tolerant(self) -> None:
        self.assertFalse(qtp.exact_match("abc ", "abc"))

    def test_pixel_abs_diff_is_euclidean(self) -> None:
        self.assertAlmostEqual(qtp.pixel_abs_diff((0.0, 0.0), (3.0, 4.0)), 5.0)
        self.assertEqual(qtp.pixel_abs_diff((7.0, 2.0), (7.0, 2.0)), 0.0)

    def test_pixel_abs_diff_none_when_either_point_missing(self) -> None:
        self.assertIsNone(qtp.pixel_abs_diff(None, (1.0, 1.0)))
        self.assertIsNone(qtp.pixel_abs_diff((1.0, 1.0), None))


class VerdictTests(unittest.TestCase):
    def test_identical_requires_every_frame_exact(self) -> None:
        rows = [_row(0.0, exact=True) for _ in range(60)]
        self.assertEqual(qtp.verdict(rows, 60), ("IDENTICAL", 0))

    def test_all_points_within_4px_but_text_differs_is_equivalent(self) -> None:
        rows = [_row(0.0, exact=True) for _ in range(59)] + [_row(2.5, exact=False)]
        self.assertEqual(qtp.verdict(rows, 60), ("EQUIVALENT", 0))

    def test_exactly_4px_still_equivalent_boundary_inclusive(self) -> None:
        rows = [_row(4.0, exact=False)]
        self.assertEqual(qtp.verdict(rows, 1), ("EQUIVALENT", 0))

    def test_just_over_4px_is_divergent_with_count(self) -> None:
        rows = [_row(0.0), _row(4.01, exact=False), _row(9.0, exact=False)]
        self.assertEqual(qtp.verdict(rows, 3), ("DIVERGENT", 2))

    def test_unparsed_frame_counts_as_divergent(self) -> None:
        rows = [_row(0.0), _row(None, exact=False, parse_new=False)]
        self.assertEqual(qtp.verdict(rows, 2), ("DIVERGENT", 1))

    def test_missing_frames_count_as_divergent_even_if_all_present_match(self) -> None:
        rows = [_row(0.0) for _ in range(59)]
        self.assertEqual(qtp.verdict(rows, 60), ("DIVERGENT", 1))

    def test_verdict_line_names_the_label_and_count(self) -> None:
        self.assertIn("IDENTICAL", qtp.verdict_line("IDENTICAL", 0, 60))
        self.assertIn("EQUIVALENT", qtp.verdict_line("EQUIVALENT", 0, 60))
        self.assertIn("DIVERGENT (7 of 60", qtp.verdict_line("DIVERGENT", 7, 60))


class RowAndSummaryTests(unittest.TestCase):
    def test_build_row_matching_text_and_points(self) -> None:
        row = qtp.build_row(_ref(), '{"target_point_xy": [10, 20]}', True, (10.0, 20.0), 1.5)
        self.assertTrue(row["exact_match"])
        self.assertEqual(row["pixel_abs_diff"], 0.0)
        self.assertEqual(row["target_point_px_new"], [10.0, 20.0])
        self.assertEqual(row["target_point_px_reference"], [10.0, 20.0])

    def test_build_row_unparsed_new_side_has_no_diff(self) -> None:
        row = qtp.build_row(_ref(), "prose", False, None, None)
        self.assertFalse(row["exact_match"])
        self.assertIsNone(row["target_point_px_new"])
        self.assertIsNone(row["pixel_abs_diff"])

    def test_summary_counts_means_and_max(self) -> None:
        rows = [_row(0.0), _row(2.0, exact=False), _row(6.0, exact=False, parse_new=False)]
        s = qtp.summarize(rows, 3)
        self.assertEqual(s["n_compared"], 3)
        self.assertEqual(s["n_exact_match"], 1)
        self.assertEqual(s["n_parse_ok_new"], 2)
        self.assertAlmostEqual(s["parse_rate_new"], 2 / 3)
        self.assertEqual(s["n_within_tolerance"], 2)
        self.assertAlmostEqual(s["mean_pixel_abs_diff"], 8.0 / 3.0)
        self.assertEqual(s["max_pixel_abs_diff"], 6.0)

    def test_summary_with_no_diffs_reports_none(self) -> None:
        s = qtp.summarize([_row(None, exact=False, parse_new=False)], 1)
        self.assertIsNone(s["mean_pixel_abs_diff"])
        self.assertIsNone(s["max_pixel_abs_diff"])


class ResumeTests(unittest.TestCase):
    def test_completed_output_is_refused(self) -> None:
        cfg = qtp.build_run_config("ref.json")
        with self.assertRaises(FileExistsError):
            qtp.check_resumable({"complete": True, "config": cfg, "frames": []}, cfg)

    def test_incomplete_checkpoint_with_different_config_is_refused(self) -> None:
        cfg = qtp.build_run_config("ref.json")
        other = dict(cfg, dtype="fp16")
        with self.assertRaises(ValueError):
            qtp.check_resumable({"complete": False, "config": other, "frames": []}, cfg)

    def test_incomplete_checkpoint_with_same_config_resumes(self) -> None:
        cfg = qtp.build_run_config("ref.json")
        self.assertEqual(qtp.check_resumable({"complete": False, "config": cfg, "frames": []}, cfg), "resume")

    def test_pending_skips_frames_already_in_output(self) -> None:
        ref_frames: List[Dict[str, Any]] = [_ref("s1", 1), _ref("s1", 2), _ref("s2", 1)]
        done = {qtp.frame_key("s1", 2)}
        pending = qtp.pending_reference_frames(ref_frames, done)
        self.assertEqual([(p["session"], p["frame_index"]) for p in pending], [("s1", 1), ("s2", 1)])

    def test_pending_is_empty_when_everything_done(self) -> None:
        ref_frames = [_ref("s1", 1)]
        self.assertEqual(qtp.pending_reference_frames(ref_frames, {qtp.frame_key("s1", 1)}), [])

    def test_run_parity_refuses_existing_completed_output_before_loading_model(self) -> None:
        if not os.path.exists(REFERENCE_JSON):
            self.skipTest("reference JSON not present on this machine")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "parity.json")
            with open(out, "w", encoding="utf-8") as fh:
                json.dump({"complete": True, "config": qtp.build_run_config(os.path.basename(REFERENCE_JSON)),
                           "frames": []}, fh)
            with self.assertRaises(FileExistsError):
                qtp.run_parity(REFERENCE_JSON, bronze_dir=tmp, output_path=out, force=False, verbose=False)


class ReferenceAndConfigTests(unittest.TestCase):
    def test_reference_config_matches_pinned_run_config(self) -> None:
        if not os.path.exists(REFERENCE_JSON):
            self.skipTest("reference JSON not present on this machine")
        ref_config, _ = qtp.load_reference_frames(REFERENCE_JSON)
        run_config = qtp.build_run_config(os.path.basename(REFERENCE_JSON))
        self.assertEqual(qtp.config_mismatches(ref_config, run_config), [])

    def test_config_mismatch_is_reported_by_key(self) -> None:
        mism = qtp.config_mismatches({"dtype": "bf16", "revision": "a"}, {"dtype": "fp32", "revision": "a"})
        self.assertTrue(any(m.startswith("dtype:") for m in mism))
        self.assertFalse(any(m.startswith("revision:") for m in mism))

    def test_reference_frames_are_the_sixty_baseline_frames(self) -> None:
        if not os.path.exists(REFERENCE_JSON):
            self.skipTest("reference JSON not present on this machine")
        _, frames = qtp.load_reference_frames(REFERENCE_JSON)
        self.assertEqual(len(frames), 60)
        self.assertEqual(len({qtp.frame_key(f["session"], f["frame_index"]) for f in frames}), 60)
        self.assertTrue(all(f["instruction"] for f in frames))


class ImportIsTorchFreeTests(unittest.TestCase):
    def test_importing_module_does_not_load_torch_or_transformers(self) -> None:
        code = (
            "import sys; sys.path.append(r'%s'); "
            "import ml_jetson_vla.deployment.qwen_transformers_parity; "
            "bad = [m for m in ('torch', 'transformers') if m in sys.modules]; "
            "print('LOADED:' + ','.join(bad) if bad else 'CLEAN')"
        ) % _HOST_SOFTWARE_DIR
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip().splitlines()[-1], "CLEAN")


if __name__ == "__main__":
    unittest.main(verbosity=2)
