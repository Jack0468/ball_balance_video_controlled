"""Tests for `ml_multimodal/core/dataset.py`'s `VLADataset` (2026-10-14).

Covers the real bug found/fixed this session: the vocabulary used to live in space-separated
form ("go red", "go blue", ...) that never matched any real logged `audio_command` string except
"hold" -- so language conditioning silently resolved to the "hold" index for every real sample.
Also covers the image-load fallback path, narrowed from a bare `except:` to
`(FileNotFoundError, OSError)` with counting/logging.

Run (stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_multimodal/tests/test_dataset.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_MULTIMODAL_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_MULTIMODAL_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_multimodal.core.dataset import VLADataset, VOCAB  # noqa: E402


def _write_json(rows: list) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(rows, f)
    return path


class VocabMatchesRealCommandStringsTests(unittest.TestCase):
    """The real, recognized vocabulary (core.minimal_vlm_policy.COLOR_COMMANDS's keys +
    DIRECTIONAL_COMMANDS) must all be real vocab keys distinct from "hold"'s index -- this is
    the exact set confirmed against real production telemetry this session."""

    REAL_COMMANDS = (
        "go_red", "go_green", "go_yellow", "go_black",
        "forward", "backward", "left", "right", "stop",
    )

    def test_every_real_command_is_a_distinct_vocab_entry(self) -> None:
        hold_idx = VOCAB["hold"]
        seen_indices = set()
        for cmd in self.REAL_COMMANDS:
            self.assertIn(cmd, VOCAB, f"{cmd!r} is a real command but missing from VOCAB")
            idx = VOCAB[cmd]
            self.assertNotEqual(idx, hold_idx, f"{cmd!r} must not silently alias 'hold'")
            self.assertNotIn(idx, seen_indices, f"{cmd!r} reuses an already-assigned index")
            seen_indices.add(idx)

    def test_old_space_separated_keys_are_gone(self) -> None:
        # The old vocab's keys never matched anything real -- confirm they're not silently
        # still present (e.g. kept as unused aliases that could mask a future regression).
        for stale_key in ("go red", "go blue", "go green", "go yellow"):
            self.assertNotIn(stale_key, VOCAB)


class GetItemCommandLookupTests(unittest.TestCase):
    def test_real_command_resolves_to_its_real_index_not_hold(self) -> None:
        path = _write_json([
            {"image_path": "/does/not/exist.png", "audio_command": "go_red",
             "state_x": 0, "state_y": 0, "action_theta_a": 0, "action_theta_b": 0, "action_theta_c": 0},
        ])
        try:
            ds = VLADataset(path)
            _img, cmd_idx, _state, _action = ds[0]
            self.assertEqual(cmd_idx, VOCAB["go_red"])
            self.assertNotEqual(cmd_idx, VOCAB["hold"])
        finally:
            os.remove(path)

    def test_missing_audio_command_defaults_to_hold(self) -> None:
        path = _write_json([
            {"image_path": "/does/not/exist.png",
             "state_x": 0, "state_y": 0, "action_theta_a": 0, "action_theta_b": 0, "action_theta_c": 0},
        ])
        try:
            ds = VLADataset(path)
            _img, cmd_idx, _state, _action = ds[0]
            self.assertEqual(cmd_idx, VOCAB["hold"])
        finally:
            os.remove(path)


class ImageFallbackTests(unittest.TestCase):
    def test_missing_image_file_falls_back_and_counts(self) -> None:
        path = _write_json([
            {"image_path": "/definitely/does/not/exist_12345.png", "audio_command": "hold",
             "state_x": 0, "state_y": 0, "action_theta_a": 0, "action_theta_b": 0, "action_theta_c": 0},
        ])
        try:
            ds = VLADataset(path)
            self.assertEqual(ds._image_fallback_count, 0)
            img, _cmd_idx, _state, _action = ds[0]
            self.assertEqual(img.size, (224, 224))
            self.assertEqual(ds._image_fallback_count, 1)
            ds[0]
            self.assertEqual(ds._image_fallback_count, 2)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
