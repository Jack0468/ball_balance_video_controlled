"""Tests for `ml_audio/audio_receiver_pytorch.py`'s label-storage fix (2026-10-14).

`AudioCommandReceiver.__init__` used to do `global LABEL_NAMES` and reassign a module-level
list -- the same bug class as the documented production incident where wrong label order
crashed accuracy from ~86% to ~7.4% (`docs/plans/audio_eval_notebook_refactor_plan.md`).
Constructing two receivers against checkpoints with different class counts in one process would
have the second construction silently overwrite the label mapping the first instance's
already-running thread depends on. Fixed to `self.labels`.

`AudioCommandClassifier` bakes in a lot of fixed registered-buffer weights (a converted/frozen
model, not freely constructible with arbitrary shapes), so this mocks the model class and
`torch.load` entirely -- the fix under test is pure label-storage bookkeeping in
`AudioCommandReceiver.__init__`, not anything about real inference, matching the "duck-typed
stand-in" mocking style already used elsewhere in this project (e.g.
`test_small_cnn_grounder_cpu.py`).

Run (stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_audio/tests/test_audio_receiver_pytorch_labels.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import torch

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_AUDIO_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_AUDIO_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_audio import audio_receiver_pytorch as arp  # noqa: E402


def _fake_checkpoint(tmp_dir: str, name: str, num_classes: int, labels: list) -> str:
    """Writes a minimal fake .pth (just the one key the constructor reads the shape of) plus a
    matching labels.json next to it -- real torch.save/torch.load round trip, no real model
    weights needed since AudioCommandClassifier itself is mocked below."""
    model_path = os.path.join(tmp_dir, name)
    torch.save({"dense_bias": torch.zeros(num_classes)}, model_path)
    with open(os.path.join(tmp_dir, "labels.json"), "w") as f:
        json.dump(labels, f)
    return model_path


class LabelStorageIsPerInstanceTests(unittest.TestCase):
    """Patches `AudioCommandClassifier` AND `sd.InputStream` for every construction here --
    without a `source_file`, `__init__` opens a REAL microphone stream and starts real
    processing threads (confirmed by first running this unpatched: it genuinely opened this
    machine's mic). No test should touch real hardware."""

    def setUp(self) -> None:
        self._receivers: list = []
        self._classifier_patch = mock.patch.object(arp, "AudioCommandClassifier", return_value=mock.Mock())
        self._stream_patch = mock.patch.object(arp.sd, "InputStream", return_value=mock.Mock())
        self._classifier_patch.start()
        self._stream_patch.start()

    def tearDown(self) -> None:
        for r in self._receivers:
            r.stop()
        self._stream_patch.stop()
        self._classifier_patch.stop()

    def _make(self, model_path: object) -> "arp.AudioCommandReceiver":
        receiver = arp.AudioCommandReceiver(model_path)
        self._receivers.append(receiver)
        return receiver

    def test_two_receivers_with_different_labels_do_not_clobber_each_other(self) -> None:
        with tempfile.TemporaryDirectory() as dir_a, tempfile.TemporaryDirectory() as dir_b:
            path_a = _fake_checkpoint(dir_a, "a.pth", 3, ["x", "y", "z"])
            path_b = _fake_checkpoint(dir_b, "b.pth", 2, ["p", "q"])

            receiver_a = self._make(path_a)
            self.assertEqual(receiver_a.labels, ["x", "y", "z"])

            # Constructing a second receiver against a DIFFERENT checkpoint must not
            # silently change receiver_a's labels -- this is the exact regression the
            # global->self.labels fix closes.
            receiver_b = self._make(path_b)
            self.assertEqual(receiver_b.labels, ["p", "q"])
            self.assertEqual(receiver_a.labels, ["x", "y", "z"])

    def test_module_level_global_is_never_mutated(self) -> None:
        original_global = list(arp.LABEL_NAMES)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _fake_checkpoint(tmp_dir, "m.pth", 3, ["only", "these", "three"])
            receiver = self._make(path)
            self.assertEqual(receiver.labels, ["only", "these", "three"])
        self.assertEqual(arp.LABEL_NAMES, original_global)

    def test_no_model_path_falls_back_to_the_module_default(self) -> None:
        receiver = self._make(None)
        self.assertEqual(receiver.labels, arp.LABEL_NAMES)


if __name__ == "__main__":
    unittest.main()
