"""Regression coverage for the gap flagged in docs/TODO.md (2026-10-14): "No automated
regression coverage of the actual production audio inference/preprocessing path."

This project has already paid for two severe train/production-preprocessing-divergence bugs,
both only caught by manual A/B investigation (docs/plans/audio_eval_notebook_refactor_plan.md):

  1. The live receiver (`audio_receiver_pytorch.py`) hardcoded a non-alphabetical 12-class
     label order that didn't match the alphabetical order the model was actually trained/
     evaluated with -- collapsed live accuracy to ~7.4%.
  2. The live receiver applied an active-region crop, peak renormalization, and a
     spectral-subtraction noise profile to already-clean, pre-cut audio -- none of which the
     offline evaluation path (`evaluations/evaluate_audio_classifier.py`) applies -- costing a
     further ~40-50 accuracy points combined.

Both fixes are already in place (see `audio_receiver_pytorch.py`'s `_process_loop` and
`__init__` comments dated 2026-10-14 and earlier), but nothing previously asserted either stays
fixed. These tests close that gap for the custom Conv1D CNN production path:

  - `OfflineLiveFeatureParityTests`: feeds a REAL recorded clip
    (`data/01_evaluation_samples/*.wav`) through the real live `_process_loop` buffer-fill +
    `waveform_to_spectrogram` call (chunk-by-chunk, via the receiver's real
    `_put_chunk_dropping_oldest`-fed `chunk_queue`) and through the real offline
    `evaluate_audio_classifier.pad_or_truncate` + `waveform_to_spectrogram` call, then asserts
    the two produce numerically identical spectrogram features for the same clip. If either
    path's preprocessing (crop/renorm/noise-profile/etc.) ever diverges from the other again,
    this fails loudly instead of silently.
  - `LabelOrderSanityTests`: loads the real `models/pytorch_v3` checkpoint (the checkpoint both
    `evaluate_audio_classifier.py`'s defaults and this parity test target) and asserts its
    `labels.json` is alphabetical, that the live receiver's `self.labels` for that checkpoint
    matches it verbatim, and that the receiver's hardcoded 12-class fallback (exercised when a
    checkpoint ships with no sibling `labels.json`) still matches that same alphabetical order --
    directly guarding the exact regression that previously crashed accuracy to ~7.4%.

Scope note: this covers the custom Conv1D CNN path (`audio_receiver_pytorch.py` +
`evaluations/evaluate_audio_classifier.py`), which is what this TODO item and its source
(`tests/test_audio.py`) named. The pretrained NeMo MatchboxNet path (production per
CLAUDE.md's Current State table; `evaluations/nemo_live_receiver.py`,
`main_onnx_shared_vision_nemo_audio.py`) needs its own equivalent coverage separately -- it
requires the separate NeMo conda env `test_nemo_noise_mix.py` already needs, and its live
receiver's label order comes from `model.cfg.labels` inside a `.nemo` checkpoint rather than a
sibling `labels.json`, so the fixture/assertion shape would differ. Flagged in docs/TODO.md
rather than silently assumed covered by this file.

Mocks `AudioCommandClassifier` (a frozen/converted model with baked-in weight shapes, not
freely constructible -- see `test_audio_receiver_pytorch_labels.py`'s docstring) and
`sd.InputStream` (constructing a real `AudioCommandReceiver` with no `source_file` opens a real
microphone stream, confirmed the hard way in that same file) -- matching this project's
established mocking style for this class. Nothing here opens real audio hardware or needs the
NeMo/Jetson environment.

Run (stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_audio/tests/test_production_pipeline_parity.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock
from unittest.mock import _patch  # type: ignore[attr-defined]

import numpy as np
import soundfile as sf
import torch

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_AUDIO_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_AUDIO_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_audio import audio_receiver_pytorch as arp  # noqa: E402
from ml_audio.audio_dsp import OUTPUT_SEQUENCE_LENGTH, SAMPLE_RATE, waveform_to_spectrogram  # noqa: E402
from ml_audio.evaluations.evaluate_audio_classifier import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_LABELS,
    load_labels,
    pad_or_truncate,
)

# Real recorded clips (not synthesized) used as the master live-stream evaluation sequence per
# create_master_audio.py / audio_eval_notebook_refactor_plan.md -- the obvious, already-existing
# fixtures for this rather than fabricating a fake waveform.
_FIXTURE_DIR: str = os.path.join(_ML_AUDIO_DIR, "data", "01_evaluation_samples")
_FIXTURE_CLIPS: list[str] = sorted(
    os.path.join(_FIXTURE_DIR, f) for f in os.listdir(_FIXTURE_DIR) if f.lower().endswith(".wav")
)

# Chosen so OUTPUT_SEQUENCE_LENGTH (20000 at the current SAMPLE_RATE=16000,
# MODEL_WINDOW_SECONDS=1.25) divides evenly -- lets the test feed a fixed-length clip to the
# live receiver in several equal real chunks (like real streaming) and assert the reconstructed
# rolling buffer is byte-identical to the original clip, not just test a single giant "chunk".
_STEP_SECONDS: float = 0.25
_NUM_CHUNKS: int = 5


def _mock_classifier_and_stream() -> tuple[_patch, _patch]:
    """Returns the two mock.patch.object context managers this project's established style
    uses to construct an AudioCommandReceiver without touching real hardware or real model
    weights (see this file's module docstring and test_audio_receiver_pytorch_labels.py)."""
    return (
        mock.patch.object(arp, "AudioCommandClassifier", return_value=mock.Mock()),
        mock.patch.object(arp.sd, "InputStream", return_value=mock.Mock()),
    )


class OfflineLiveFeatureParityTests(unittest.TestCase):
    """Asserts the live `_process_loop` path and the offline `evaluate_audio_classifier.py`
    path extract numerically identical spectrogram features from the same real clip."""

    def setUp(self) -> None:
        self.assertTrue(
            _FIXTURE_CLIPS, f"No .wav fixtures found under {_FIXTURE_DIR}"
        )
        self.assertEqual(
            OUTPUT_SEQUENCE_LENGTH % int(SAMPLE_RATE * _STEP_SECONDS),
            0,
            "Test setup assumption broken: OUTPUT_SEQUENCE_LENGTH must divide evenly by the "
            "chosen step size, or the chunked-feed simulation below doesn't reconstruct a clean "
            "fixed-length buffer.",
        )

    def _drive_live_process_loop(
        self, fixed_waveform: np.ndarray
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Constructs a real AudioCommandReceiver (model/mic mocked), feeds `fixed_waveform`
        through its real chunk_queue -> real `_process_loop` -> real `waveform_to_spectrogram`
        call in `_NUM_CHUNKS` equal pieces (simulating real streaming, not one giant chunk), and
        returns (final_audio_buffer, last_spec_fed_to_the_model).

        Captures the spec the production code actually computed by patching the mocked model's
        call so it's recorded -- this exercises `_process_loop`'s real code, not a
        reimplementation of it, so a future change to its preprocessing (e.g. reintroducing a
        crop/renorm/noise-profile step before the model call) would show up here as a mismatch
        against the offline path rather than being invisible.
        """
        captured_specs: list[torch.Tensor] = []

        def _fake_forward(spec: torch.Tensor) -> torch.Tensor:
            captured_specs.append(spec.clone())
            return torch.zeros(1, len(arp.LABEL_NAMES))

        classifier_patch, stream_patch = _mock_classifier_and_stream()
        with classifier_patch, stream_patch:
            receiver = arp.AudioCommandReceiver(None, step_seconds=_STEP_SECONDS)
            receiver.model = mock.Mock(side_effect=_fake_forward)
            try:
                chunks = np.split(fixed_waveform, _NUM_CHUNKS)
                for chunk in chunks:
                    receiver.chunk_queue.put(chunk)

                deadline = time.time() + 5.0
                while len(captured_specs) < _NUM_CHUNKS and time.time() < deadline:
                    time.sleep(0.01)
                self.assertEqual(
                    len(captured_specs),
                    _NUM_CHUNKS,
                    "_process_loop did not consume and process all fed chunks within 5s",
                )
                final_buffer = receiver.audio_buffer.copy()
            finally:
                receiver.stop()

        return final_buffer, captured_specs[-1]

    def _assert_parity_for_clip(self, wav_path: str) -> None:
        waveform, sr = sf.read(wav_path, dtype="float32")
        self.assertEqual(sr, SAMPLE_RATE, f"{wav_path} is not at the expected {SAMPLE_RATE} Hz")

        # Fit to the model's fixed window length using the real offline helper -- this removes
        # the (separate, already-known/accepted) alignment-at-the-edges difference between
        # "pad/truncate a pre-cut clip" and "roll a live buffer that may be mid-command", and
        # isolates exactly what this test is meant to catch: do the two paths apply the SAME
        # transform to the SAME fixed-length array.
        fixed_waveform = pad_or_truncate(waveform, OUTPUT_SEQUENCE_LENGTH)

        final_buffer, live_spec = self._drive_live_process_loop(fixed_waveform)

        # The real chunk-by-chunk roll-buffer-fill logic in _process_loop must reconstruct the
        # exact original array (order/alignment), not just something close to it.
        np.testing.assert_array_equal(
            final_buffer,
            fixed_waveform,
            err_msg=(
                f"Live receiver's reconstructed rolling buffer does not match the original "
                f"fixed-length clip for {os.path.basename(wav_path)} -- chunk-feed order/roll "
                f"logic has regressed."
            ),
        )

        offline_spec = waveform_to_spectrogram(fixed_waveform, noise_profile=None)

        self.assertTrue(
            torch.allclose(offline_spec, live_spec, atol=1e-6),
            msg=(
                f"Offline (evaluate_audio_classifier.py) and live (_process_loop) spectrogram "
                f"features diverged for {os.path.basename(wav_path)} -- one of the two "
                f"preprocessing paths has changed (crop/renorm/noise-profile/etc.) without the "
                f"other. max abs diff="
                f"{(offline_spec - live_spec).abs().max().item():.6g}"
            ),
        )

    def test_feature_parity_on_every_real_fixture_clip(self) -> None:
        for wav_path in _FIXTURE_CLIPS:
            with self.subTest(clip=os.path.basename(wav_path)):
                self._assert_parity_for_clip(wav_path)


class LabelOrderSanityTests(unittest.TestCase):
    """Guards the exact historical regression (docs/plans/audio_eval_notebook_refactor_plan.md,
    "Stage 1 Fix"): a hardcoded live-receiver label order drifting out of sync with the
    alphabetical order the model was actually trained with, which collapsed live accuracy from
    ~86% to ~7.4%."""

    def test_reference_checkpoints_labels_json_is_alphabetical(self) -> None:
        labels = load_labels(DEFAULT_LABELS)
        self.assertEqual(
            labels,
            sorted(labels),
            f"{DEFAULT_LABELS} is not alphabetically ordered -- this is the training-time "
            f"convention (scanning dataset class folders) that both the live receiver and the "
            f"offline evaluator assume.",
        )

    def test_live_receiver_loads_labels_json_verbatim_for_the_reference_checkpoint(self) -> None:
        classifier_patch, stream_patch = _mock_classifier_and_stream()
        with classifier_patch, stream_patch:
            receiver = arp.AudioCommandReceiver(DEFAULT_CHECKPOINT, step_seconds=_STEP_SECONDS)
            try:
                offline_labels = load_labels(DEFAULT_LABELS)
                self.assertEqual(
                    receiver.labels,
                    offline_labels,
                    "Live receiver's self.labels for the reference checkpoint does not match "
                    "the offline evaluator's label source (labels.json) -- the two code paths "
                    "would decode the same model's predictions as different words.",
                )
                self.assertEqual(receiver.labels, sorted(receiver.labels))
            finally:
                receiver.stop()

    def test_hardcoded_12class_fallback_still_matches_the_alphabetical_order(self) -> None:
        """Exercises the `elif num_classes == 12:` hardcoded fallback in
        AudioCommandReceiver.__init__ for real, by pointing it at a copy of the real reference
        checkpoint with its sibling labels.json deliberately withheld (simulating an older
        checkpoint shipped without one). This is the exact branch whose drift from alphabetical
        order caused the original ~7.4%-accuracy incident -- if it's ever hand-edited out of
        sync again, this fails immediately instead of only showing up as an unexplained
        live-stream accuracy drop."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_copy = os.path.join(tmp_dir, os.path.basename(DEFAULT_CHECKPOINT))
            shutil.copy(DEFAULT_CHECKPOINT, checkpoint_copy)
            self.assertFalse(
                os.path.exists(os.path.join(tmp_dir, "labels.json")),
                "Test setup error: labels.json must be absent to exercise the hardcoded fallback.",
            )

            classifier_patch, stream_patch = _mock_classifier_and_stream()
            with classifier_patch, stream_patch:
                receiver = arp.AudioCommandReceiver(checkpoint_copy, step_seconds=_STEP_SECONDS)
                try:
                    expected = load_labels(DEFAULT_LABELS)
                    self.assertEqual(
                        receiver.labels,
                        expected,
                        "Hardcoded 12-class fallback label order no longer matches the real "
                        "alphabetical training order -- this is the exact bug class that "
                        "crashed live accuracy to ~7.4%.",
                    )
                    self.assertEqual(receiver.labels, sorted(receiver.labels))
                finally:
                    receiver.stop()


if __name__ == "__main__":
    unittest.main()
