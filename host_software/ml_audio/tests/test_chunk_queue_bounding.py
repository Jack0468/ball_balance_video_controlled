"""Tests for the chunk_queue bounding/drop-oldest fix (2026-10-14).

Both `AudioCommandReceiver` (audio_receiver_pytorch.py) and `NemoAudioCommandReceiver`
(evaluations/nemo_live_receiver.py) had an unbounded `chunk_queue` fed by a real-time audio
callback with a single, blocking-inference consumer -- if inference ever averaged slower than
`step_seconds`, chunks accumulated silently forever with no backpressure or staleness detection.
Fixed with a bounded queue (`CHUNK_QUEUE_MAXSIZE=5`) and a `_put_chunk_dropping_oldest()` helper:
when full, drop the oldest buffered chunk rather than blocking the real-time callback or growing
unboundedly.

Tests the fix logic directly against each class's real `_put_chunk_dropping_oldest` method and
real `queue.Queue`, without needing a model/microphone -- these are pure queue-mechanics checks.

Run (stdlib unittest, no new dependency):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe host_software/ml_audio/tests/test_chunk_queue_bounding.py
"""

from __future__ import annotations

import os
import queue
import sys
import unittest
from unittest import mock

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_AUDIO_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR: str = os.path.abspath(os.path.join(_ML_AUDIO_DIR, ".."))
if _HOST_SOFTWARE_DIR not in sys.path:
    sys.path.append(_HOST_SOFTWARE_DIR)

from ml_audio import audio_receiver_pytorch as arp  # noqa: E402
from ml_audio.evaluations import nemo_live_receiver as nlr  # noqa: E402


class _Bare:
    """A bare object carrying just the attributes `_put_chunk_dropping_oldest` reads/writes --
    avoids constructing a full receiver (real model/mic) for a pure queue-mechanics test."""

    def __init__(self, maxsize: int) -> None:
        self.chunk_queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self.dropped_chunk_count = 0
        self.latest_inference_time_ms = 0.0
        self.step_seconds = 0.2


class AudioReceiverPytorchQueueBoundingTests(unittest.TestCase):
    def test_queue_is_bounded_at_the_documented_maxsize(self) -> None:
        self.assertEqual(arp.CHUNK_QUEUE_MAXSIZE, 5)

    def test_put_under_capacity_never_drops(self) -> None:
        receiver = _Bare(maxsize=arp.CHUNK_QUEUE_MAXSIZE)
        for i in range(arp.CHUNK_QUEUE_MAXSIZE):
            arp.AudioCommandReceiver._put_chunk_dropping_oldest(receiver, i)
        self.assertEqual(receiver.dropped_chunk_count, 0)
        self.assertEqual(receiver.chunk_queue.qsize(), arp.CHUNK_QUEUE_MAXSIZE)

    def test_put_over_capacity_drops_oldest_not_newest(self) -> None:
        receiver = _Bare(maxsize=3)
        for i in range(3):
            arp.AudioCommandReceiver._put_chunk_dropping_oldest(receiver, i)
        # Queue is now [0, 1, 2]. One more push must drop 0 (oldest), keep [1, 2, 3].
        arp.AudioCommandReceiver._put_chunk_dropping_oldest(receiver, 3)
        self.assertEqual(receiver.dropped_chunk_count, 1)
        remaining = [receiver.chunk_queue.get_nowait() for _ in range(3)]
        self.assertEqual(remaining, [1, 2, 3])

    def test_warning_is_rate_limited_not_logged_every_drop(self) -> None:
        receiver = _Bare(maxsize=1)
        arp.AudioCommandReceiver._put_chunk_dropping_oldest(receiver, 0)
        with mock.patch("builtins.print") as mock_print:
            for i in range(1, 25):
                arp.AudioCommandReceiver._put_chunk_dropping_oldest(receiver, i)
        # 24 pushes past the first -> 24 drops; logged on drop counts 1, 11, 21 (every 10th).
        self.assertEqual(receiver.dropped_chunk_count, 24)
        warning_calls = [c for c in mock_print.call_args_list if "WARNING" in str(c)]
        self.assertEqual(len(warning_calls), 3)


class NemoLiveReceiverQueueBoundingTests(unittest.TestCase):
    def test_queue_is_bounded_at_the_documented_maxsize(self) -> None:
        self.assertEqual(nlr.CHUNK_QUEUE_MAXSIZE, 5)

    def test_put_over_capacity_drops_oldest_not_newest(self) -> None:
        receiver = _Bare(maxsize=3)
        for i in range(3):
            nlr.NemoAudioCommandReceiver._put_chunk_dropping_oldest(receiver, i)
        nlr.NemoAudioCommandReceiver._put_chunk_dropping_oldest(receiver, 3)
        self.assertEqual(receiver.dropped_chunk_count, 1)
        remaining = [receiver.chunk_queue.get_nowait() for _ in range(3)]
        self.assertEqual(remaining, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
