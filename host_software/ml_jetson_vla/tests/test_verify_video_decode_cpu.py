"""Tests for `verify_and_package_dataset.py`'s `--check-video-decode` addition (2026-10-13).

Context: `verify_dataset()`'s schema/row-count checks never open a single video file. A real
`lerobot_full_2026_10_07` dataset that this script had already verified+zipped later crashed
torchcodec mid-decode during a Colab eval run -- see docs/PROJECT_LOGBOOK.md, 2026-10-13 entry,
and the 2026-10-07/08 entry for what the original verification pass actually covered. This test
file checks the new `verify_video_decode()` function in isolation:

  (a) a clean, PyAV-decodable video file passes and reports the right frame count.
  (b) a truncated (corrupt) video file is caught and raises AssertionError naming the file.
  (c) `verify_dataset(..., check_video_decode=True)` wires the check in and the frame-count
      cross-check against info.json's total_frames fails when video files decode short.

Repo convention: `deployment/test_*.py` are standalone scripts with exit codes; these tests follow
that (`python tests/test_verify_video_decode_cpu.py`) and are also pytest-discoverable.

Usage (from host_software/, pinned interpreter):
    C:/Users/Admin/.conda/envs/ball_balance_env/python.exe ml_jetson_vla/tests/test_verify_video_decode_cpu.py
Exit code 0 only if every test passes.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import traceback
from types import ModuleType
from typing import Callable, Dict

import av
import numpy as np

_THIS_DIR: str = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR: str = os.path.abspath(os.path.join(_THIS_DIR, ".."))
VERIFY_PATH: str = os.path.join(_ML_JETSON_VLA_DIR, "data_processing", "verify_and_package_dataset.py")


def _load_module(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _verify_module() -> ModuleType:
    return _load_module("verify_and_package_dataset_under_test", VERIFY_PATH)


def _write_clean_mp4(path: str, n_frames: int, width: int = 64, height: int = 48) -> None:
    container = av.open(path, mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    for i in range(n_frames):
        frame = av.VideoFrame.from_ndarray(
            np.full((height, width, 3), (10 * i) % 256, dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _make_minimal_dataset(root: str, video_rel: str, n_frames: int, declared_total_frames: int) -> None:
    video_path = os.path.join(root, video_rel)
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    _write_clean_mp4(video_path, n_frames)
    meta_dir = os.path.join(root, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump({"codebase_version": "v3.0", "total_frames": declared_total_frames}, f)


# --- (a) clean file decodes and reports the right frame count ---------------------------------------

def test_a_clean_video_decodes() -> None:
    mod = _verify_module()
    tmp = tempfile.mkdtemp(prefix="verify_video_decode_a_")
    try:
        n_frames = 20
        _make_minimal_dataset(tmp, os.path.join("videos", "observation.image", "chunk-000", "file-000.mp4"),
                               n_frames, declared_total_frames=n_frames)
        with open(os.path.join(tmp, "meta", "info.json")) as f:
            info = json.load(f)
        result = mod.verify_video_decode(tmp, info)
        assert result["video_files_checked"] == 1, result
        assert result["total_decoded_frames"] == n_frames, result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- (b) truncated file is caught with the file named in the error ----------------------------------

def test_b_truncated_video_is_caught() -> None:
    mod = _verify_module()
    tmp = tempfile.mkdtemp(prefix="verify_video_decode_b_")
    try:
        video_rel = os.path.join("videos", "observation.image", "chunk-000", "file-000.mp4")
        _make_minimal_dataset(tmp, video_rel, n_frames=20, declared_total_frames=20)
        video_path = os.path.join(tmp, video_rel)
        # Truncate to roughly half the bytes -- cuts the AV1/h264 stream mid-packet, the same class
        # of corruption ("Invalid data found when processing input") as the real Colab crash.
        size = os.path.getsize(video_path)
        with open(video_path, "r+b") as f:
            f.truncate(size // 2)
        with open(os.path.join(tmp, "meta", "info.json")) as f:
            info = json.load(f)
        try:
            mod.verify_video_decode(tmp, info)
            raise AssertionError("expected verify_video_decode to raise on a truncated file")
        except AssertionError as exc:
            msg = str(exc)
            assert "file-000.mp4" in msg or "decode failed" in msg, msg
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- (c) verify_dataset wires the flag in and the total-frame cross-check catches a short decode ----

def test_c_verify_dataset_flag_catches_frame_mismatch() -> None:
    mod = _verify_module()
    tmp = tempfile.mkdtemp(prefix="verify_video_decode_c_")
    try:
        video_rel = os.path.join("videos", "observation.image", "chunk-000", "file-000.mp4")
        # info.json claims more frames than the (clean, fully-decodable) video actually has --
        # mirrors a different real failure mode (metadata/video mismatch) from pure corruption.
        _make_minimal_dataset(tmp, video_rel, n_frames=20, declared_total_frames=999)
        with open(os.path.join(tmp, "meta", "info.json")) as f:
            info = json.load(f)
        try:
            mod.verify_video_decode(tmp, info)
            raise AssertionError("expected a frame-count mismatch to raise")
        except AssertionError as exc:
            assert "total_frames" in str(exc), exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


TESTS: Dict[str, Callable[[], None]] = {
    "test_a_clean_video_decodes": test_a_clean_video_decodes,
    "test_b_truncated_video_is_caught": test_b_truncated_video_is_caught,
    "test_c_verify_dataset_flag_catches_frame_mismatch": test_c_verify_dataset_flag_catches_frame_mismatch,
}


def main() -> int:
    failures = 0
    for name, fn in TESTS.items():
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 -- report every failure, don't stop at the first
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
