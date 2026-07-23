import struct
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from combined_audio_video_controller import build_error_payload, resolve_target_coordinates


def test_resolve_target_coordinates_for_colour_select():
    target = resolve_target_coordinates({"mode": "colour_select", "target_colour": "red"})
    assert target == (52.75, 18.0)


def test_build_error_payload_converts_to_signed_ints():
    payload = build_error_payload(100.0, 50.0, 10.0, 20.0)
    marker, err_x_int, err_y_int = struct.unpack("<chh", payload)

    assert marker == ord("<")
    assert err_x_int == 90
    assert err_y_int == 30
