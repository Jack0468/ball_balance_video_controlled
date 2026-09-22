"""Model-agnostic Arm-2 minimal-baseline policy wrapper (`docs/ARM2_MINIMAL_BASELINE_SCOPE.md`
S2/S6b, extended for multiple candidates by
`docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md`). Implements the "honest minimum" path that
doc's S3 calls for: prompt a general-purpose VLM with the image + current instruction, parse
its text output into a structured target, return it -- **no custom-trained heads, no
fine-tuning**, ever (that is `core/qwen_multihead_policy.py`'s separate, parked
specialization-track job -- see that module and the scope doc's S5 table).

Satisfies `core/policy_interface.py`'s `Policy` protocol (`act(image, instruction, state) ->
PolicyCommand`), same shape `JetsonExpertPolicy` (`runtime/run_jetson_standalone.py`) already
implements, so this can be swapped into the same runtime loop -- the model-agnostic point is
`MinimalVLMPolicy(backend=<any VLMBackend from core/vlm_backends.py>)`: swapping the
underlying model is a constructor argument, not new code per model.

## Prompt / parse contract (this module's own design choice, documented here)

Ask every backend for the SAME thing regardless of which model sits underneath: point to
where the current instruction's target is, as a single (x, y) pixel location in the image
handed to it. This was chosen over asking directly for `theta_a/b/c` motor angles because:
  - A general-purpose VLM has been trained on zero data about this platform's motor geometry
    (RLControl.cpp's 9-dim contract, MotorControl.h's IK) -- asking it to emit angles would be
    asking it to blindly guess a mapping it has no basis for, which is a materially different
    (and less honest) ask than "point to the thing you can actually see and reason about,"
    which is exactly the visual grounding capability general VLMs ARE trained for.
  - It is directly, cheaply scoreable against this project's real logged data (target_x/y in
    `telemetry.csv`, mm) without inventing a new ground-truth signal -- see
    `deployment/score_minimal_baseline_offline.py`.
  - The scope doc's S3 "honest minimum" explicitly allows EITHER `theta_a/b/c` OR a target
    position -- this module picks the position, and documents why, rather than silently
    picking one without saying so.

`PolicyCommand.target_x_mm/target_y_mm` (the Policy protocol's required fields) are only
populated when a `pixel_to_mm` callable is supplied at construction time (e.g. from a real
ArUco homography computed the same way `run_jetson_standalone.py`'s `JetsonExpertPolicy`
already does, or `ml_vision.core.coordinate_math.HomographyProjector`, reused read-only per
`docs/BOOTSTRAP_MODEL_COMPARISON_PLAN.md`'s Metric A precedent -- see that doc's S2.1). When
no calibration is supplied (the common case for a candidate that hasn't been paired with a
real calibrated rig yet), `target_x_mm`/`target_y_mm` are `float("nan")` rather than a silently
fabricated number, and the raw pixel point is always available in `debug["target_point_px"]`
regardless. **This is a deliberate deviation from `PolicyCommand`'s field being framed as
"always populated" in `policy_interface.py`'s docstring** -- NaN-on-no-calibration is judged
more honest than either raising (which would make the wrapper unusable for the offline
parse/latency smoke tests this task also needs) or fabricating a value.

## Directional/hold/stop commands -- documented weak fallback, not a real capability claim

Color-target commands (`go_red`/`go_green`/`go_yellow`/`go_black`) have a well-defined visual
target (a marker) -- this is what the prompt/parser above is actually built and scored for.
Directional commands (`forward/left/right/backward/hold/stop`) do NOT have a well-defined
point-in-image target for a general VLM with no notion of the platform's control convention --
this is the exact same reason `docs/BOOTSTRAP_MODEL_COMPARISON_PLAN.md` S2.4 explicitly
descoped them from Metric A rather than inventing a scoring method. This module mirrors that
precedent for the live policy path too: for a directional/hold/stop instruction, the prompt
instead asks the model to point to "the ball's current position" (a well-posed grounding
task), and the parsed result is used as a hold-in-place target -- `debug["directional_fallback"]
= True` marks every such frame so a reader never mistakes this for a working "the VLM
understood spatial directions" result. It is a documented stand-in, not a claim.

## Prompt variants (2026-09-18 follow-up to `ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md`)

The n=4 offline smoke score (that doc's S7.2) showed near-chance zero-shot localization
(mean error ~186-190mm on a 187.5x142mm platform -- roughly the platform's own half-width).
Before concluding anything about Arm 2's viability, this module now offers a second prompt
template (`PROMPT_TEMPLATE_ORIENTED`, selected via `prompt_variant="oriented"`) that adds real,
verified platform/task context (dimensions, coordinate convention, ArUco-vs-target
disambiguation, ball-and-plate task framing -- see the "Verified facts" comment block above
`PROMPT_TEMPLATE`) to test whether richer grounding context helps, as a real A/B comparison
against the original bare `PROMPT_TEMPLATE` ("baseline"), not a replacement of it. The output
contract (a single pixel point, same JSON shape, same parser, homography-to-mm conversion done
entirely outside the model) is unchanged by either variant.

2026-09-19 adds a third variant, `oriented_aruco` (A/B/C): the oriented prompt plus the real
ArUco marker layout (IDs, mm positions, size -- generated from
`hardware/platform_templates/ground_truth_manifest.json`, never typed in) and a statement that the
markers are visible landmarks. **Prompt text only**: the model still receives the raw, un-warped
camera frame; the ArUco homography stays Track 1's preprocessing and is used only by the scorer to
convert the model's predicted pixel to mm. See the "Verified facts" block for the corrected mm-frame
conventions and the 180-degree camera-orientation finding that this variant's text encodes.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
import time
from typing import Callable, Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ML_JETSON_VLA_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_HOST_SOFTWARE_DIR = os.path.abspath(os.path.join(_ML_JETSON_VLA_DIR, ".."))
_REPO_ROOT_DIR = os.path.abspath(os.path.join(_HOST_SOFTWARE_DIR, ".."))
for _p in (_HOST_SOFTWARE_DIR, _REPO_ROOT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from ml_jetson_vla.core.policy_interface import Policy, PolicyCommand  # noqa: E402
from ml_jetson_vla.core.vlm_backends import BackendOutput, VLMBackend  # noqa: E402

# --- Instruction -> prompt vocabulary -------------------------------------------------
# Matches the real recognized vocabulary this project's audio classifier / state machine
# use (see `src/state_machine.py`, `docs/BOOTSTRAP_MODEL_COMPARISON_PLAN.md` S2.1's own
# "the only commands with a well-defined target marker location" list) -- not invented here.
COLOR_COMMANDS = {
    "go_red": "red marker",
    "go_green": "green marker",
    "go_yellow": "yellow marker",
    "go_black": "black marker",
}
DIRECTIONAL_COMMANDS = {"forward", "left", "right", "backward", "hold", "stop"}

# --- Verified facts used to ground the "oriented" prompt variant below -----------------
# Every number/claim here was checked against a real source this session (2026-09-18), not
# invented -- see the ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md follow-up task for the audit.
#   - Platform physical size: CLAUDE.md "System Constraints" -- 187.5mm x 142.0mm is the
#     current source of truth (explicitly NOT 182.5x147.0, an old stale value the same doc
#     calls out).
#   - mm coordinate convention -- **CORRECTED 2026-09-19, the earlier version of this note was
#     WRONG about half of it**. Two different mm frames exist and must not be conflated:
#       (1) the MANIFEST / homography frame: hardware/platform_templates/ground_truth_manifest.json's
#           "coordinate_convention" -- "Y=0 at top-left, Y increases downward", origin at the
#           printed sheet's top-left corner. `estimate_homography_from_aruco()`'s mm side is this
#           frame. That half of the old note was right.
#       (2) the TELEMETRY / touch frame: telemetry.csv's `target_x`/`target_y` (and touch_x/
#           touch_y) are CENTER-origin -- (0,0) is the platform centre -- with
#               target_x = W/2 - x_manifest_mm,   target_y = y_manifest_mm - H/2
#           (x mirrored, y not). The old note claimed target_x/y were top-left-origin because
#           frame 1 of session_jetson_track4_20260915_151627 has target=(0.42, 29.61) and
#           "small positive values sit near an edge"; that was wrong -- (0.42, 29.61) is 29.6mm
#           from the CENTRE, which is where the green marker is. The repo's own
#           auto_label_shared_vision.py already documents the same relationship for touch_x/y
#           (`ball_x_mm = W/2 - touch_x`, `ball_y_mm = H/2 + touch_y`, 2026-08-12 point-
#           reflection fix), and it was re-verified empirically 2026-09-19: HSV-detected marker
#           centroids projected through the per-frame ArUco homography match
#           (W/2 - x, y - H/2) to ~1mm median (0.3-7mm) over 25 frames across all 10 sessions,
#           versus 78-170mm if compared as top-left mm. The scorer
#           (`deployment/score_minimal_baseline_offline.py`) compared homography-mm predictions
#           directly against telemetry target_x/y until 2026-09-19, so every `error_mm`/`hit`
#           it reported before then is invalid -- see `touch_frame_to_manifest_mm()` there and
#           the erratum in docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md.
#   - Camera orientation (added 2026-09-19): in every Track 4 session the raw camera image is
#     the platform rotated 180 degrees relative to the manifest frame (manifest ID 0, the
#     "top-left-corner" marker, appears at the image's BOTTOM-RIGHT; H[0,0] and H[1,1] are
#     negative in all 60 sampled frames; mean marker pixel positions verified over those 60
#     frames, ~11-15px std). The `oriented_aruco` prompt below states this explicitly, since
#     giving a model manifest-frame roles ("top-left-corner") for markers that appear at the
#     opposite corner of the image would be actively misleading.
#   - ArUco markers' role: auto_label_shared_vision.py's own load_manifest_full() docstring --
#     "aruco_markers -- ArUco fiducial markers used for homography computation" vs.
#     "features -- Colored target markers" are two distinct, separately-schemaed lists in the
#     manifest. ArUco markers are fixed calibration reference points, never the ball or a
#     target.
#   - Ball physical size/mass: CHECKED, genuinely undocumented. Searched
#     hardware/platform_templates/ (manifest + LaTeX sheets), docs/PROJECT_LOGBOOK.md, and
#     CLAUDE.md for a ball diameter/mass spec -- none found anywhere in the repo. Not invented
#     here; the oriented prompt below describes the ball only qualitatively ("a small ball"),
#     never with a fabricated number.
PLATFORM_WIDTH_MM = 187.5
PLATFORM_HEIGHT_MM = 142.0

PROMPT_TEMPLATE = (
    "You are looking straight down at a small 3-DOF ball-balancing platform. "
    "Find the {target_label} in this image. "
    "Respond with ONLY a single JSON object and nothing else, in exactly this form: "
    '{{"target_point_xy": [x, y]}} '
    "where x and y are pixel coordinates in THIS image (x in [0, {width}], y in [0, {height}])."
)

# "Oriented" variant -- adds real, verified task/platform context (see block above) to help
# the model's visual grounding/attention. Deliberately does NOT change the ask or the output
# contract: still a single pixel point, in the same JSON shape, parsed by the exact same
# parse_minimal_baseline_output() -- the homography-to-mm conversion still happens entirely
# outside the model, unchanged. Also folds in a brief "why" framing (the answer steers a real
# physical platform) rather than testing that as a separate third prompt variant -- a
# deliberate scope decision given this candidate's real CPU generate() latency (~2-3 min/call,
# see the multi-candidate doc's S7.1/7.2), where a third full variant would roughly triple an
# already many-hour scoring run for a secondary hypothesis; the module keeps both the doc note
# and the option to split it out later if the oriented variant's result makes that worth doing.
PROMPT_TEMPLATE_ORIENTED = (
    "You are looking straight down through a fixed overhead camera at a small 3-DOF "
    "ball-balancing robot platform -- a 'ball-and-plate' control problem. The platform surface "
    "is a rectangle {platform_w_mm:.1f}mm wide by {platform_h_mm:.1f}mm deep. It tilts on 3 "
    "motors, and a small ball rolls freely across the surface under gravity as it tilts -- the "
    "ball is the only object on the platform that moves. Several small colored circular "
    "markers are fixed (non-moving) possible targets on the platform surface. You may also see "
    "small black-and-white square patterns on the platform -- those are fixed ArUco camera-"
    "calibration reference markers, not the ball and not a target; ignore them when looking "
    "for the {target_label}. "
    "Find the {target_label} in this image. "
    "Respond with ONLY a single JSON object and nothing else, in exactly this form: "
    '{{"target_point_xy": [x, y]}} '
    "where x and y are pixel coordinates in THIS image (x in [0, {width}], y in [0, {height}]), "
    "with (0, 0) at the top-left corner of the image and y increasing downward. "
    "Your answer will be used directly to steer the real physical platform, so precise, "
    "careful localization matters."
)

# --- "oriented_aruco" variant (2026-09-19): oriented prompt + real ArUco marker knowledge ---
# Prompt TEXT only. The model is still handed the raw, un-warped camera frame -- the ArUco
# homography is Track 1's preprocessing (`ARCHITECTURE.md`'s confound-avoidance rationale,
# `ARM2_MINIMAL_BASELINE_SCOPE.md`) and never touches Arm 2's model input; it is used only in
# the scorer, to convert the model's predicted pixel into mm. Output contract unchanged.
#
# Every number in the text comes from ground_truth_manifest.json at import time (marker IDs,
# centres, size, platform size) -- nothing is typed in here, so it cannot drift from the
# manifest. Two facts are NOT in the manifest and are stated as what they are: (a) the manifest's
# `features` list is EMPTY, i.e. it holds no colored-marker positions (they are deliberately not
# put in the prompt either -- that would hand the model the answer), and (b) the 180-degree image
# rotation described above, verified from the data, not the manifest.
_MANIFEST_RELPATH = os.path.join("hardware", "platform_templates", "ground_truth_manifest.json")

# Role in the manifest frame -> where that marker appears in the raw camera image, under the
# verified 180-degree rotation (see "Camera orientation" note above). An unknown role raises
# rather than guessing.
_IMAGE_PLACEMENT_UNDER_180_ROTATION = {
    "top-left-corner": "bottom-right corner of the image",
    "top-right-corner": "bottom-left corner of the image",
    "bottom-right-corner": "top-left corner of the image",
    "bottom-left-corner": "top-right corner of the image",
    "mid-left-edge": "middle of the image's right side",
    "mid-right-edge": "middle of the image's left side",
}


def load_aruco_manifest(manifest_path: Optional[str] = None) -> dict:
    """Loads ground_truth_manifest.json. Search order: explicit arg, $VRI_GROUND_TRUTH_MANIFEST,
    <repo root>/hardware/platform_templates/ground_truth_manifest.json."""
    candidates = [manifest_path, os.environ.get("VRI_GROUND_TRUTH_MANIFEST"),
                  os.path.join(_REPO_ROOT_DIR, _MANIFEST_RELPATH)]
    for cand in candidates:
        if cand and os.path.exists(cand):
            with open(cand, "r", encoding="utf-8") as fh:
                return json.load(fh)
    raise FileNotFoundError(
        f"ground_truth_manifest.json not found (tried: {[c for c in candidates if c]}). The "
        f"'oriented_aruco' prompt is generated from it and refuses to invent marker numbers."
    )


def _fmt_mm(v: float) -> str:
    return f"{float(v):.1f}"


def build_aruco_prompt_block(manifest: dict) -> str:
    """Renders the ArUco paragraph from a loaded manifest. Contains no str.format() braces
    (it is spliced into a template that IS later .format()-ed)."""
    markers = sorted(manifest["aruco_markers"], key=lambda m: int(m["id"]))
    if not markers:
        raise ValueError("manifest has no aruco_markers")
    sizes = sorted({float(m["size_mm"]) for m in markers})
    size_txt = f"{_fmt_mm(sizes[0])}mm" if len(sizes) == 1 else \
        f"{_fmt_mm(sizes[0])}-{_fmt_mm(sizes[-1])}mm"
    w_mm, h_mm = float(manifest["platform_width_mm"]), float(manifest["platform_height_mm"])
    entries = []
    for m in markers:
        role = m["role"]
        if role not in _IMAGE_PLACEMENT_UNDER_180_ROTATION:
            raise ValueError(f"no image-placement rule for manifest role {role!r}; refusing to guess")
        cx, cy = m["center_mm"]
        entries.append(
            f"ID {int(m['id'])} ({role.replace('-', ' ')}) centred at ({_fmt_mm(cx)}, {_fmt_mm(cy)})mm, "
            f"which appears at the {_IMAGE_PLACEMENT_UNDER_180_ROTATION[role]}"
        )
    return (
        f"The ArUco markers are {len(markers)} square black-and-white fiducial patterns, each about "
        f"{size_txt} across, printed at known fixed positions on the platform, and they are visible "
        f"in this image -- use them as spatial reference landmarks to reason about where things sit "
        f"on the platform. Their positions are given in millimetres in the platform's own coordinate "
        f"system: origin at the platform's top-left corner as printed, x increasing to the right, y "
        f"increasing downward, platform {_fmt_mm(w_mm)}mm wide by {_fmt_mm(h_mm)}mm tall. "
        f"{'; '.join(entries)}. "
        f"Note the camera is mounted so the platform appears rotated by 180 degrees in the image "
        f"relative to that coordinate system, which is why the placements above are the opposite "
        f"corners from the printed roles. "
    )


_ORIENTED_IGNORE_SENTENCE = "ignore them when looking for the {target_label}. "


def build_oriented_aruco_template(manifest: Optional[dict] = None) -> str:
    """PROMPT_TEMPLATE_ORIENTED, with its 'ignore the ArUco markers' sentence replaced by
    'never answer with one of them' + the manifest-derived ArUco paragraph. Derived from the
    oriented template by substitution (asserted to apply), so it cannot silently diverge from it."""
    if _ORIENTED_IGNORE_SENTENCE not in PROMPT_TEMPLATE_ORIENTED:
        raise AssertionError("PROMPT_TEMPLATE_ORIENTED changed; update _ORIENTED_IGNORE_SENTENCE")
    block = build_aruco_prompt_block(manifest if manifest is not None else load_aruco_manifest())
    return PROMPT_TEMPLATE_ORIENTED.replace(
        _ORIENTED_IGNORE_SENTENCE, "never answer with one of them. " + block, 1
    )


# Name -> template string. Selectable via MinimalVLMPolicy(prompt_variant=...) or
# score_minimal_baseline_offline.py's --prompt-variants flag. "baseline" is the original,
# unchanged bare prompt (kept as a real A/B comparison arm, not replaced); "oriented" and
# "oriented_aruco" are A/B/C companions -- see docs/ARM2_MINIMAL_BASELINE_MULTI_CANDIDATE.md.
PROMPT_VARIANTS = {
    "baseline": PROMPT_TEMPLATE,
    "oriented": PROMPT_TEMPLATE_ORIENTED,
}
try:
    PROMPT_TEMPLATE_ORIENTED_ARUCO = build_oriented_aruco_template()
    PROMPT_VARIANTS["oriented_aruco"] = PROMPT_TEMPLATE_ORIENTED_ARUCO
    ARUCO_PROMPT_UNAVAILABLE_REASON: Optional[str] = None
except FileNotFoundError as _exc:  # e.g. a deployment bundle without hardware/ -- degrade, don't crash import
    PROMPT_TEMPLATE_ORIENTED_ARUCO = ""
    ARUCO_PROMPT_UNAVAILABLE_REASON = str(_exc)


def build_prompt(
    instruction: Optional[str], image_w: int, image_h: int, prompt_variant: str = "baseline"
) -> tuple:
    """Returns (prompt_text, target_label, is_directional_fallback). `target_label` is the
    short noun phrase backends with a native label-taking API (`Moondream2Backend`) use
    directly instead of the free-text prompt. `prompt_variant` selects a template from
    `PROMPT_VARIANTS` -- every template is formatted with the same superset of kwargs
    (str.format() silently ignores keys a given template doesn't reference), so adding a new
    variant never requires touching this function."""
    if prompt_variant not in PROMPT_VARIANTS:
        extra = (f" ('oriented_aruco' unavailable: {ARUCO_PROMPT_UNAVAILABLE_REASON})"
                 if prompt_variant == "oriented_aruco" and ARUCO_PROMPT_UNAVAILABLE_REASON else "")
        raise ValueError(
            f"Unknown prompt_variant={prompt_variant!r}; choices are {list(PROMPT_VARIANTS)}{extra}"
        )
    template = PROMPT_VARIANTS[prompt_variant]
    format_kwargs = dict(
        width=image_w,
        height=image_h,
        platform_w_mm=PLATFORM_WIDTH_MM,
        platform_h_mm=PLATFORM_HEIGHT_MM,
    )
    if instruction in COLOR_COMMANDS:
        target_label = COLOR_COMMANDS[instruction]
        return (
            template.format(target_label=target_label, **format_kwargs),
            target_label,
            False,
        )
    # Directional/hold/stop, unrecognized, or None -- documented weak fallback, see module
    # docstring. Never silently treated as a color command.
    target_label = "ball"
    return (
        template.format(target_label=target_label, **format_kwargs),
        target_label,
        True,
    )


# --- Parsing -----------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)
_QWEN_POINT_LIST_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)
_PALIGEMMA_LOC_RE = re.compile(r"<loc(\d{4})>")


@dataclasses.dataclass
class ParseResult:
    ok: bool
    x_px: Optional[float] = None
    y_px: Optional[float] = None
    reason: str = ""  # which convention matched, or why parsing failed
    raw_text: str = ""


def parse_minimal_baseline_output(
    raw_text: str, image_w: int, image_h: int, native_points: Optional[list] = None
) -> ParseResult:
    """Tries, in order, every output convention this task's candidates are documented (not
    guessed) to actually use. Returns `ok=False` rather than a fabricated point when nothing
    matches -- callers MUST check `ok` before using `x_px`/`y_px`. This is a deliberate design
    choice per this task's "decide and document what happens when it doesn't parse cleanly"
    instruction, not an oversight.

    1. `native_points` (set only by a backend whose real API already returns structured
       points, e.g. `Moondream2Backend.point()`) -- used as-is, first entry, no text parsing
       needed at all.
    2. This module's own asked-for contract: `{"target_point_xy": [x, y]}`, optionally wrapped
       in prose or a markdown code fence -- the first `{...}` substring is extracted and
       json-parsed.
    3. Qwen2.5-VL's native grounding convention (confirmed real via WebSearch, 2026-09-18):
       a JSON list of `{"point_2d": [x, y], ...}` or `{"bbox_2d": [x1,y1,x2,y2], ...}` objects
       -- first entry's point, or bbox center, used.
    4. PaliGemma2's native convention (confirmed real via WebSearch, 2026-09-18): four
       `<locNNNN>` tokens in `<locY1><locX1><locY2><locX2>` order, each normalized 0-1023
       against a 1024-cell grid -- denormalized against `image_w`/`image_h`, bbox center used.
       A count that is not a positive multiple of 4 = not this convention, falls through; a
       larger multiple of 4 (several boxes) uses the first box (2026-09-19).
    5. Anything else: `ok=False`, `reason` explains what was tried, `raw_text` preserved
       verbatim for debugging -- never a guessed point.
    """
    if native_points:
        x, y = native_points[0]
        return ParseResult(ok=True, x_px=float(x), y_px=float(y), reason="native_points", raw_text=raw_text)

    stripped = raw_text.strip()

    # 2. Our own asked-for contract.
    match = _JSON_OBJECT_RE.search(stripped)
    if match:
        try:
            obj = json.loads(match.group(0))
            pt = obj.get("target_point_xy")
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                return ParseResult(
                    ok=True, x_px=float(pt[0]), y_px=float(pt[1]),
                    reason="asked_json_contract", raw_text=raw_text,
                )
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            pass

    # 3. Qwen native point_2d / bbox_2d list convention.
    match = _QWEN_POINT_LIST_RE.search(stripped)
    if match:
        try:
            objs = json.loads(match.group(0))
            if isinstance(objs, list) and objs:
                first = objs[0]
                if "point_2d" in first:
                    x, y = first["point_2d"]
                    return ParseResult(
                        ok=True, x_px=float(x), y_px=float(y),
                        reason="qwen_point_2d", raw_text=raw_text,
                    )
                if "bbox_2d" in first:
                    x1, y1, x2, y2 = first["bbox_2d"]
                    return ParseResult(
                        ok=True, x_px=(float(x1) + float(x2)) / 2.0,
                        y_px=(float(y1) + float(y2)) / 2.0,
                        reason="qwen_bbox_2d_center", raw_text=raw_text,
                    )
        except (json.JSONDecodeError, TypeError, ValueError, KeyError, IndexError):
            pass

    # 4. PaliGemma2 native <loc> tokens (Y1 X1 Y2 X2, 1024-cell grid).
    loc_tokens = _PALIGEMMA_LOC_RE.findall(stripped)
    # Exactly 4 tokens is the original behavior (unchanged). 2026-09-19: a multiple of 4 (several
    # boxes, e.g. "detect" returning ';'-separated detections) now uses the FIRST box -- same
    # first-entry convention as the Qwen list branch. Previously that case was a parse failure
    # even though a valid first box was present, which would have mis-scored PaliGemma's
    # native "detect" mode as unparseable. Only affects outputs that formerly failed to parse.
    if len(loc_tokens) >= 4 and len(loc_tokens) % 4 == 0:
        try:
            y1, x1, y2, x2 = (int(t) / 1024.0 for t in loc_tokens[:4])
            x_px = ((x1 + x2) / 2.0) * image_w
            y_px = ((y1 + y2) / 2.0) * image_h
            return ParseResult(
                ok=True, x_px=x_px, y_px=y_px, reason="paligemma2_loc_tokens", raw_text=raw_text
            )
        except (ValueError, ZeroDivisionError):
            pass

    return ParseResult(
        ok=False,
        reason="no_known_convention_matched (tried: asked_json_contract, qwen_point_2d/bbox_2d, "
               "paligemma2_loc_tokens)",
        raw_text=raw_text,
    )


# Coordinate spaces a backend may declare for its parsed (x, y). A trailing "_yx" means the model
# answered (row, column) i.e. the parsed pair is (y, x) and is swapped before mapping. Added
# 2026-09-22 (`deployment/coord_space_probe.py`): the calibration probe evaluates exactly these
# names as hypotheses, so the fix it recommends for a mismatched backend is a string this function
# already understands (`norm1000`/`norm1` were previously only diagnostics in `score_prediction`).
COORD_SPACES = ("raw_image", "model_input", "norm1000", "norm1")


def parse_coord_space(coord_space: str) -> tuple:
    """`(base_space, swapped)` for a declared coord_space name; ValueError on an unknown name
    (a typo must never silently fall back to raw pixels)."""
    swapped = coord_space.endswith("_yx")
    base = coord_space[:-3] if swapped else coord_space
    if base not in COORD_SPACES:
        raise ValueError(f"unknown coord_space {coord_space!r}; known: {list(COORD_SPACES)} (each optionally "
                         f"suffixed '_yx' for row-first answers)")
    return base, swapped


def to_raw_px(
    x: float, y: float, coord_space: str, model_input_hw: Optional[tuple], raw_hw: tuple
) -> tuple:
    """Maps a parsed point into the RAW frame's pixel space.
      - `"raw_image"` (default): unchanged.
      - `"model_input"` with a known `model_input_hw=(h, w)`: rescales by raw/model_input per axis
        (unknown size: returned unchanged, as before).
      - `"norm1000"`: 0-1000 normalized over the raw frame (InternVL's native grounding convention).
      - `"norm1"`: 0-1 normalized over the raw frame.
      - any of the above + `"_yx"`: the model answered (y, x); swapped first."""
    base, swapped = parse_coord_space(coord_space)
    if swapped:
        x, y = y, x
    raw_h, raw_w = raw_hw
    if base == "model_input" and model_input_hw:
        mi_h, mi_w = model_input_hw
        return x * raw_w / float(mi_w), y * raw_h / float(mi_h)
    if base == "norm1000":
        return x / 1000.0 * raw_w, y / 1000.0 * raw_h
    if base == "norm1":
        return float(x) * raw_w, float(y) * raw_h
    return float(x), float(y)


class MinimalVLMPolicy(Policy):
    """One `Policy` implementation for ANY candidate backend -- see module docstring.
    `pixel_to_mm`: optional `Callable[[float, float], tuple[float, float]]` (e.g.
    `ml_vision.core.coordinate_math.HomographyProjector.project_point`, or a closure around
    `PixelToPhysicalMapper.pixels_to_mm`) for converting the parsed pixel point to real mm.
    Without it, `target_x_mm`/`target_y_mm` are NaN (see module docstring's "documented
    deviation" note) but the raw pixel point is still returned in `debug`."""

    def __init__(
        self,
        backend: VLMBackend,
        pixel_to_mm: Optional[Callable[[float, float], tuple]] = None,
        prompt_variant: str = "baseline",
    ) -> None:
        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"Unknown prompt_variant={prompt_variant!r}; choices are {list(PROMPT_VARIANTS)}"
            )
        self.backend = backend
        self.pixel_to_mm = pixel_to_mm
        self.prompt_variant = prompt_variant
        self.last_debug: dict = {}

    def reset(self) -> None:
        # This policy carries no cross-call state (each frame's prompt is self-contained --
        # unlike JetsonExpertPolicy's PredictionGate, there is no EMA/seed-window state to
        # reset here). Present to satisfy the Policy protocol.
        self.last_debug = {}

    def act(self, image: np.ndarray, instruction: Optional[str], state: dict) -> PolicyCommand:
        # Policy.act()'s documented image convention is "BGR frame straight from the camera
        # receiver" (see policy_interface.py) -- backends expect RGB (matches
        # qwen_vl_smoke_test.py / PIL.Image.fromarray conventions), so convert once here
        # rather than making every backend guess the channel order it was handed.
        if image.shape[-1] == 3:
            image_rgb = image[..., ::-1]
        else:
            image_rgb = image
        image_h, image_w = image_rgb.shape[0], image_rgb.shape[1]

        prompt, target_label, is_directional_fallback = build_prompt(
            instruction, image_w, image_h, prompt_variant=self.prompt_variant
        )

        t0 = time.time()
        backend_out: BackendOutput = self.backend.generate(image_rgb, prompt, target_label=target_label)
        total_act_s = time.time() - t0

        parsed = parse_minimal_baseline_output(
            backend_out.raw_text, image_w, image_h, native_points=backend_out.native_points
        )

        debug = {
            "backend_name": backend_out.backend_name,
            "prompt_variant": self.prompt_variant,
            "instruction": instruction,
            "target_label": target_label,
            "directional_fallback": is_directional_fallback,
            "raw_text": backend_out.raw_text,
            "parse_ok": parsed.ok,
            "parse_reason": parsed.reason,
            "generate_latency_s": backend_out.latency_s,
            "total_act_s": total_act_s,
            "load_time_s": backend_out.load_time_s,
            "peak_memory_gb": backend_out.peak_memory_gb,
            "dtype": backend_out.dtype,
            "coord_space": backend_out.coord_space,
            "model_input_hw": backend_out.model_input_hw,
            "raw_image_hw": (image_h, image_w),
        }

        if not parsed.ok:
            self.last_debug = debug
            # No fabricated target on parse failure -- NaN, same convention as the
            # no-calibration case below, so a caller checking for NaN catches both failure
            # modes uniformly. debug['parse_ok']=False is the authoritative signal either way.
            return PolicyCommand(
                target_x_mm=float("nan"), target_y_mm=float("nan"), debug=debug
            )

        # `target_point_px`: coordinates exactly as the model/parser produced them (unchanged
        # meaning, so older result files stay comparable). `target_point_raw_px`: the same point
        # expressed in the RAW frame's pixels, per the backend's declared coordinate space --
        # identical for "raw_image" backends, rescaled by raw/model_input for "model_input" ones
        # (Qwen2.5-VL). Anything converting to mm must use the raw-pixel point.
        debug["target_point_px"] = (parsed.x_px, parsed.y_px)
        raw_x, raw_y = to_raw_px(
            parsed.x_px, parsed.y_px, backend_out.coord_space, backend_out.model_input_hw,
            (image_h, image_w),
        )
        debug["target_point_raw_px"] = (raw_x, raw_y)
        self.last_debug = debug

        if self.pixel_to_mm is not None:
            target_x_mm, target_y_mm = self.pixel_to_mm(raw_x, raw_y)
        else:
            target_x_mm, target_y_mm = float("nan"), float("nan")

        return PolicyCommand(target_x_mm=target_x_mm, target_y_mm=target_y_mm, debug=debug)
