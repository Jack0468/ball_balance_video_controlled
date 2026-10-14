"""Per-session, pixel-only adaptive marker-colour calibration.

Context (docs/TODO.md, "Marker colour classification needs a robust design,
not another hardcoded fix -- REOPENED 2026-10-14"; docs/PROJECT_LOGBOOK.md
13-14/10/2026 entries): `marker_classifier.py`'s `COLOR_BINS`/`COLOR_HUE_CENTERS`
are hand-tuned constants calibrated once against one historical session
(`session_20260810_114330`, sheet `aruco_markers_03`, laptop webcam). The
physically-mounted sheet for live operation is a DIFFERENT sheet
(`aruco_markers_01`) under different lighting/camera -- real logged blobs from
that live camera (docs/PROJECT_LOGBOOK.md 13/10/2026, "Live-hardware
verification closes...") measured black_circle at (h=7.8, s=47.3, v=148.2) and
green_circle at (h=34.2, s=51.9, v=157.6), both of which fail every static bin
in the file (black's S cap is 20, far below 47.3; green's own strict bin check
loses to yellow's at H=34.2). Re-tuning the static bins a third time was
explicitly rejected in that entry as "not fixable by bin tuning": the failure
is a confounded sheet-identity + lighting/colour-cast shift, not an isolated
per-colour miscalibration.

A 2026-10-13 attempt fixed this by looking a blob's colour up by its already-
known physical position against `hardware/platform_templates/*_manifest.json`
(ground truth). That was reverted 2026-10-14 by explicit user instruction: it
answers "where is this marker" from the camera but "what colour is it" from
the answer key, defeating the actual point of running a vision pipeline at
inference time. See `marker_classifier.py`'s `manifest_path` parameter (kept,
but restricted to evaluation/training scripts, never live operation) and the
corresponding reverted-call-sites note in docs/PROJECT_LOGBOOK.md.

This module is the real fix: everything here is computed FROM THE CURRENT
SESSION'S OWN CAMERA PIXELS, never from `hardware/platform_templates/`. It
leans on two facts that are true regardless of which sheet is mounted or what
the lighting looks like:

  1. A colour marker is a static physical object for the life of a session
     (same fact `host_software/src/state_machine.py`'s `_MarkerPositionFilter`
     already relies on) -- so the SAME blob's HSV reading can be safely
     accumulated/averaged across many frames at (roughly) the same position,
     turning a single noisy per-frame reading into a stable per-slot estimate.
  2. This project's marker palette is always drawn from the same 5 colours
     (red/yellow/green/blue/black -- confirmed directly: every one of
     `aruco_markers_01/02/03_manifest.json`'s `features` lists only ever uses
     these 5 "color" values), and those colours have a FIXED, physically-true
     relative structure that survives a global lighting/colour-cast shift even
     when absolute HSV values don't:
       - black ink reflects less light than any saturated ink colour, so
         whichever marker is genuinely black is the darkest (lowest V) of the
         set actually present, even under a bad colour cast -- confirmed
         against the real logged `aruco_markers_01` frame above: V=148.2 for
         the true-black blob is still the lowest of all 5 real blobs there
         (yellow=234.7, red=214.8, green=157.6, blue=157.9), despite that same
         frame's black blob having far too much saturation (47.3) to be caught
         by any realistic achromatic S/V bin.
       - red/yellow/green/blue sit in a fixed cyclic hue order around the
         colour wheel (red ~0 deg -> yellow ~20-40 -> green ~45-85 -> blue
         ~90-150 -> wraps back to red) that a uniform lighting/white-balance
         shift rotates or compresses, but does not reorder, for colours this
         well separated. On the SAME real logged `aruco_markers_01` frame,
         absolute hues are nowhere near their historical `_03` centers
         (red=1.9, yellow=15.1, green=34.2, blue=129.8 -- green's own hue is
         actually below yellow's historical ceiling), yet their RELATIVE order
         (red < yellow < green < blue) is exactly right.

So: track each visually-distinct, spatially-stable blob as a "slot" across
frames; once enough slots have enough samples, split out the darkest slot as
"black" (only if it's clearly, not marginally, darker than the rest -- see
MIN_BLACK_V_GAP), then assign each remaining chromatic slot its own distinct
canonical colour by finding whichever contiguous run of the fixed [red,
yellow, green, blue] cyclic order best matches the whole ordered set of real
hues TOGETHER (see `_best_fit_canonical_run()` -- scored as one run, not each
slot against its own nearest fixed centre in isolation, which is what lets a
uniform hue shift wash out instead of flipping individual labels). No step
anywhere reads `hardware/platform_templates/`.

Known, deliberately-not-solved limitation: this assumes each distinct
position slot is a DIFFERENT colour, which is true for both sheets this
project has real captured data from (`aruco_markers_01`, `_03` -- each uses
all 4 chromatic colours exactly once). `aruco_markers_02` (4 same-coloured
shapes, per `hardware/platform_templates/aruco_markers_02_manifest.json`) is
NOT handled correctly by this version -- it would resolve the 4 shapes to 4
different (3 wrong) colours rather than recognising they're all the same.
Flagged rather than silently papered over; fixing it needs a real captured
`_02` frame to design and verify against, which doesn't exist yet.

This capability is OFF by default (`MarkerClassifier(adaptive_calibration=False)`,
matching `manifest_path=None`'s existing opt-in convention) -- wiring it into
any live call site is an explicit, separate decision left to orchestration,
same discipline as the reverted manifest fix. Even once wired, the static
`COLOR_BINS` path in `marker_classifier.py` always stays available as the
fallback for frames/slots this calibration hasn't resolved yet (warm-up, too
few markers visible, degenerate case) -- this fulfils CLAUDE.md's own locked
"static HSV bin thresholds first; [learned] fallback if <90% accuracy" design,
rather than replacing it.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

# Deliberately NOT importing marker_classifier.COLOR_HUE_CENTERS/
# _circular_hue_distance here, even though the values below are intentionally
# identical to them: marker_classifier.py imports THIS module (to build its
# opt-in adaptive-calibration tier), so an import in the other direction would
# be a circular import between the two files. Same discipline
# marker_classifier.py's own `_load_feature_positions_touch_mm()` docstring
# already argues for (duplicating ~10 stable lines rather than risking a
# cross-module import), applied to ~4 lines here instead.
#
# These are the SAME canonical OpenCV-hue (0-180) centers marker_classifier.py
# uses for its own nearest-hue fallback tier -- not a second, independently-
# tuned set of constants. If marker_classifier.py's centers are ever
# recalibrated, update both in the same change.
_CANONICAL_HUE_CENTERS = {
    "blue": 120.0,
    "red": 0.0,  # red wraps the hue circle; 0 == 180 on this scale
    "yellow": 31.0,
    "green": 65.0,
}


def _circular_hue_distance(h1: float, h2: float, hue_range: float = 180.0) -> float:
    """Wrap-aware distance between two OpenCV hues (0-180 scale, wraps at
    180->0). Duplicate of marker_classifier._circular_hue_distance() -- see
    this module's import-order note above for why it isn't imported instead."""
    diff = abs(h1 - h2) % hue_range
    return min(diff, hue_range - diff)


# Fixed, physically-true cyclic order of the 4 chromatic colours this project
# ever uses, ascending by each one's canonical OpenCV hue (0-180) center. Used
# only for relative/ordinal reasoning (which cluster is "the Nth one going
# round the wheel"), never as an absolute threshold.
CANONICAL_CHROMATIC_ORDER: List[str] = sorted(_CANONICAL_HUE_CENTERS, key=lambda c: _CANONICAL_HUE_CENTERS[c])

# A slot needs this many accumulated blob observations before its mean HSV is
# trusted enough to participate in calibration at all. Low enough that a
# marker which is only intermittently visible (partial occlusion by the ball,
# flaky mask-head confidence right at the detection threshold) still qualifies
# within a normal warm-up window, high enough that a single noisy frame can't
# define a slot's whole identity.
MIN_SAMPLES_PER_SLOT = 5

# Calibration fires once at least this many distinct slots exist (need >= 2 to
# do any relative reasoning at all -- 1 slot has no "relative" structure to
# compare against) AND each qualifying slot has >= MIN_SAMPLES_PER_SLOT, OR
# once MAX_WARMUP_FRAMES elapses, whichever comes first -- so a permanently
# occluded/undetected marker (matches the 15/09/2026 black_circle-at-dead-
# center logbook entry) can't stall calibration forever.
MIN_SLOTS_FOR_CALIBRATION = 2
MAX_WARMUP_FRAMES = 90

# Same "half the sheet's real 30mm inter-marker spacing" convention already
# used by marker_classifier.DEFAULT_MANIFEST_MATCH_RADIUS_MM and
# state_machine._MarkerPositionFilter.jump_threshold_mm -- a real blob
# revisiting its own slot next frame should land well inside this; a
# different marker's own blob (>=30mm away) should land well outside it.
SLOT_MATCH_RADIUS_MM = 15.0

# A slot is only labeled "black" if it is at least this much darker (mean V,
# averaged across all its accumulated observations -- not a single frame)
# than the next-darkest slot. Guards against forcing a "black" label onto
# whichever slot merely happens to be the dimmest of an all-chromatic set
# (e.g. aruco_markers_02's 4 blue shapes under slightly uneven lighting) when
# no genuinely-achromatic marker is present at all.
#
# Chosen (not guessed) against two real, independent data points
# (experiments/marker_classifier_adaptive_calibration_eval.py):
#   - The real logged aruco_markers_01 frame (module docstring) has black at
#     V=148.2 vs. the next-darkest real marker (green) at V=157.6 -- a gap of
#     only 9.4. This is the one real frame this fix specifically targets, so
#     the default MUST be below 9.4 or the fix does nothing for it.
#   - Swept 5/8/9/9.4/10/15/20 end-to-end against the real aruco_markers_03
#     dataset (1200 real sequential frames, session_20260810_114330): the
#     calibrator's own AGGREGATED (many-frame-averaged) black-vs-next-darkest
#     gap on that data is large enough that every value in this range gives
#     byte-identical results (black_circle 1199/1199, 100% every time) -- the
#     per-FRAME gap distribution has real noise (mean=29.7, but p5=-38.8,
#     16.7% of individual frames measure <9.4), yet averaging across the
#     several real observations a slot accumulates before calibration fires
#     washes almost all of that out well before any threshold in this range
#     would matter. Shipped 8.0 -- safely below the real aruco_markers_01
#     gap, with zero measured cost on the one large real dataset available.
MIN_BLACK_V_GAP = 8.0


@dataclass
class _Slot:
    x_mm: float
    y_mm: float
    n: int = 0
    _sin_sum: float = 0.0
    _cos_sum: float = 0.0
    s_sum: float = 0.0
    v_sum: float = 0.0

    def update(self, h: float, s: float, v: float, x_mm: float, y_mm: float) -> None:
        angle = h * (2.0 * math.pi / 180.0)
        self._sin_sum += math.sin(angle)
        self._cos_sum += math.cos(angle)
        self.s_sum += s
        self.v_sum += v
        self.n += 1
        # Slot position drifts toward the running mean of its own observations
        # (not just the first) -- smooths out soft-argmax jitter frame to
        # frame, same spirit as _MarkerPositionFilter's own running estimate.
        self.x_mm += (x_mm - self.x_mm) / self.n
        self.y_mm += (y_mm - self.y_mm) / self.n

    @property
    def mean_h(self) -> float:
        angle = math.atan2(self._sin_sum, self._cos_sum)
        return (angle * 180.0 / (2.0 * math.pi)) % 180.0

    @property
    def mean_s(self) -> float:
        return self.s_sum / self.n if self.n else 0.0

    @property
    def mean_v(self) -> float:
        return self.v_sum / self.n if self.n else 0.0


@dataclass
class CalibratedFeature:
    """One self-derived (not ground-truth) marker slot: a physical position
    this session's own camera has repeatedly observed, plus the colour label
    this calibrator's relative reasoning assigned it. Shape mirrors
    marker_classifier._load_feature_positions_touch_mm()'s return dicts
    closely enough that MarkerClassifier.classify() can match a live blob
    against either source with the same claim-by-position logic -- but the
    *content* here was never read from a manifest file."""

    x_mm: float
    y_mm: float
    color: str


@dataclass
class _CalibrationResult:
    features: List[CalibratedFeature] = field(default_factory=list)


class AdaptiveColorCalibrator:
    """Accumulates per-slot HSV statistics across frames and, once enough
    data exists, derives a position -> colour table from relative reasoning
    over the CURRENT session's own observed markers -- never from
    `hardware/platform_templates/`. See module docstring for the full method.

    Usage (mirrors MarkerClassifier's internal use): call `observe()` once per
    detected blob per frame; call `maybe_calibrate()` once per frame after all
    of that frame's blobs have been observed; once `is_calibrated` is True,
    `classify_position()` resolves a blob's position to a colour the same way
    `MarkerClassifier`'s existing manifest-position tier already does for
    ground-truth features.
    """

    def __init__(
        self,
        min_samples_per_slot: int = MIN_SAMPLES_PER_SLOT,
        min_slots_for_calibration: int = MIN_SLOTS_FOR_CALIBRATION,
        max_warmup_frames: int = MAX_WARMUP_FRAMES,
        slot_match_radius_mm: float = SLOT_MATCH_RADIUS_MM,
        min_black_v_gap: float = MIN_BLACK_V_GAP,
    ) -> None:
        self.min_samples_per_slot = min_samples_per_slot
        self.min_slots_for_calibration = min_slots_for_calibration
        self.max_warmup_frames = max_warmup_frames
        self.slot_match_radius_mm = slot_match_radius_mm
        self.min_black_v_gap = min_black_v_gap

        self._slots: List[_Slot] = []
        self._frames_seen = 0
        self._result: Optional[_CalibrationResult] = None

    @property
    def is_calibrated(self) -> bool:
        return self._result is not None

    @property
    def features(self) -> List[CalibratedFeature]:
        return self._result.features if self._result is not None else []

    def observe(self, h: float, s: float, v: float, x_mm: float, y_mm: float) -> None:
        """Feed one detected blob's mean HSV + position for this frame. No-op
        once calibration has already run -- recalibrating mid-session on top
        of an already-resolved table isn't needed for this fix and risks
        flip-flopping a label on transient noise; `reset()` starts over
        explicitly if that's ever wanted (e.g. a new sheet is mounted)."""
        if self.is_calibrated:
            return
        for slot in self._slots:
            if math.hypot(slot.x_mm - x_mm, slot.y_mm - y_mm) <= self.slot_match_radius_mm:
                slot.update(h, s, v, x_mm, y_mm)
                return
        new_slot = _Slot(x_mm=x_mm, y_mm=y_mm)
        new_slot.update(h, s, v, x_mm, y_mm)
        self._slots.append(new_slot)

    def maybe_calibrate(self) -> bool:
        """Call once per frame after that frame's observe() calls. Attempts
        calibration once the warm-up condition is satisfied; returns whether
        calibration is now complete (idempotent -- safe to call every frame)."""
        if self.is_calibrated:
            return True
        self._frames_seen += 1

        qualifying = [s for s in self._slots if s.n >= self.min_samples_per_slot]
        ready = len(qualifying) >= self.min_slots_for_calibration
        timed_out = self._frames_seen >= self.max_warmup_frames and len(qualifying) >= 1

        if not (ready or timed_out):
            return False

        self._result = _CalibrationResult(features=_label_slots(qualifying, self.min_black_v_gap))
        return True

    def classify_position(self, x_mm: float, y_mm: float) -> Optional[str]:
        """Nearest-slot colour lookup, same shape as MarkerClassifier's
        manifest-position tier. Returns None (never a wrong guess) if nothing
        is calibrated yet or nothing is close enough."""
        if not self.is_calibrated:
            return None
        best_color: Optional[str] = None
        best_dist = float("inf")
        for feature in self.features:
            dist = math.hypot(feature.x_mm - x_mm, feature.y_mm - y_mm)
            if dist < best_dist:
                best_dist = dist
                best_color = feature.color
        if best_color is not None and best_dist <= self.slot_match_radius_mm:
            return best_color
        return None

    def reset(self) -> None:
        """Discard all accumulated state and any calibration result -- e.g.
        operator confirms a different sheet has been mounted mid-process."""
        self._slots = []
        self._frames_seen = 0
        self._result = None


def _rotate_to_largest_gap(hues: List[float]) -> List[int]:
    """Returns an index permutation (into `hues`) that visits every hue in
    ascending circular order, starting just after the single largest circular
    gap between consecutive (circularly-sorted) hues -- so the 180->0 wrap
    point never lands in the middle of two genuinely-close hues, and the
    resulting sequence can be treated as a plain ascending list for the
    monotonic assignment below."""
    n = len(hues)
    order = sorted(range(n), key=lambda i: hues[i])
    sorted_hues = [hues[i] for i in order]
    if n == 1:
        return order
    gaps = [(sorted_hues[(i + 1) % n] - sorted_hues[i]) % 180.0 for i in range(n)]
    cut_idx = max(range(n), key=lambda i: gaps[i])
    return order[cut_idx + 1 :] + order[: cut_idx + 1]


def _best_fit_canonical_run(ordered_hues: List[float]) -> List[str]:
    """Given K hues already placed in a fixed ascending circular order (see
    _rotate_to_largest_gap), picks whichever length-K contiguous run of
    CANONICAL_CHROMATIC_ORDER (there are exactly 4 such runs, one per
    rotation) has the lowest total circular hue distance when matched
    INDEX-FOR-INDEX (position 1 of our data <-> position 1 of the run, etc),
    and returns that run's labels. This is genuinely ordinal, not nearest-
    absolute-centre-per-point: it is what lets a globally hue-shifted frame
    (e.g. the real aruco_markers_01 numbers in this module's docstring, where
    yellow's absolute hue is nearer red's historical centre than its own)
    still resolve correctly, since the WHOLE run is scored together and a
    uniform shift changes every term's cost by roughly the same amount,
    rather than changing which individual label wins a point-by-point nearest-
    centre contest. Verified end-to-end against those exact real numbers (see
    tests) to be the deciding reason this works where plain nearest-centre
    matching (what _nearest_hue_fallback() in marker_classifier.py already
    does) does not.

    Does NOT allow two of the K points to share a label (no merging) --
    deliberately: this project's two sheets with confirmed real data
    (aruco_markers_01, _03) each use all 4 chromatic colours exactly once, so
    assuming every distinct *position* slot is a distinct colour is correct
    for them. aruco_markers_02 (4 same-coloured shapes, no real data captured
    yet) would be mis-handled by this assumption -- a known, deliberately NOT
    solved limitation, flagged rather than hidden (see module docstring).
    """
    k = len(ordered_hues)
    order = CANONICAL_CHROMATIC_ORDER
    m = len(order)
    if k == 0:
        return []
    if k > m:
        # More distinct chromatic slots than this project has chromatic
        # colours -- most likely a spurious extra blob holding a stable
        # position (see min_blob_area_px's own docstring note in
        # marker_classifier.py for the known, separate, low-priority spurious-
        # blob issue) rather than a 5th real colour. Label the best-fitting
        # `m` of them and leave the rest unresolved instead of guessing.
        return _best_fit_canonical_run(ordered_hues[:m]) + ["unknown"] * (k - m)

    best_labels: Optional[List[str]] = None
    best_cost = float("inf")
    for start in range(m):
        run = [order[(start + i) % m] for i in range(k)]
        cost = sum(_circular_hue_distance(ordered_hues[i], _CANONICAL_HUE_CENTERS[run[i]]) for i in range(k))
        if cost < best_cost:
            best_cost = cost
            best_labels = run
    assert best_labels is not None
    return best_labels


def _assign_chromatic_labels(slots: List[_Slot]) -> List[str]:
    """Assigns each chromatic slot a distinct canonical colour label, via
    _best_fit_canonical_run() above. Returns labels in the SAME order as
    `slots` (not the internal rotated order), so callers can zip() directly."""
    if not slots:
        return []
    hues = [s.mean_h for s in slots]
    perm = _rotate_to_largest_gap(hues)
    ordered_hues = [hues[i] for i in perm]
    ordered_labels = _best_fit_canonical_run(ordered_hues)
    labels_by_slot_index = [""] * len(slots)
    for pos, slot_idx in enumerate(perm):
        labels_by_slot_index[slot_idx] = ordered_labels[pos]
    return labels_by_slot_index


def _label_slots(slots: List[_Slot], min_black_v_gap: float) -> List[CalibratedFeature]:
    """Full relative-labeling pipeline: split off at most one clearly-darkest
    "black" slot (see MIN_BLACK_V_GAP), then assign the rest distinct
    canonical chromatic labels via the ordinal best-fit-run match above."""
    if not slots:
        return []

    by_v = sorted(slots, key=lambda s: s.mean_v)
    black_slot: Optional[_Slot] = None
    if len(by_v) >= 2 and (by_v[1].mean_v - by_v[0].mean_v) >= min_black_v_gap:
        black_slot = by_v[0]

    chromatic = [s for s in slots if s is not black_slot]
    chromatic_labels = _assign_chromatic_labels(chromatic)

    features: List[CalibratedFeature] = []
    if black_slot is not None:
        features.append(CalibratedFeature(x_mm=black_slot.x_mm, y_mm=black_slot.y_mm, color="black"))
    for slot, label in zip(chromatic, chromatic_labels):
        features.append(CalibratedFeature(x_mm=slot.x_mm, y_mm=slot.y_mm, color=label))
    return features
