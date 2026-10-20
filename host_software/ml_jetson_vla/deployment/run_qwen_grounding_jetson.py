"""Plain local-execution driver for the full Qwen2.5-VL-3B grounding evaluation
(`training/colab/qwen_grounding_colab.ipynb`), for running directly on the Jetson AGX Orin instead
of via Colab (2026-10-20: Colab's 5.5h free-tier session cap, plus needing a fresh Google account
every time the quota runs out, makes it hard to finish a ~53,400-call run in one sitting).

**This script's real job today is to mop up whatever a Colab run didn't finish, not to start
from scratch.** As of 2026-10-20 the real checkpoint
(`host_software/data/qwen_results/qwen_grounding_results/checkpoint.jsonl`) already has 42,546+
calls from prior Colab sessions. Point `--checkpoint-path` at a local copy of that same file and
this script will skip every `(image_file, test)` pair already in it (`load_done_keys()`, identical
logic to the notebook's own resume check) and only run what's left.

Ports -- does not reinterpret -- `qwen_grounding_colab.ipynb` cells (c)/(d)/(e) verbatim (model
load at the pinned revision, prompt templates, parsers, the per-call generate()+checkpoint loop),
so new calls here are directly comparable to the ones already in the checkpoint: same `MODEL_ID`,
same pinned `REVISION`, same `MIN_PIXELS`/`MAX_PIXELS`, same prompts, same parsers, same JSONL
schema. Real differences from the notebook, and why:

- **No Drive, no zip mount.** `--bundle-dir` is a plain local path to an already-extracted bundle
  (`<bundle-dir>/manifest.csv` + `<bundle-dir>/images/`) -- this repo's own
  `host_software/data/colab_bundle/bundle/` already has both (built once by
  `build_colab_grounding_bundle.py` for every Colab upload so far), so there is nothing to
  re-derive from the bronze sessions or re-run ArUco homography for here.
- **`--checkpoint-path` is a plain local file**, not a Drive path. Point it at a local copy of
  whatever checkpoint a Colab run most recently wrote, to continue it rather than restart it.
- **No Colab runtime-type GPU assertion.** The notebook's `ACCEPTED_GPUS` check
  (`A100`/`L4`/`T4`) existed to catch picking the wrong Colab runtime type; irrelevant on a fixed
  device. This script just prints `torch.cuda.get_device_name(0)` for the record.
- **Per-call exception isolation, not in the notebook.** A single bad frame/decode error would
  kill the whole Colab run outright. Since this is meant to run unattended for hours
  (CLAUDE.md's >30-minute checkpointing rule), one failing call is recorded to the checkpoint with
  `parse_ok=False, parse_reason="generate_exception: ..."` and the run continues, instead of dying
  and losing whatever was left of an unattended session.

Run inside the already-built, already-proven `arm2-t5:r36.4.0` image
(`docs/JETSON_ARM2_SWEEP_LOCAL.md` has the Dockerfile/build steps -- this is the same image that
produced the validated 68.3% hit-rate baseline sweep, pinned to `transformers==5.17.0`, confirmed
`Qwen2_5_VLForConditionalGeneration` importable). Example, matching that doc's bind-mount shape:

    docker run --rm --runtime nvidia --init \\
      -v "$REPO":/workspace/VRI_2026 \\
      -v "$HFCACHE":/root/.cache/huggingface \\
      -w /workspace/VRI_2026/host_software \\
      arm2-t5:r36.4.0 \\
      python3 -u ml_jetson_vla/deployment/run_qwen_grounding_jetson.py \\
        --bundle-dir data/colab_bundle/bundle \\
        --checkpoint-path data/qwen_results/qwen_grounding_results/checkpoint.jsonl

Run in `tmux` on the host so an SSH drop cannot kill it -- the checkpoint survives a kill
regardless (every call is fsync'd immediately), but tmux lets you reattach and watch it finish
rather than just inferring it did from the file growing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# Must match qwen_grounding_colab.ipynb cell (c) exactly -- this is what makes new calls here
# comparable to the ones already in the checkpoint.
REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
MIN_PIXELS = 50176
MAX_PIXELS = 200704
DEFAULT_MAX_NEW_TOKENS = 24

# Verbatim port of qwen_grounding_colab.ipynb cell (d) / core/minimal_vlm_policy.py's
# COLOR_COMMANDS/PROMPT_TEMPLATE/build_prompt (that notebook's own comment: "not paraphrased") --
# kept byte-identical so new calls score the same way as the 42,546+ already checkpointed.
COLOR_COMMANDS = {
    "go_red": "red marker",
    "go_green": "green marker",
    "go_yellow": "yellow marker",
    "go_black": "black marker",
}
COLOR_SET: Set[str] = set(COLOR_COMMANDS)
PROMPT_TEMPLATE = (
    "You are looking straight down at a small 3-DOF ball-balancing platform. "
    "Find the {target_label} in this image. "
    "Respond with ONLY a single JSON object and nothing else, in exactly this form: "
    '{{"target_point_xy": [x, y]}} '
    "where x and y are pixel coordinates in THIS image (x in [0, {width}], y in [0, {height}])."
)
MOTOR_PROMPT = (
    "Estimate the current joint angles of this 3-DOF ball-balancing platform: theta_a, theta_b and theta_c. "
    "Respond with ONLY a single JSON object and nothing else, in exactly this form: "
    '{"theta_a": <degrees>, "theta_b": <degrees>, "theta_c": <degrees>} '
    "where each value is the angle of that joint in degrees."
)
MOTOR_KEYS = ("theta_a", "theta_b", "theta_c")

_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)
_QWEN_POINT_LIST_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)


def build_prompt(instruction: Optional[str], image_w: int, image_h: int) -> Tuple[str, str]:
    """Returns (prompt_text, target_label). Non-colour instructions (and None) get the 'ball'
    label, which is the ball-position prompt. Verbatim port of the notebook cell (d) helper."""
    label = COLOR_COMMANDS[instruction] if instruction in COLOR_COMMANDS else "ball"
    return PROMPT_TEMPLATE.format(target_label=label, width=image_w, height=image_h), label


def parse_point_output(raw_text: str) -> Tuple[bool, Optional[float], Optional[float], str]:
    """Verbatim port of the notebook cell (d) parser (Qwen-only conventions, no PaliGemma <loc>)."""
    stripped = raw_text.strip()
    match = _JSON_OBJECT_RE.search(stripped)
    if match:
        try:
            obj = json.loads(match.group(0))
            pt = obj.get("target_point_xy")
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                return True, float(pt[0]), float(pt[1]), "asked_json_contract"
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            pass
    match = _QWEN_POINT_LIST_RE.search(stripped)
    if match:
        try:
            objs = json.loads(match.group(0))
            if isinstance(objs, list) and objs:
                first = objs[0]
                if "point_2d" in first:
                    x, y = first["point_2d"]
                    return True, float(x), float(y), "qwen_point_2d"
                if "bbox_2d" in first:
                    x1, y1, x2, y2 = first["bbox_2d"]
                    return True, (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0, "qwen_bbox_2d_center"
        except (json.JSONDecodeError, TypeError, ValueError, KeyError, IndexError):
            pass
    return False, None, None, "no_known_convention_matched"


def parse_motor_output(raw_text: str) -> Tuple[bool, Optional[List[float]], str]:
    """Verbatim port of the notebook cell (d) parser."""
    match = _JSON_OBJECT_RE.search(raw_text.strip())
    if not match:
        return False, None, "no_json_object"
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False, None, "json_decode_error"
    try:
        values = [float(obj[k]) for k in MOTOR_KEYS]
    except (KeyError, TypeError, ValueError):
        return False, None, "missing_or_nonnumeric_key"
    return True, values, "theta_json"


def load_done_keys(path: str) -> Set[Tuple[str, str]]:
    """Identical logic to the notebook cell (e) helper of the same name."""
    done: Set[Tuple[str, str]] = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    done.add((rec["image_file"], rec["test"]))
    return done


def append_record(path: str, rec: dict) -> None:
    """Identical logic to the notebook cell (e) helper: fsync after every line so a kill -9 or a
    power loss never loses more than the single in-flight call."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_model(revision: str = REVISION) -> Tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    print("Loading", MODEL_ID, "at revision", revision, "...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, revision=revision, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
    )
    loaded_commit = getattr(model.config, "_commit_hash", None)
    print("pinned revision:", revision, "| config _commit_hash:", loaded_commit)
    if loaded_commit is not None and loaded_commit != revision:
        raise RuntimeError(f"loaded commit {loaded_commit} differs from pinned {revision}")
    print("parameter dtype:", next(model.parameters()).dtype)
    return model, processor


def generate(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[str, float, List[int]]:
    """Verbatim port of the notebook cell (e) generate() -- same message shape as
    `deployment/qwen_vl_smoke_test.py`'s `generate_qwen_vl()`, including its per-image
    `max_pixels`, so the vision tower sees the same input the earlier Jetson/Colab runs did."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path, "max_pixels": 1280 * 28 * 28},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",
    ).to("cuda")
    grid = inputs["image_grid_thw"]
    patch = int(getattr(processor.image_processor, "patch_size", 14))
    model_input_hw = [int(grid[0][1]) * patch, int(grid[0][2]) * patch]
    start = time.time()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    latency = time.time() - start
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return output_text, latency, model_input_hw


def run(bundle_dir: str, checkpoint_path: str, max_frames: Optional[int], max_new_tokens: int) -> None:
    manifest = pd.read_csv(os.path.join(bundle_dir, "manifest.csv"))
    # manifest["image_file"] is already "images/<name>.jpg" (build_colab_grounding_bundle.py writes
    # it relative to bundle_dir, not relative to an images/ subdirectory) -- join directly against
    # bundle_dir, matching the notebook's own `os.path.join(BUNDLE, row.image_file)`. Joining against
    # a separate images_dir here doubled the prefix (2026-10-20, first real run on the Jetson: every
    # one of the 10,850 remaining calls failed with FileNotFoundError on .../images/images/...).
    work_manifest = manifest.head(max_frames) if max_frames is not None else manifest

    done = load_done_keys(checkpoint_path)
    total_calls_needed = sum(
        3 if row.audio_command in COLOR_SET else 2 for row in work_manifest.itertuples(index=False)
    )
    print(
        f"already checkpointed: {len(done)} | frames this run: {len(work_manifest)} "
        f"| total calls needed for this manifest slice: {total_calls_needed} "
        f"| remaining: {max(total_calls_needed - len(done), 0)}"
    )

    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: no CUDA device visible -- this will be extremely slow on CPU", file=sys.stderr)

    model, processor = load_model()

    written = 0
    skipped = 0
    errors = 0
    run_start = time.time()
    for row in work_manifest.itertuples(index=False):
        image_path = os.path.join(bundle_dir, row.image_file)
        tests = ["ball", "motor"] + (["marker"] if row.audio_command in COLOR_SET else [])
        for test in tests:
            key = (row.image_file, test)
            if key in done:
                skipped += 1
                continue
            try:
                img_w, img_h = Image.open(image_path).size
                if test == "ball":
                    prompt, _ = build_prompt(None, img_w, img_h)
                elif test == "marker":
                    prompt, _ = build_prompt(row.audio_command, img_w, img_h)
                else:
                    prompt = MOTOR_PROMPT
                raw_text, latency, model_input_hw = generate(model, processor, image_path, prompt, max_new_tokens)
                if test == "motor":
                    ok, values, reason = parse_motor_output(raw_text)
                    x_px = y_px = None
                    theta = values
                else:
                    ok, x_px, y_px, reason = parse_point_output(raw_text)
                    theta = None
                record = {
                    "image_file": row.image_file, "session": row.session,
                    "frame_index": int(row.frame_index), "audio_command": row.audio_command,
                    "test": test, "raw_text": raw_text, "parse_ok": ok, "parse_reason": reason,
                    "x_px": x_px, "y_px": y_px, "theta": theta, "image_w": img_w, "image_h": img_h,
                    "model_input_hw": model_input_hw, "latency_s": latency, "revision": REVISION,
                }
            except Exception as exc:  # noqa: BLE001 -- deliberate: one bad frame must not kill an unattended run
                # Deliberately NOT checkpointed and NOT added to `done`: an exception here (bad
                # path, corrupt image, a transient CUDA error, ...) means this pair was never
                # actually attempted with the model, so it must stay eligible for the NEXT run to
                # retry -- checkpointing it as a "done" parse_ok=False result would permanently
                # skip it on every future run even after the underlying problem is fixed (exactly
                # what happened 2026-10-20 with the images/images/ path bug: all 10,850 affected
                # pairs got wrongly marked done and had to be stripped out of the checkpoint by
                # hand before they could be retried).
                errors += 1
                print(f"ERROR on {row.image_file} / {test}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            append_record(checkpoint_path, record)
            done.add(key)
            written += 1
            if written % 20 == 0:
                print(f"written {written} | skipped {skipped} | errors {errors} | elapsed {time.time() - run_start:.0f}s")

    print(
        f"done this run: written {written} | skipped {skipped} | errors {errors} "
        f"| elapsed {time.time() - run_start:.0f}s"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--bundle-dir", default="data/colab_bundle/bundle",
        help="Directory with manifest.csv + images/ (same layout the Colab bundle unzips to). "
             "Path is relative to the container's working directory (host_software/), matching "
             "docs/JETSON_ARM2_SWEEP_LOCAL.md's bind-mount convention.",
    )
    ap.add_argument(
        "--checkpoint-path", default="data/qwen_results/qwen_grounding_results/checkpoint.jsonl",
        help="Local JSONL checkpoint, same schema as the Colab run's Drive checkpoint. "
             "Already-done (image_file, test) pairs are skipped -- point this at a copy of the "
             "most recent Colab checkpoint to continue it, not restart it.",
    )
    ap.add_argument(
        "--max-frames", type=int, default=None,
        help="Limit to the first N manifest rows, for a smoke run. Default: the whole manifest "
             "(already-done pairs are still skipped, so this is safe to leave unset).",
    )
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    args = ap.parse_args(argv)

    run(args.bundle_dir, args.checkpoint_path, args.max_frames, args.max_new_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
