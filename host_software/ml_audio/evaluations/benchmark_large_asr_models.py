"""Benchmark large general-purpose ASR models against the production NeMo
MatchboxNet command classifier, on the exact same held-out val split every
`nemo_finetune_confusion_matrix_*` report already uses.

Approach (per explicit user direction, not a default choice): full
transcribe -> parse-command. Each large ASR model transcribes a clip as free
text; the transcript is mapped onto our 12-class command vocabulary via
keyword/substring matching, then scored with the same confusion-matrix
schema `evaluate_nemo_checkpoint.py` already writes, plus per-clip latency
and a model footprint (param count, and peak process RSS as a stand-in for
VRAM -- this machine's `ball_balance_env` torch build is CPU-only, verified
via `torch.cuda.is_available()` before writing this, not assumed).

Uses `build_manifest_entries(DEFAULT_DATASET_ROOT, "val", labels)` -- the
identical function `evaluate_nemo_checkpoint.py` calls -- so this is the
same 2448-clip split, not a new one (per `dataset-integrity-check`: same
underlying session/file identity, not just "looks similar").

Note: the dataset's actual discovered labels are the 12 classes trained
against (`_background_, backward, forward, go_blue, go_green, go_grey,
go_red, go_yellow, hold, left, right, stop`) -- `go_black` is not currently
a trained/eval-able class in `training_v2/`, so it is intentionally absent
from `COMMAND_KEYWORDS` below; adding it here without matching training
data would make the keyword map wider than what's actually being compared.

Usage:
    python benchmark_large_asr_models.py --model whisper-large-v3 --limit 5
    python benchmark_large_asr_models.py --model whisper-large-v3
    python benchmark_large_asr_models.py --model canary-1b
    python benchmark_large_asr_models.py --model qwen2-audio-7b

Colab / GPU setup (this is where all three models should actually be run at
reportable scale, not on the dev machine): a real 5-clip local CPU timing
probe on `whisper-large-v3` measured 44.4s/clip mean -- ~30 hours
extrapolated for the full 2448-clip val split, for that model alone.
`qwen2-audio-7b` (7B params vs. Whisper's 1.5B) is expected to be worse on
both latency and memory (fp32 weights alone are ~28GB against this dev
machine's 34GB total RAM). Deferred here for exactly that reason -- verified
by measurement, not assumed.

All paths in this module are built from `SCRIPT_DIR`/`ML_AUDIO_DIR` via
`os.path.join()` (no hardcoded OS-specific separators or absolute Windows
paths) -- the same pattern `colab_nemo_finetune.ipynb` and every other
Colab-run script in this plan already uses, so this only needs the repo
layout present, not a specific OS.

1. Get the dataset onto the Colab instance the same way `colab_nemo_finetune.ipynb`
   currently does (per "Colab Data Flow: DVC Revert Was Intentional,
   2026-09-17" in `docs/plans/audio_eval_notebook_refactor_plan.md` -- the
   zip/Drive flow via `prepare_colab_package.py` is the current, intentional
   convention, not the DVC/git-clone flow an earlier draft of that plan tried
   and reverted): run `prepare_colab_package.py` locally, upload the zip to
   Drive, mount Drive in the Colab session, and extract it -- then pass the
   extracted `training_v2/`'s parent directory as `--dataset-root` (defaults
   to this repo's local `data/synthetic+real_dataset_large/training_v2`,
   which won't exist on a fresh Colab instance).
2. Install dependencies (matches what's already in this repo's
   `environment.yml`/`requirements.txt`):
       pip install transformers accelerate psutil soundfile
       pip install "nemo_toolkit[asr]"   # only needed for --model canary-1b
3. Smoke-test before committing to a full run:
       python benchmark_large_asr_models.py --model whisper-large-v3 --limit 5
   Confirms the model loads, transcribes, and the keyword-mapping/scoring
   path works end-to-end, and gives a real per-clip latency reading for
   *that* session's hardware before spending GPU time on the full 2448 clips.
4. Full run (drop `--limit`):
       python benchmark_large_asr_models.py --model whisper-large-v3
       python benchmark_large_asr_models.py --model canary-1b
       python benchmark_large_asr_models.py --model qwen2-audio-7b
   Each writes `evaluations/reports/asr_benchmark_<model>_<timestamp>.json`
   (accuracy, confusion matrix, latency percentiles, n_params, device,
   peak RSS/VRAM) -- copy these back out of the Colab session (e.g. into
   Drive, or `git diff`-committed manually) since Colab's local disk doesn't
   persist between sessions the way this repo's `evaluations/reports/` does
   locally.
"""

import argparse
import json
import os
import re
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from typing import Protocol

import numpy as np
import psutil
import soundfile as sf
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)

from ml_audio.training.nemo_manifest import build_manifest_entries  # noqa: E402
from ml_audio.training.train_audio_command_classifier import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    discover_labels,
)

DEFAULT_REPORT_DIR = os.path.join(SCRIPT_DIR, "reports")
ASR_MODELS_DIR = os.path.join(ML_AUDIO_DIR, "models", "benchmark_asr")

# Keyword/substring map, checked in this order (first match wins) against a
# lowercased, punctuation-stripped transcript. Order matters: "go grey"
# before "grey"-only substrings that could collide, longer/more specific
# phrases first so e.g. "go red" doesn't get pre-empted by a bare "red".
COMMAND_KEYWORDS: dict[str, list[str]] = {
    "go_blue": ["go blue", "blue"],
    "go_green": ["go green", "green"],
    "go_grey": ["go grey", "go gray", "grey", "gray"],
    "go_red": ["go red", "red"],
    "go_yellow": ["go yellow", "yellow"],
    "backward": ["backward", "back ward", "go back"],
    "forward": ["forward", "for word"],
    "left": ["left"],
    "right": ["right"],
    "hold": ["hold"],
    "stop": ["stop"],
}


def transcript_to_command(transcript: str, labels: list[str]) -> str:
    """Map a free-text ASR transcript onto the trained 12-class vocabulary.

    Empty/unmatched transcripts map to `_background_` -- the same "safe
    failure mode" semantics the production classifier's confidence/margin
    gating already uses (see `nemo_live_receiver.py`), so a model that
    correctly stays silent on background noise scores a correct
    `_background_` prediction rather than an unfair miss.
    """
    normalized = re.sub(r"[^a-z\s]", " ", transcript.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for label in labels:
        if label == "_background_":
            continue
        for keyword in COMMAND_KEYWORDS.get(label, []):
            if keyword in normalized:
                return label
    return "_background_"


class ASRModel(Protocol):
    n_params: int
    device: str

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


class WhisperASRModel:
    """OpenAI Whisper large-v3 via `transformers` (already a project
    dependency through other pipelines -- avoids adding a second,
    functionally-duplicate `openai-whisper` package for the same model)."""

    def __init__(self, model_id: str = "openai/whisper-large-v3") -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = os.path.join(ASR_MODELS_DIR, "whisper-large-v3")
        print(f"Loading {model_id} (cache: {cache_dir}, device: {self.device}) ...")
        self.processor = WhisperProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir, torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="en", task="transcribe"
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(self.device)
        with torch.no_grad():
            predicted_ids = self.model.generate(
                inputs, forced_decoder_ids=self.forced_decoder_ids, max_new_tokens=32
            )
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


class CanaryASRModel:
    """NVIDIA Canary-1B via NeMo -- reuses the `nemo_toolkit[asr]` install
    already added to this project for MatchboxNet, per the plan's own
    "least new plumbing" reasoning."""

    def __init__(self, model_id: str = "nvidia/canary-1b") -> None:
        import nemo.collections.asr as nemo_asr

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_id} (device: {self.device}) ...")
        self.model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        import tempfile

        assert sample_rate == 16000, "Canary-1B expects 16kHz input"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            tmp_path = tmp.name
        try:
            with torch.no_grad():
                result = self.model.transcribe([tmp_path], batch_size=1)
        finally:
            os.remove(tmp_path)
        first = result[0]
        return first.text if hasattr(first, "text") else str(first)


class Qwen2AudioASRModel:
    """Qwen2-Audio-7B-Instruct via `transformers`. Largest of the three --
    ties into the project's existing Qwen-derived model interest
    (Track 4 / Arm 2)."""

    def __init__(self, model_id: str = "Qwen/Qwen2-Audio-7B-Instruct") -> None:
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cache_dir = os.path.join(ASR_MODELS_DIR, "qwen2-audio-7b-instruct")
        print(f"Loading {model_id} (cache: {cache_dir}, device: {self.device}) ...")
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, cache_dir=cache_dir, torch_dtype=torch.float32
        ).to(self.device)
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.prompt = (
            "<|audio_bos|><|AUDIO|><|audio_eos|>"
            "Transcribe the spoken audio exactly as a short command or word."
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        inputs = self.processor(
            text=self.prompt, audios=[audio], sampling_rate=sample_rate, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=32)
        generated_ids = generated_ids[:, inputs["input_ids"].size(1):]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


MODEL_REGISTRY: dict[str, tuple[type, str]] = {
    "whisper-large-v3": (WhisperASRModel, "openai/whisper-large-v3"),
    "canary-1b": (CanaryASRModel, "nvidia/canary-1b"),
    "qwen2-audio-7b": (Qwen2AudioASRModel, "Qwen/Qwen2-Audio-7B-Instruct"),
}


def load_model(model_key: str) -> ASRModel:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Choices: {list(MODEL_REGISTRY)}")
    cls, model_id = MODEL_REGISTRY[model_key]
    return cls(model_id)


def run_benchmark(
    model: ASRModel,
    model_key: str,
    dataset_root: str,
    limit: int | None,
    report_dir: str,
) -> dict:
    labels = discover_labels(dataset_root)
    val_entries = build_manifest_entries(dataset_root, "val", labels)
    if limit is not None:
        val_entries = val_entries[:limit]
    print(f"Evaluating {model_key} against {len(val_entries)} val clips...")

    label_to_idx = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    latencies_ms: list[float] = []
    sample_transcripts: list[dict] = []

    process = psutil.Process(os.getpid())
    peak_rss_mb = process.memory_info().rss / 1e6

    for i, entry in enumerate(val_entries):
        audio, sr = sf.read(entry["audio_filepath"], dtype="float32")
        assert sr == 16000

        start = time.perf_counter()
        transcript = model.transcribe(audio, sr)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

        predicted_label = transcript_to_command(transcript, labels)
        true_idx = label_to_idx[entry["label"]]
        pred_idx = label_to_idx[predicted_label]
        matrix[true_idx, pred_idx] += 1

        peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / 1e6)

        if i < 20 or predicted_label != entry["label"]:
            sample_transcripts.append({
                "true_label": entry["label"],
                "predicted_label": predicted_label,
                "transcript": transcript,
                "latency_ms": round(elapsed_ms, 1),
            })

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(val_entries)} clips ({elapsed_ms:.0f}ms last clip)")

    correct = int(np.trace(matrix))
    total = len(val_entries)
    latencies_arr = np.array(latencies_ms)

    print(f"\n{model_key} accuracy: {correct/total:.3%} ({correct}/{total})")
    print(f"Latency ms: mean={latencies_arr.mean():.1f} p50={np.percentile(latencies_arr, 50):.1f} "
          f"p95={np.percentile(latencies_arr, 95):.1f} max={latencies_arr.max():.1f}")
    print(f"n_params={model.n_params:,} device={model.device} peak_rss_mb={peak_rss_mb:.0f}")

    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = os.path.join(report_dir, f"asr_benchmark_{model_key}_{timestamp}.json")
    report = {
        "generated_at": timestamp,
        "model_key": model_key,
        "n_params": int(model.n_params),
        "device": model.device,
        "peak_rss_mb": round(peak_rss_mb, 1),
        "labels": labels,
        "accuracy": correct / total,
        "total_clips": total,
        "correct": correct,
        "matrix": matrix.tolist(),
        "latency_ms_mean": float(latencies_arr.mean()),
        "latency_ms_p50": float(np.percentile(latencies_arr, 50)),
        "latency_ms_p95": float(np.percentile(latencies_arr, 95)),
        "latency_ms_max": float(latencies_arr.max()),
        "sample_transcripts": sample_transcripts,
        "note": (
            "device is CPU on this machine (torch.cuda.is_available() == False, "
            "verified before running) -- peak_rss_mb is process RSS, not VRAM."
            if model.device == "cpu" else "peak_rss_mb tracked alongside CUDA device; not VRAM."
        ),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a large ASR model (transcribe -> keyword-parse -> score) "
        "against the same held-out val split MatchboxNet reports use."
    )
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N val clips (for a timing/smoke probe before a full run).",
    )
    args = parser.parse_args()

    tracemalloc.start()
    model = load_model(args.model)
    run_benchmark(model, args.model, args.dataset_root, args.limit, args.report_dir)


if __name__ == "__main__":
    main()
