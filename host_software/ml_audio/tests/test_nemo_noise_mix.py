"""Functional smoke test for the NOISE_MIX wiring in colab_nemo_finetune.ipynb
(see docs/plans/audio_eval_notebook_refactor_plan.md, "Scaling-Up Result" and
the noise-mixing section that follows it).

Requires NeMo, which is NOT installed in the project's default
ball_balance_env -- run this with the nemo_local conda env instead, same as
every other NeMo-track script in this plan:
    C:/Users/Admin/.conda/envs/nemo_local/python.exe tests/test_nemo_noise_mix.py

Exists because this wiring already caught two real bugs when it was first
tested ad hoc (not committed anywhere, so the next person/agent touching
this would have had to re-discover both from scratch):
1. Our raw recordings in data/01_background_noise/ are stereo; NeMo's
   NoisePerturbation requires the noise file's channel count to match the
   (mono) command clip and raises ValueError otherwise -- resolved by
   nemo_noise_manifest.py's mono/16kHz caching step, checked here again.
2. model.cfg.train_ds.augmentor is an OmegaConf DictConfig in "struct" mode
   -- adding a new `noise` key via plain attribute assignment raises
   `ConfigAttributeError: Key 'noise' is not in struct`; needs
   omegaconf.open_dict(). Checked here so a future refactor that drops the
   open_dict() call by accident fails fast, locally, in seconds -- not an
   hour into a paid Colab run.

Cheap and offline-ish (loads the pretrained checkpoint from cache/downloads
it once, builds tiny manifests, does not train) -- meant to run in seconds,
before spending Colab compute on a real NOISE_MIX fine-tuning pass.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ML_AUDIO_DIR = os.path.dirname(SCRIPT_DIR)
HOST_SOFTWARE_DIR = os.path.dirname(ML_AUDIO_DIR)
if HOST_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, HOST_SOFTWARE_DIR)


def test_noise_perturbation_on_real_clips(tmp_dir: str) -> None:
    from nemo.collections.asr.parts.preprocessing import perturb
    from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

    from ml_audio.training.nemo_noise_manifest import build_noise_manifest
    from ml_audio.training.train_audio_command_classifier import DEFAULT_DATASET_ROOT

    manifest_path = os.path.join(tmp_dir, "noise_manifest.json")
    entries = build_noise_manifest(os.path.join(tmp_dir, "noise_cache"), manifest_path)
    assert len(entries) == 3, f"expected 3 noise sources, got {len(entries)}"

    noiser = perturb.NoisePerturbation(manifest_path=manifest_path, min_snr_db=0, max_snr_db=15, rng=0)

    backward_dir = os.path.join(DEFAULT_DATASET_ROOT, "train", "backward")
    candidates = sorted(os.path.join(backward_dir, f) for f in os.listdir(backward_dir))[:5]
    assert candidates, f"no backward clips found under {backward_dir}"

    for path in candidates:
        seg = AudioSegment.from_file(path, target_sr=16000)
        clean = seg._samples.copy()
        noiser.perturb(seg)  # would raise ValueError on a channel mismatch
        diff = np.max(np.abs(seg._samples - clean))
        assert diff > 1e-6, f"{path}: perturb() did not change the audio at all"

    print(f"PASS: NoisePerturbation mixed noise into {len(candidates)} real backward clips, no channel-mismatch errors.")


def test_augmentor_wiring_through_dataloader(tmp_dir: str) -> None:
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    from ml_audio.training.nemo_manifest import build_manifest_entries, write_manifest
    from ml_audio.training.nemo_noise_manifest import build_noise_manifest
    from ml_audio.training.train_audio_command_classifier import DEFAULT_DATASET_ROOT, discover_labels

    labels = discover_labels(DEFAULT_DATASET_ROOT)

    # Stratified subset (a few clips per class) so the label-order assertion
    # after setup_training_data is meaningful -- an unstratified head-of-list
    # slice is dominated by whichever class sorts first (_background_) and
    # will fail that assertion for reasons unrelated to NOISE_MIX.
    all_entries = build_manifest_entries(DEFAULT_DATASET_ROOT, "val", labels)
    by_label: dict[str, list[dict]] = {}
    for e in all_entries:
        by_label.setdefault(e["label"], []).append(e)
    subset = [e for label in labels for e in by_label[label][:6]]
    assert len(set(e["label"] for e in subset)) == len(labels), "stratified subset is missing a class"

    train_manifest_path = os.path.join(tmp_dir, "train_manifest.json")
    write_manifest(subset, train_manifest_path)

    noise_manifest_path = os.path.join(tmp_dir, "noise_manifest.json")
    build_noise_manifest(os.path.join(tmp_dir, "noise_cache"), noise_manifest_path)

    model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
        model_name="commandrecognition_en_matchboxnet3x1x64_v2"
    )
    model.change_labels(labels)
    assert list(model.cfg.labels) == labels

    model.cfg.train_ds.manifest_filepath = os.path.abspath(train_manifest_path)
    model.cfg.train_ds.labels = labels
    model.cfg.train_ds.batch_size = 8
    model.cfg.train_ds.shuffle = True

    # The open_dict() this test exists to guard: without it, this raises
    # ConfigAttributeError because "noise" isn't in the pretrained recipe's
    # original augmentor schema (only "shift"/"white_noise" are).
    with open_dict(model.cfg.train_ds.augmentor):
        model.cfg.train_ds.augmentor.noise = {
            "manifest_path": os.path.abspath(noise_manifest_path),
            "prob": 1.0,
            "min_snr_db": 0,
            "max_snr_db": 15,
        }

    model.setup_training_data(train_data_layer_config=model.cfg.train_ds)
    assert list(model.labels) == labels, f"label order drifted: {model.labels}"

    batch = next(iter(model.train_dataloader()))
    audio, audio_len, targets, _targets_len = batch
    assert audio.shape[0] == 8

    logits = model(input_signal=audio, input_signal_length=audio_len)
    assert logits.shape[-1] == len(labels), f"expected {len(labels)} classes, got {logits.shape[-1]}"

    print("PASS: augmentor accepts the noise key via open_dict(), dataloader + forward pass run end-to-end.")


def main() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="nemo_noise_mix_test_")
    try:
        test_noise_perturbation_on_real_clips(tmp_dir)
        test_augmentor_wiring_through_dataloader(tmp_dir)
        print("\nALL CHECKS PASSED.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
