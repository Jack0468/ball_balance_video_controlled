import json
import logging
import torch
from torch.utils.data import Dataset
from PIL import Image

LOG = logging.getLogger(__name__)

# Matches the real, recognized audio_command vocabulary (core.minimal_vlm_policy.COLOR_COMMANDS'
# keys + DIRECTIONAL_COMMANDS), confirmed directly against real production telemetry
# (session_jetson_track4_*/telemetry.csv) -- fixed 2026-10-14. The previous space-separated vocab
# ("go red", "go blue", ...) never matched any real logged command string except "hold", so
# cmd_idx silently resolved to the "hold" index for every real sample -- language conditioning
# via FiLM (vla_architecture.py) was a no-op for the whole real-language-labeled dataset. A
# checkpoint trained under the old mapping must be retrained under this one.
VOCAB = {
    "hold": 0,
    "go_red": 1,
    "go_green": 2,
    "go_yellow": 3,
    "go_black": 4,
    "forward": 5,
    "backward": 6,
    "left": 7,
    "right": 8,
    "stop": 9,
}


class VLADataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.transform = transform
        self.vocab = VOCAB
        self._image_fallback_count = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image (Vision Token)
        img_path = item["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            # Real fallback path (drive remount, a refactor moving 03_gold/images/, a partial
            # sync) -- narrowed from a bare `except:` 2026-10-14 (that also caught
            # KeyboardInterrupt/SystemExit) and now counted/logged so a high fallback rate is
            # visible instead of silently degrading training/eval on blank images.
            self._image_fallback_count += 1
            LOG.warning(
                "VLADataset: image load failed for %r (%s); substituting blank image "
                "(fallback count this instance: %d)", img_path, exc, self._image_fallback_count,
            )
            img = Image.new("RGB", (224, 224))

        if self.transform:
            img = self.transform(img)

        # Audio Token (Word index)
        cmd_idx = self.vocab.get(item.get("audio_command", "hold"), 0)

        # State Token
        state = torch.tensor(
            [item.get("state_x", 0), item.get("state_y", 0)], dtype=torch.float32
        )

        # Action Targets (Expert PID output to imitate)
        action = torch.tensor(
            [
                item.get("action_theta_a", 0),
                item.get("action_theta_b", 0),
                item.get("action_theta_c", 0),
            ],
            dtype=torch.float32,
        )

        return img, cmd_idx, state, action
