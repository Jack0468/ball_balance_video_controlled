"""One-off diagnostic: times ONE `ACTPolicy.predict_action_chunk()` call against a real trained
checkpoint, using a dummy image+state tensor -- no camera/serial/motors involved at all. Built
2026-10-13 after `run_policy_live.py` ran the trained ACT checkpoint live with a real ball+touch
for 60s (CUDA) and 30s (CPU) and never saw a single chunk land in either case (not rejected, not
accepted -- the background `_FastWorker`'s one in-flight `fast_layer.plan()` call never returned),
ruling out both cold-start latency and CUDA/threading as the cause (same hang on CPU). This isolates
whether the hang is inside the model's forward pass itself (and on which device) without spending
any more live-hardware time or risking anything with the motors to find out.

Run inside the `arm2-lerobot` Docker image on the Jetson (needs the correctly-pinned
lerobot/transformers<5; attempting this on a transformers>=5 environment fails at import with an
unrelated, already-documented `backbone_cfg` TypeError -- see `Dockerfile.arm2-lerobot`'s header).
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="ACT pretrained_model directory")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args(argv)

    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    print("loading checkpoint...")
    t0 = time.perf_counter()
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy = policy.to(device=args.device, dtype=torch.float32).eval()
    print(f"loaded in {time.perf_counter() - t0:.2f}s")
    print("config.vision_backbone:", policy.config.vision_backbone)
    print("config.image_features:", policy.config.image_features)
    print("config.chunk_size:", policy.config.chunk_size)
    print("config.use_vae:", policy.config.use_vae)

    image = torch.rand(1, 3, 480, 640, dtype=torch.float32, device=args.device)
    state = torch.rand(1, 4, dtype=torch.float32, device=args.device)
    batch = {"observation.image": image, "observation.state": state}

    print("running predict_action_chunk() ONCE, timing it...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = policy.predict_action_chunk(batch)
    dt = time.perf_counter() - t0
    print(f"predict_action_chunk() returned in {dt:.2f}s, output shape {tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
