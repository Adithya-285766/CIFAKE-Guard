"""Build an *unseen-generator* benchmark: CIFAR-10-style fakes from a pixel-space DDPM.

CIFAKE's FAKE class was produced by Stable Diffusion v1.4 (a latent diffusion model at 512 px, then
down-scaled). Here we generate images with `google/ddpm-cifar10-32` (Ho et al., 2020): a *different*
architecture (pixel-space UNet, no VAE decoder), trained directly on CIFAR-10 at 32x32 - i.e. on the
same distribution as CIFAKE's REAL class. A detector that merely learned "SD-decoder artefacts" or
"down-scaling statistics" is expected to struggle here, which is exactly what we want to measure.

To remove storage format as a confounder, fakes are saved exactly like CIFAKE images: 32x32 RGB JPEG,
libjpeg quality 75 (verified to reproduce CIFAKE's quantisation tables). REAL images are copied from
CIFAKE's official test split (never used for training).

Output layout:  <out>/REAL/*.jpg  <out>/FAKE/*.jpg  <out>/meta.json
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from cifake_guard.data import find_split_root, resolve_data_root  # noqa: E402
from cifake_guard.utils import get_logger, save_json  # noqa: E402

log = get_logger("unseen")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_unseen/ddpm_cifar10")
    ap.add_argument("--n", type=int, default=1000, help="number of fake (and real) images")
    ap.add_argument("--steps", type=int, default=100, help="DDIM sampling steps")
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    from diffusers import DDIMScheduler, DDPMPipeline

    out = Path(args.out)
    (out / "FAKE").mkdir(parents=True, exist_ok=True)
    (out / "REAL").mkdir(parents=True, exist_ok=True)
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    if torch.cuda.is_available():
        pipe.to("cuda")              # the CPU generator below keeps sampling reproducible across devices
    gen = torch.Generator().manual_seed(args.seed)

    done = len(list((out / "FAKE").glob("*.jpg")))
    t0 = time.time()
    while done < args.n:
        bs = min(args.batch, args.n - done)
        images = pipe(batch_size=bs, generator=gen, num_inference_steps=args.steps, output_type="pil").images
        for im in images:
            im.convert("RGB").save(out / "FAKE" / f"ddpm_{done:05d}.jpg", format="JPEG", quality=75)
            done += 1
        log.info("generated %d/%d fakes (%.1f s)", done, args.n, time.time() - t0)

    # REAL counterpart: a seeded random sample of CIFAKE's official test REAL images
    test_real = find_split_root(resolve_data_root(args.data_root), "test") / "REAL"
    names = sorted(p.name for p in test_real.iterdir())
    rng = np.random.default_rng(args.seed)
    for name in rng.choice(names, size=args.n, replace=False):
        shutil.copy(test_real / name, out / "REAL" / name)
    save_json({"generator": "google/ddpm-cifar10-32 (DDIM sampler)", "steps": args.steps, "seed": args.seed,
               "n_fake": args.n, "n_real": args.n, "real_source": "CIFAKE test/REAL (CIFAR-10 test)",
               "storage": "32x32 JPEG quality 75 (matches CIFAKE)"}, out / "meta.json")
    log.info("Unseen-generator benchmark written to %s", out)


if __name__ == "__main__":
    main()
