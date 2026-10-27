"""
Tests for the Text-to-Image + CLIP lab app.

Run from the project folder:

    python test_pipeline.py

The generation test uses a deliberately TINY real Stable Diffusion pipeline
(`hf-internal-testing/tiny-sd-pipe`, a few MB) so it runs on a
laptop CPU. It is a genuine Diffusers diffusion pipeline: prompt encoding,
latent preparation, the scheduler loop, VAE decode and postprocessing all run.
Only the model is small - nothing is stubbed, and no placeholder image is
substituted.

The CLIP test uses the real `openai/clip-vit-base-patch32` checkpoint.
"""

from __future__ import annotations

import gc
import threading
import traceback

import numpy as np
import torch
from PIL import Image

import app

TINY_MODEL = "hf-internal-testing/tiny-sd-pipe"
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn() or ""
        results.append((PASS, name, detail))
    except Exception as exc:                      # noqa: BLE001 - test harness
        results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))
        traceback.print_exc()


# --------------------------------------------------------------------------- #
# 1. Shape / layout validation (CHW vs HWC, RGBA vs RGB, batch dims)
# --------------------------------------------------------------------------- #

def test_chw_float_array() -> str:
    chw = np.random.rand(3, 32, 32).astype("float32")   # CHW, 0..1
    img = app.to_rgb_pil(chw)
    assert isinstance(img, Image.Image) and img.mode == "RGB", img
    assert img.size == (32, 32), img.size
    return "3x32x32 CHW float -> RGB PIL 32x32"


def test_rgba_hwc_array() -> str:
    rgba = np.zeros((16, 16, 4), dtype="uint8")         # HWC with alpha
    rgba[..., 0] = 200
    img = app.to_rgb_pil(rgba)
    assert img.mode == "RGB" and img.size == (16, 16)
    return "16x16x4 RGBA -> RGB (no channel-3 IndexError)"


def test_batched_4d_array() -> str:
    batch = np.random.rand(1, 3, 24, 24).astype("float32")
    img = app.to_rgb_pil(batch)
    assert img.size == (24, 24)
    return "1x3x24x24 batched -> RGB PIL 24x24"


def test_grayscale_and_pil() -> str:
    grey = np.full((8, 8), 128, dtype="uint8")
    assert app.to_rgb_pil(grey).mode == "RGB"
    pal = Image.new("P", (8, 8))
    assert app.to_rgb_pil(pal).mode == "RGB"
    return "2-D grayscale and palette PIL both -> RGB"


def test_bad_layout_rejected() -> str:
    try:
        app.to_rgb_pil(np.random.rand(2, 3, 4, 5, 6))
    except app.GenerationError as exc:
        return f"rejected with GenerationError: {str(exc)[:60]}..."
    raise AssertionError("a 5-D array was accepted")


# --------------------------------------------------------------------------- #
# 2. Scheduler state: the exact runtime bug and its fix
# --------------------------------------------------------------------------- #

def test_scheduler_overrun_reproduced() -> str:
    """The reported IndexError, reproduced deterministically.

    A scheduler configured for N steps holds N+1 sigmas. For the app's default
    2 steps that is 3 sigmas, and stepping once too often asks for sigmas[3]:
        IndexError: index 3 is out of bounds for dimension 0 with size 3
    One extra step is exactly what happens when a second generation calls
    set_timesteps() on the shared pipeline while a loop is still running.
    """
    from diffusers import EulerAncestralDiscreteScheduler

    sched = EulerAncestralDiscreteScheduler.from_config({"num_train_timesteps": 1000})
    sched.set_timesteps(2, device="cpu")
    assert len(sched.sigmas) == 3, len(sched.sigmas)
    latents = torch.randn(1, 4, 8, 8)
    for t in sched.timesteps:                       # the correct, complete loop
        latents = sched.step(torch.randn_like(latents), t, latents, return_dict=False)[0]

    try:                                            # one iteration too many
        sched.step(torch.randn_like(latents), sched.timesteps[-1], latents, return_dict=False)
    except IndexError as exc:
        assert "index 3 is out of bounds for dimension 0 with size 3" in str(exc), exc
        return f"reproduced: IndexError: {exc}"
    raise AssertionError("expected the scheduler overrun IndexError")


def test_prepare_scheduler_restores_state() -> str:
    from diffusers import EulerAncestralDiscreteScheduler

    class FakePipe:
        scheduler = EulerAncestralDiscreteScheduler.from_config({"num_train_timesteps": 1000})

    pipe = FakePipe()
    pipe.scheduler.set_timesteps(2, device="cpu")        # stale state from another run
    app.prepare_scheduler(pipe, 3, "cpu")                # must repair for 3 steps
    assert len(pipe.scheduler.timesteps) == 3
    assert len(pipe.scheduler.sigmas) == 4
    return "stale 2-step state repaired to 3 steps (sigmas 4)"


def test_size_validation() -> str:
    class FakePipe:
        class vae:
            class config:
                block_out_channels = [4, 8, 16, 16]      # vae_scale_factor = 8

        scheduler = None

    assert app._vae_scale_factor(FakePipe()) == 8
    try:
        app.validate_requested_size(FakePipe(), 250)
    except app.GenerationError as exc:
        return f"250px rejected: {str(exc)[:55]}..."
    raise AssertionError("250px should be rejected for a factor-8 VAE")


# --------------------------------------------------------------------------- #
# 3. REAL diffusion generation through the app's own code path
# --------------------------------------------------------------------------- #

def test_real_generation_and_concurrency() -> str:
    """Generate with a real (tiny) pipeline, twice concurrently with different
    step counts - the situation that used to raise the IndexError."""
    pipe = app.load_local_pipeline(TINY_MODEL)
    prompt = "a red bicycle beside a blue wall"
    outputs: dict[str, object] = {}
    errors: dict[str, str] = {}

    def worker(steps: int) -> None:
        try:
            img, secs = app.generate_local(
                prompt, steps, 64, seed=1234, pipe=pipe, model_id=TINY_MODEL
            )
            outputs[f"{steps}"] = (img, secs)
        except Exception as exc:                       # noqa: BLE001
            errors[f"{steps}"] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=worker, args=(2,)),
               threading.Thread(target=worker, args=(4,))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"generation failed: {errors}"
    for steps in ("2", "4"):
        img, secs = outputs[steps]
        assert isinstance(img, Image.Image), type(img)
        assert img.mode == "RGB", img.mode
        assert img.size == (64, 64), img.size
        assert np.asarray(img).std() > 0, "image is a flat colour"
        assert secs >= 0
    return (f"2-step and 4-step runs on one shared pipeline both OK "
            f"({outputs['2'][1]:.2f}s / {outputs['4'][1]:.2f}s, 64x64 RGB)")


def test_pipeline_lock_serialises() -> str:
    """Two overlapping generations must never enter the pipeline together.

    The cached pipeline owns ONE scheduler, so concurrent entry is what corrupts
    it. This asserts the lock actually serialises, deterministically.
    """
    import time as _time

    peak = {"now": 0, "max": 0}
    guard = threading.Lock()

    class FakeScheduler:
        def __init__(self):
            self.timesteps = []
            self.sigmas = []

        def set_timesteps(self, n, device=None):
            self.timesteps = list(range(n))
            self.sigmas = list(range(n + 1))

    class FakePipe:
        scheduler = FakeScheduler()

        class vae:
            class config:
                block_out_channels = [4, 8, 16, 16]

        def __call__(self, **kwargs):
            with guard:
                peak["now"] += 1
                peak["max"] = max(peak["max"], peak["now"])
            _time.sleep(0.15)                     # hold the scheduler for a while
            with guard:
                peak["now"] -= 1

            class Out:
                images = [Image.new("RGB", (64, 64), (10, 20, 30))]
            return Out()

    pipe = FakePipe()
    errors = []

    def worker(steps):
        try:
            app.generate_local("x", steps, 64, seed=1, pipe=pipe, model_id="fake")
        except Exception as exc:                   # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(2,)),
               threading.Thread(target=worker, args=(4,))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert peak["max"] == 1, f"pipeline entered concurrently {peak['max']}x"
    return f"max concurrent pipeline entries = {peak['max']} (serialised)"


# --------------------------------------------------------------------------- #
# 4. REAL CLIP scoring
# --------------------------------------------------------------------------- #

def test_real_clip_scoring() -> str:
    img = Image.new("RGB", (128, 128), (200, 40, 40))
    score = app.clip_similarity(
        "A small red bicycle parked beside a blue brick wall on a rainy city street.",
        img,
    )
    assert -1.0 <= score <= 1.0, score
    score2 = app.clip_similarity("a photo of a red object", img)
    assert abs(score2) <= 1.0
    return f"cosine={score:+.4f} (in range), second prompt={score2:+.4f}"


def test_clip_rejects_bad_tensor() -> str:
    class Broken:
        shape = (2, 3, 4, 5, 6)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((2, 3, 4, 5, 6))

    try:
        app.clip_similarity("x", Broken())
    except (app.ScoringError, app.GenerationError) as exc:
        return f"rejected: {type(exc).__name__}: {str(exc)[:55]}..."
    raise AssertionError("a 5-D image was accepted by CLIP scoring")


# --------------------------------------------------------------------------- #

def main() -> int:
    print("=" * 72)
    print("Shape / layout validation")
    print("=" * 72)
    check("CHW float array -> RGB PIL", test_chw_float_array)
    check("RGBA array -> RGB PIL", test_rgba_hwc_array)
    check("batched 4-D array -> RGB PIL", test_batched_4d_array)
    check("grayscale + palette PIL", test_grayscale_and_pil)
    check("5-D array rejected", test_bad_layout_rejected)

    print("=" * 72)
    print("Scheduler state (the reported IndexError)")
    print("=" * 72)
    check("overrun reproduced", test_scheduler_overrun_reproduced)
    check("prepare_scheduler repairs stale state", test_prepare_scheduler_restores_state)
    check("size validation", test_size_validation)

    print("=" * 72)
    print("Real diffusion generation (tiny pipeline, real code path)")
    print("=" * 72)
    check("generation + concurrency", test_real_generation_and_concurrency)
    check("lock serialises the pipeline", test_pipeline_lock_serialises)

    gc.collect()

    print("=" * 72)
    print("Real CLIP scoring")
    print("=" * 72)
    check("CLIP cosine similarity", test_real_clip_scoring)
    check("CLIP rejects bad tensor", test_clip_rejects_bad_tensor)

    print()
    width = max(len(n) for _, n, _ in results)
    for status, name, detail in results:
        print(f"[{status}] {name:<{width}}  {detail}")
    failed = sum(1 for s, _, _ in results if s == FAIL)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
