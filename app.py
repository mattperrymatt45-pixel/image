"""
Text-to-Image Generation and Semantic Evaluation using Generative AI
Multimodal Generative AI Laboratory

A small, working Streamlit application that generates images from natural-language
prompts and calculates a raw CLIP text-image cosine similarity score.

Backends
--------
* Local Diffusers  : stabilityai/sd-turbo  (no API key, no per-image fee)
* Hosted HF API    : stabilityai/stable-diffusion-3-medium-diffusers via
                     Hugging Face Inference Providers (needs an HF User Access Token)

CLIP scoring (openai/clip-vit-base-patch32) always runs locally on CPU.

All images and scores in this app are real runtime outputs. The app never
substitutes a random image or a fabricated score: failures are shown as errors.

Run:
    python -m streamlit run app.py
"""

from __future__ import annotations

import importlib.util
import io
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SD_TURBO_MODEL_ID = "stabilityai/sd-turbo"
SD3_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

BACKEND_LOCAL = "Local Diffusers (SD-Turbo)"
BACKEND_HOSTED = "Hosted Hugging Face Inference API"

# SD-Turbo is distilled for 1-4 denoising steps and targets 512x512.
ALLOWED_STEPS = [1, 2, 3, 4]
ALLOWED_SIZES = [256, 384, 512]
DEFAULT_SIZE = 512
DEFAULT_STEPS = 2
# The SD-Turbo model card uses guidance_scale=0.0 (no classifier-free guidance).
SD_TURBO_GUIDANCE = 0.0
MAX_RESULTS = 12

# The Diffusers pipeline is cached once per server process and shared by every
# Streamlit session. Its scheduler is STATEFUL (timesteps, sigmas, step_index),
# so two overlapping generations corrupt each other's scheduler state and the
# scheduler indexes past the end of its sigmas array. Every pipeline execution
# is therefore serialised by this lock.
_PIPELINE_LOCK = threading.Lock()

SAMPLE_PROMPTS = [
    "A small red bicycle parked beside a blue brick wall on a rainy city street at dusk, editorial photograph.",
    "A small red bicycle leaning against a blue brick wall in a quiet city alley at sunrise, photograph.",
    "A yellow kayak floating on a calm mountain lake at sunrise, watercolor illustration.",
]

REQUIRED_PACKAGES = {
    "torch": "torch",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "huggingface_hub": "huggingface_hub",
    "PIL": "Pillow",
}


class GenerationError(RuntimeError):
    """Raised when image generation fails. Never swallowed silently."""


class ScoringError(RuntimeError):
    """Raised when CLIP scoring fails. Never replaced by an invented value."""


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    """One successful generation + (optionally) its CLIP evaluation."""

    prompt: str
    image: object                      # PIL.Image.Image (RGB)
    backend: str
    model_id: str
    seed: int
    width: int
    height: int
    steps: int
    gen_seconds: float
    clip_score: float | None = None
    clip_error: str | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%H:%M:%S")
    )

    def png_bytes(self) -> bytes:
        buf = io.BytesIO()
        self.image.save(buf, format="PNG")
        return buf.getvalue()

    def settings_text(self) -> str:
        return (
            f"{self.width}x{self.height} · {self.steps} step"
            f"{'s' if self.steps != 1 else ''} · seed {self.seed}"
        )


def get_results() -> list[Result]:
    return st.session_state.setdefault("results", [])


def add_result(result: Result) -> None:
    results = get_results()
    results.insert(0, result)
    del results[MAX_RESULTS:]          # keep only the latest N


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #

def package_presence() -> dict[str, bool]:
    """True/False per importable package, without importing heavy libraries."""
    return {
        module: importlib.util.find_spec(module) is not None
        for module in REQUIRED_PACKAGES
    }


def require_packages(*modules: str) -> None:
    """Raise a clear error if a backend's dependencies are missing."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        pretty = ", ".join(REQUIRED_PACKAGES.get(m, m) for m in missing)
        raise GenerationError(
            f"Missing Python package(s): {pretty}. Activate the virtual "
            "environment and run `python -m pip install -r requirements.txt` "
            "(`python -m pip install torch` first if PyTorch is absent), then "
            "restart Streamlit."
        )


@st.cache_resource(show_spinner=False)
def torch_report() -> dict:
    """Device / version information for the sidebar status line."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"ok": False, "error": str(exc)}
    cuda = torch.cuda.is_available()
    return {
        "ok": True,
        "version": torch.__version__,
        "cuda": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "CPU",
    }


# --------------------------------------------------------------------------- #
# Shape / state validation
#
# These checks exist so that a bad tensor layout, a wrong channel count or a
# corrupted scheduler produces a precise, actionable message instead of a raw
# library IndexError/RuntimeError.
# --------------------------------------------------------------------------- #

def _vae_scale_factor(pipe) -> int:
    """Spatial downscale factor of the pipeline's VAE (8 for SD 1.x/2.x/turbo)."""
    try:
        return 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    except Exception:
        return 8


def validate_requested_size(pipe, size: int) -> None:
    """Fail early, and clearly, if width/height cannot be encoded by the VAE."""
    factor = _vae_scale_factor(pipe)
    size = int(size)
    if size <= 0 or size % factor != 0:
        raise GenerationError(
            f"Image size {size}x{size} is not usable by this model: width and "
            f"height must be positive multiples of {factor} (the VAE downsample "
            f"factor). Choose one of {ALLOWED_SIZES}."
        )


def prepare_scheduler(pipe, steps: int, device: str) -> None:
    """Put the shared, stateful scheduler into a known state and verify it.

    A scheduler configured for N steps has len(timesteps) == N and
    len(sigmas) == N + 1. If those disagree with the loop that is about to run,
    the scheduler indexes past the end of sigmas and raises
    ``IndexError: index 3 is out of bounds for dimension 0 with size 3``
    (3 == sigmas length for a 2-step schedule).
    """
    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is None:
        return
    try:
        scheduler.set_timesteps(int(steps), device=device)
    except TypeError:                      # older schedulers take no device kwarg
        scheduler.set_timesteps(int(steps))

    n_timesteps = len(getattr(scheduler, "timesteps", []))
    if n_timesteps != int(steps):
        raise GenerationError(
            f"Scheduler/mismatch before generation: requested {steps} steps but "
            f"the scheduler produced {n_timesteps} timesteps. This usually means "
            f"another session was using the shared pipeline at the same time."
        )
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is not None and len(sigmas) != int(steps) + 1:
        raise GenerationError(
            f"Scheduler state is inconsistent: {steps} steps require "
            f"{int(steps) + 1} sigma values but {len(sigmas)} were found. "
            "Restart the Streamlit server to clear the cached pipeline."
        )


def reset_scheduler_after_run(pipe, steps: int, device: str) -> None:
    """Best-effort self-heal: leave the shared scheduler clean for the next run.

    Called after an image has already been produced successfully, so a failure
    here must never destroy a valid result.
    """
    scheduler = getattr(pipe, "scheduler", None)
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None or len(sigmas) == int(steps) + 1:
        return
    try:
        prepare_scheduler(pipe, steps, device)
    except Exception:
        pass


def to_rgb_pil(image, expected_size: int | None = None):
    """Coerce a generator output into an RGB PIL image, validating the layout.

    Handles PIL images, numpy/torch arrays in HWC or CHW layout, with or without
    a leading batch dimension, and 1/3/4 channel data. A 4-channel (RGBA) input
    is converted to RGB rather than indexed at channel 3 and failing.
    """
    from PIL import Image as _Image

    if isinstance(image, _Image.Image):
        return image.convert("RGB") if image.mode != "RGB" else image

    if not hasattr(image, "shape"):
        raise GenerationError(
            f"The generator returned {type(image).__name__}, but a PIL image or "
            "an image array was expected."
        )

    import numpy as np

    array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    array = np.squeeze(array)

    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = array.transpose(1, 2, 0)          # CHW -> HWC
    elif array.ndim == 3 and array.shape[-1] in (3, 4):
        pass                                      # already HWC
    elif array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)    # grayscale -> RGB
    else:
        raise GenerationError(
            f"Unsupported image array shape {tuple(array.shape)}. Expected CHW "
            "or HWC with 1, 3 or 4 channels."
        )

    if array.dtype != np.uint8:
        if array.max() <= 1.0 + 1e-6:
            array = array * 255.0
        array = np.clip(array, 0, 255).round().astype("uint8")

    pil = _Image.fromarray(array)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    if expected_size is not None and pil.size != (int(expected_size), int(expected_size)):
        # Not fatal: some hosted providers round to their own size. The result
        # records the real dimensions it received.
        pass
    return pil


# --------------------------------------------------------------------------- #
# Model loading (lazy, cached)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def load_local_pipeline(model_id: str):
    """Load and cache the SD-Turbo pipeline (FP16 on CUDA, FP32 on CPU)."""
    require_packages("torch", "diffusers", "transformers", "accelerate")
    import torch
    from diffusers import AutoPipelineForText2Image

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    kwargs = {"torch_dtype": dtype, "use_safetensors": True}
    if use_cuda:
        kwargs["variant"] = "fp16"

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
    except OSError as exc:
        # Some checkpoints ship only .bin weights; retry without safetensors
        # rather than failing the whole run.
        if "safetensors" not in str(exc).lower():
            raise
        kwargs.pop("use_safetensors", None)
        pipe = AutoPipelineForText2Image.from_pretrained(model_id, **kwargs)
    pipe = pipe.to("cuda" if use_cuda else "cpu")
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()   # small VRAM / RAM footprint
    except Exception:
        pass
    return pipe


@st.cache_resource(show_spinner=False)
def load_clip():
    """Load CLIP on CPU in FP32 so GPU memory stays available for diffusion."""
    require_packages("torch", "transformers")
    import torch
    from transformers import CLIPModel, CLIPProcessor

    # Loaded in default FP32: no dtype kwarg, so this works across the
    # transformers 4.x/5.x API change (torch_dtype -> dtype) without warnings.
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
    model.to("cpu")
    model.eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    return model, processor


# --------------------------------------------------------------------------- #
# Generation backends
# --------------------------------------------------------------------------- #

def generate_local(
    prompt: str,
    steps: int,
    size: int,
    seed: int,
    *,
    pipe=None,
    model_id: str = SD_TURBO_MODEL_ID,
) -> tuple[object, float]:
    """Generate one image locally with Diffusers.

    ``pipe``/``model_id`` exist so tests can inject a small pipeline; the UI
    always uses the defaults (the cached SD-Turbo pipeline).
    """
    require_packages("torch", "diffusers", "transformers", "accelerate")
    import torch

    if pipe is None:
        pipe = load_local_pipeline(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    validate_requested_size(pipe, size)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    started = time.perf_counter()
    # Serialise the whole denoising loop. The cached pipeline is shared by all
    # sessions and its scheduler is mutable; concurrent runs would otherwise
    # reset each other's set_timesteps()/step_index mid-loop and raise
    # "IndexError: index 3 is out of bounds for dimension 0 with size 3".
    with _PIPELINE_LOCK:
        prepare_scheduler(pipe, steps, device)
        try:
            out = pipe(
                prompt=prompt,
                num_inference_steps=int(steps),
                guidance_scale=SD_TURBO_GUIDANCE,
                width=int(size),
                height=int(size),
                generator=generator,
            )
        except IndexError as exc:
            # A scheduler overrun is a state problem, not a bad prompt. Report it
            # as such instead of letting a bare IndexError reach the user.
            raise GenerationError(
                "The diffusion scheduler ran out of timesteps "
                f"(IndexError: {exc}). The shared pipeline's scheduler state was "
                "invalid: another generation was probably in progress. Retry; if "
                "it persists, restart the Streamlit server."
            ) from exc
        reset_scheduler_after_run(pipe, steps, device)
    elapsed = time.perf_counter() - started

    images = getattr(out, "images", None)
    if not images:
        raise GenerationError(
            "The pipeline returned no images (output.images was empty or missing)."
        )
    return to_rgb_pil(images[0], expected_size=size), elapsed


def generate_hosted(
    prompt: str, steps: int, size: int, seed: int, token: str
) -> tuple[object, float]:
    if not token:
        raise GenerationError(
            "Hosted mode needs a Hugging Face User Access Token with Inference "
            "Providers permission. Paste it in the sidebar or set HF_TOKEN. "
            "A GitHub token will not work here."
        )

    require_packages("huggingface_hub")
    from huggingface_hub import InferenceClient

    client = InferenceClient(model=SD3_MODEL_ID, token=token, timeout=180)

    started = time.perf_counter()
    try:
        image = client.text_to_image(
            prompt,
            width=int(size),
            height=int(size),
            num_inference_steps=int(steps),
            seed=int(seed),
        )
    except TypeError:
        # Some provider routes do not forward every kwarg; retry with the
        # required minimum and report the settings actually used.
        image = client.text_to_image(prompt, width=int(size), height=int(size))
    except Exception as exc:
        raise GenerationError(_explain_hosted_error(exc)) from exc
    elapsed = time.perf_counter() - started
    return to_rgb_pil(image, expected_size=size), elapsed


def _explain_hosted_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    if "401" in text or "unauthorized" in low or "invalid token" in low:
        return (
            "Hugging Face rejected the token (401). Check that it is an HF User "
            f"Access Token with Inference Providers permission. Details: {text}"
        )
    if "403" in text or "forbidden" in low or "gated" in low:
        return (
            "Access forbidden (403). Accept the model terms on the model page "
            f"and confirm the token's permissions. Details: {text}"
        )
    if "429" in text or "rate limit" in low:
        return (
            "Rate limited or out of credits (429). Hosted inference is "
            f"usage-metered — check your billing/credits. Details: {text}"
        )
    return f"Hosted generation failed: {text}"


# --------------------------------------------------------------------------- #
# CLIP evaluation
# --------------------------------------------------------------------------- #

def clip_similarity(prompt: str, image) -> float:
    """Raw cosine similarity between L2-normalised CLIP text/image embeddings."""
    require_packages("torch", "transformers")
    import torch

    model, processor = load_clip()
    rgb = to_rgb_pil(image)
    inputs = processor(
        text=[prompt],
        images=rgb,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    if "pixel_values" not in inputs:
        raise ScoringError(
            "CLIP preprocessing produced no pixel_values; the image could not be "
            f"read (got {type(image).__name__})."
        )
    pixel_shape = tuple(inputs["pixel_values"].shape)
    if len(pixel_shape) != 4 or pixel_shape[1] != 3:
        raise ScoringError(
            f"CLIP expected pixel_values shaped (batch, 3, height, width) but got "
            f"{pixel_shape}. A CHW/HWC or RGBA/RGB mismatch would cause this."
        )

    with torch.no_grad():
        # One forward pass returns the *projected* embeddings in the shared
        # space (text_embeds / image_embeds). This is stable across the
        # transformers 4.x and 5.x APIs, unlike get_text_features(), whose
        # return type changed.
        outputs = model(**inputs)
        text_emb = getattr(outputs, "text_embeds", None)
        image_emb = getattr(outputs, "image_embeds", None)

    if text_emb is None or image_emb is None:
        raise ScoringError(
            "CLIP returned no projected embeddings (text_embeds/image_embeds "
            "missing from the model output)."
        )
    if text_emb.ndim != 2 or image_emb.ndim != 2:
        raise ScoringError(
            "CLIP embeddings must be 2-D (batch, dim) but got "
            f"{tuple(text_emb.shape)} and {tuple(image_emb.shape)}."
        )
    if text_emb.shape[0] != 1 or image_emb.shape[0] != 1:
        raise ScoringError(
            f"Expected a batch of 1 prompt/image, got {text_emb.shape[0]} text and "
            f"{image_emb.shape[0]} image embeddings."
        )
    if text_emb.shape[-1] != image_emb.shape[-1]:
        raise ScoringError(
            f"CLIP embedding dimensions differ: text {text_emb.shape[-1]} vs image "
            f"{image_emb.shape[-1]}."
        )
    if not torch.isfinite(text_emb).all() or not torch.isfinite(image_emb).all():
        raise ScoringError("CLIP produced non-finite embeddings (NaN or inf).")

    text_norm = text_emb.norm(p=2, dim=-1, keepdim=True)
    image_norm = image_emb.norm(p=2, dim=-1, keepdim=True)
    if float(text_norm.min()) == 0.0 or float(image_norm.min()) == 0.0:
        raise ScoringError(
            "Cannot normalise a zero-length CLIP embedding; cosine similarity is "
            "undefined for this image/prompt pair."
        )

    cosine = ((text_emb / text_norm) * (image_emb / image_norm)).sum(dim=-1)
    value = float(cosine.item())
    if not (-1.0 - 1e-4 <= value <= 1.0 + 1e-4):
        raise ScoringError(
            f"Cosine similarity {value:.6f} is outside the mathematical range "
            "[-1, 1]; the embeddings or normalization are wrong."
        )
    return value


def evaluate(result: Result) -> None:
    """Fill in the CLIP score, or record why it is unavailable."""
    try:
        result.clip_score = clip_similarity(result.prompt, result.image)
        result.clip_error = None
    except Exception as exc:
        result.clip_score = None
        result.clip_error = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# One end-to-end run
# --------------------------------------------------------------------------- #

def run_pipeline(
    prompt: str, backend: str, steps: int, size: int, seed: int, token: str
) -> Result:
    if backend == BACKEND_LOCAL:
        image, elapsed = generate_local(prompt, steps, size, seed)
        model_id = SD_TURBO_MODEL_ID
    else:
        image, elapsed = generate_hosted(prompt, steps, size, seed, token)
        model_id = SD3_MODEL_ID

    result = Result(
        prompt=prompt,
        image=image,
        backend=backend,
        model_id=model_id,
        seed=seed,
        width=image.width,
        height=image.height,
        steps=steps,
        gen_seconds=elapsed,
    )
    evaluate(result)
    return result


# --------------------------------------------------------------------------- #
# UI pieces
# --------------------------------------------------------------------------- #

def render_result(result: Result, key: str, show_prompt: bool = True) -> None:
    st.image(
        result.image,
        caption=f"Generated at {result.timestamp} · {result.settings_text()}",
        width="stretch",
    )
    if show_prompt:
        st.markdown(f"**Prompt:** {result.prompt}")
    st.caption(
        f"Backend: {result.backend} · model: `{result.model_id}` · "
        f"generation time: {result.gen_seconds:.2f} s"
    )

    if result.clip_score is not None:
        st.metric("CLIP cosine similarity", f"{result.clip_score:+.4f}")
        st.caption(
            "Raw cosine similarity between L2-normalised CLIP embeddings. "
            "Bounded by −1 and 1. Not a probability or a correctness percentage."
        )
    else:
        st.warning(
            "CLIP score unavailable — scoring failed for this image. "
            f"Details: {result.clip_error}"
        )

    st.download_button(
        "Download PNG",
        data=result.png_bytes(),
        file_name=f"generated_{result.seed}.png",
        mime="image/png",
        key=f"dl_{key}",
    )


def sidebar() -> dict:
    st.sidebar.header("Settings")

    backend = st.sidebar.radio(
        "Image generation backend",
        [BACKEND_LOCAL, BACKEND_HOSTED],
        help="Local avoids per-image fees but needs the weights and your hardware. "
        "Hosted sends the prompt to a Hugging Face provider and is usage-metered.",
    )

    model_id = SD_TURBO_MODEL_ID if backend == BACKEND_LOCAL else SD3_MODEL_ID
    st.sidebar.caption(f"Model: `{model_id}`")

    steps = st.sidebar.slider(
        "Denoising steps",
        min_value=ALLOWED_STEPS[0],
        max_value=ALLOWED_STEPS[-1],
        value=DEFAULT_STEPS,
        help="SD-Turbo is distilled for 1-4 steps only.",
    )
    size = st.sidebar.selectbox(
        "Image size (px, square)",
        ALLOWED_SIZES,
        index=ALLOWED_SIZES.index(DEFAULT_SIZE),
    )
    st.session_state.setdefault("seed_box", 1234)
    seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=2**31 - 1,
        step=1,
        key="seed_box",
        help="Same seed + same settings + same model reproduces the image.",
    )
    def _randomise_seed() -> None:
        # Must run as an on_click callback: assigning to a widget's key after
        # the widget is instantiated raises StreamlitWidgetAlreadyInstantiatedError.
        st.session_state["seed_box"] = random.randint(0, 2**31 - 1)

    st.sidebar.button("Randomise seed", on_click=_randomise_seed)

    token = ""
    if backend == BACKEND_HOSTED:
        token = st.sidebar.text_input(
            "Hugging Face token",
            type="password",
            value=os.environ.get("HF_TOKEN", ""),
            help="Needs Inference Providers permission. Never commit it.",
        )

    with st.sidebar.expander("Environment status"):
        presence = package_presence()
        for module, ok in presence.items():
            st.write(f"{'✅' if ok else '❌'} `{REQUIRED_PACKAGES[module]}`")
        info = torch_report()
        if info["ok"]:
            st.write(f"PyTorch: `{info['version']}`")
            st.write(f"CUDA available: `{info['cuda']}`")
            st.write(f"Device: `{info['device']}`")
            if not info["cuda"]:
                st.caption("CPU inference works but can be very slow.")
        else:
            st.error(f"PyTorch not importable: {info['error']}")

    st.sidebar.divider()
    st.sidebar.caption(
        "Images and scores shown are real runtime outputs. Nothing is "
        "pre-bundled or fabricated; failures are reported as errors."
    )

    return {
        "backend": backend,
        "model_id": model_id,
        "steps": steps,
        "size": size,
        "seed": int(seed),
        "token": token,
    }


def tab_generate(cfg: dict) -> None:
    st.subheader("Generate an image and score it with CLIP")

    if "prompt" not in st.session_state:
        st.session_state["prompt"] = SAMPLE_PROMPTS[0]

    cols = st.columns([3, 1])
    with cols[1]:
        st.write("Sample prompts")
        for i, sample in enumerate(SAMPLE_PROMPTS):
            if st.button(f"Sample {i + 1}", key=f"sample_{i}", width="stretch"):
                st.session_state["prompt"] = sample
                st.rerun()

    prompt = st.text_area(
        "Prompt", key="prompt", height=110, help="Ctrl+Enter does not submit here."
    )

    if st.button("Generate image and calculate CLIP score", type="primary"):
        if not prompt.strip():
            st.error("Enter a prompt first.")
            return
        try:
            with st.spinner("Generating image..."):
                result = run_pipeline(
                    prompt.strip(),
                    cfg["backend"],
                    cfg["steps"],
                    cfg["size"],
                    cfg["seed"],
                    cfg["token"],
                )
        except GenerationError as exc:
            st.error(f"Generation failed. {exc}")
            return
        except Exception as exc:  # unexpected runtime failure
            st.error(f"Generation failed ({type(exc).__name__}): {exc}")
            return

        add_result(result)
        st.session_state["last_result"] = result

    last = st.session_state.get("last_result")
    if last is not None:
        st.divider()
        left, right = st.columns([1, 1])
        with left:
            render_result(last, key="last")
        with right:
            st.markdown("#### What was measured")
            st.markdown(
                "- Input: the exact prompt shown above.\n"
                "- Image: the returned PIL image, converted to RGB for CLIP.\n"
                "- Evaluator: `openai/clip-vit-base-patch32` on CPU.\n"
                "- Metric: raw cosine similarity of L2-normalised embeddings."
            )
            if last.clip_score is not None:
                st.markdown(
                    f"Observed score: **{last.clip_score:+.4f}**. A higher value is "
                    "evidence of closer embedding alignment, not proof that every "
                    "prompt detail is satisfied."
                )


def tab_compare(cfg: dict) -> None:
    st.subheader("Compare two prompts")
    st.caption(
        "Both images use the same backend, seed, size and step count so the "
        "comparison is controlled."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        prompt_a = st.text_area("Prompt A", value=SAMPLE_PROMPTS[0], height=110)
    with col_b:
        prompt_b = st.text_area("Prompt B", value=SAMPLE_PROMPTS[1], height=110)

    if st.button("Generate both images and compare", type="primary"):
        if not prompt_a.strip() or not prompt_b.strip():
            st.error("Both prompts must be filled in.")
            return

        outcomes: list[Result | None] = []
        for label, text in (("A", prompt_a), ("B", prompt_b)):
            try:
                with st.spinner(f"Generating image {label}..."):
                    result = run_pipeline(
                        text.strip(),
                        cfg["backend"],
                        cfg["steps"],
                        cfg["size"],
                        cfg["seed"],
                        cfg["token"],
                    )
            except GenerationError as exc:
                st.error(f"Generation failed for prompt {label}. {exc}")
                outcomes.append(None)
                continue
            except Exception as exc:
                st.error(f"Generation failed for prompt {label} ({type(exc).__name__}): {exc}")
                outcomes.append(None)
                continue
            add_result(result)
            outcomes.append(result)

        st.session_state["compare_pair"] = outcomes

    pair = st.session_state.get("compare_pair")
    if pair:
        st.divider()
        left, right = st.columns(2)
        for column, label, result in zip((left, right), ("A", "B"), pair):
            with column:
                st.markdown(f"### Prompt {label}")
                if result is None:
                    st.warning("No image — generation failed for this prompt.")
                else:
                    render_result(result, key=f"cmp_{label}")

        if all(r is not None for r in pair) and all(
            r.clip_score is not None for r in pair
        ):
            a, b = pair
            delta = a.clip_score - b.clip_score
            st.markdown("#### Comparison")
            st.write(
                f"Score A = **{a.clip_score:+.4f}**, Score B = **{b.clip_score:+.4f}**, "
                f"difference (A − B) = **{delta:+.4f}**."
            )
            st.caption(
                "A larger score only means closer CLIP embedding alignment under "
                "this exact checkpoint and preprocessing. It is not a correctness "
                "verdict — check the images yourself."
            )
        elif pair:
            st.info("Scores are not comparable because at least one is unavailable.")


def tab_dashboard() -> None:
    st.subheader("Results dashboard")
    results = get_results()

    if not results:
        st.info(
            "No results yet in this session. Generate an image on the Generate tab "
            "(or Compare prompts) to populate the dashboard."
        )
        return

    st.caption(
        f"{len(results)} result(s) in this session (latest {MAX_RESULTS} kept). "
        "Session-local only — nothing is persisted to a database."
    )

    rows = [
        {
            "Time": r.timestamp,
            "Prompt": r.prompt if len(r.prompt) <= 60 else r.prompt[:57] + "...",
            "Backend": "Local" if r.backend == BACKEND_LOCAL else "Hosted",
            "Model": r.model_id.split("/")[-1],
            "Settings": r.settings_text(),
            "Gen (s)": round(r.gen_seconds, 2),
            "CLIP score": None if r.clip_score is None else round(r.clip_score, 4),
        }
        for r in results
    ]
    st.dataframe(rows, width="stretch", hide_index=True)

    scored = [r for r in results if r.clip_score is not None]
    if scored:
        st.markdown("#### CLIP cosine similarity per result")
        import pandas as pd

        chart_rows = [
            {
                "Result": f"#{i + 1} · {r.prompt[:32]}"
                + ("..." if len(r.prompt) > 32 else ""),
                "CLIP cosine similarity": r.clip_score,
            }
            for i, r in enumerate(reversed(scored))
        ]
        st.bar_chart(
            pd.DataFrame(chart_rows),
            x="Result",
            y="CLIP cosine similarity",
            height=320,
        )
        scores = [r.clip_score for r in scored]
        c1, c2, c3 = st.columns(3)
        c1.metric("Scored results", len(scores))
        c2.metric("Mean score", f"{sum(scores) / len(scores):+.4f}")
        c3.metric("Best score", f"{max(scores):+.4f}")
        st.caption(
            "Cosine similarity is bounded by −1 and 1. It is an embedding-space "
            "signal, not a probability, a percentage, or a guarantee of correctness."
        )
    else:
        st.warning("No successful CLIP scores yet — nothing to chart.")

    st.divider()
    st.markdown("#### Downloads and entries")
    for index, result in enumerate(results):
        with st.container(border=True):
            cols = st.columns([1, 3])
            with cols[0]:
                st.image(result.image, width=140)
            with cols[1]:
                st.markdown(f"**{result.prompt}**")
                st.caption(
                    f"{result.timestamp} · {result.backend} · `{result.model_id}` · "
                    f"{result.settings_text()} · {result.gen_seconds:.2f} s"
                )
                if result.clip_score is not None:
                    st.write(f"CLIP cosine similarity: `{result.clip_score:+.4f}`")
                else:
                    st.write("CLIP cosine similarity: unavailable — " + str(result.clip_error))
                st.download_button(
                    "Download PNG",
                    data=result.png_bytes(),
                    file_name=f"generated_{result.timestamp.replace(':', '')}_{index}.png",
                    mime="image/png",
                    key=f"dash_dl_{index}",
                )

    if st.button("Clear session results"):
        st.session_state["results"] = []
        st.session_state.pop("last_result", None)
        st.session_state.pop("compare_pair", None)
        st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(
        page_title="Text-to-Image + CLIP Evaluation",
        page_icon="🖼️",
        layout="wide",
    )

    st.title("Text-to-Image Generation and Semantic Evaluation using Generative AI")
    st.caption(
        "Multimodal Generative AI Laboratory · Diffusers text-to-image generation + "
        "local CLIP cosine similarity"
    )

    cfg = sidebar()

    tabs = st.tabs(["Generate", "Compare prompts", "Results dashboard"])
    with tabs[0]:
        tab_generate(cfg)
    with tabs[1]:
        tab_compare(cfg)
    with tabs[2]:
        tab_dashboard()

    with st.sidebar.expander("Theory and caveats"):
        st.markdown(
            "**Diffusion.** Training adds noise at many timesteps; generation starts "
            "from noise and denoises under text conditioning. Steps = number of "
            "denoising updates, so more steps usually cost more time.\n\n"
            "**CLIP.** Separate text and image encoders map into one shared space. "
            "After L2 normalisation, the dot product is the cosine similarity.\n\n"
            "**Not a probability.** The score is uncalibrated and can be affected by "
            "wording, style, preprocessing, subject prominence, model bias and "
            "out-of-distribution content."
        )


if __name__ == "__main__":
    main()
