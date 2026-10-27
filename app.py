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

def generate_local(prompt: str, steps: int, size: int, seed: int) -> tuple[object, float]:
    require_packages("torch", "diffusers", "transformers", "accelerate")
    import torch

    pipe = load_local_pipeline(SD_TURBO_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(int(seed))

    started = time.perf_counter()
    out = pipe(
        prompt=prompt,
        num_inference_steps=int(steps),
        guidance_scale=SD_TURBO_GUIDANCE,
        width=int(size),
        height=int(size),
        generator=generator,
    )
    elapsed = time.perf_counter() - started
    return out.images[0].convert("RGB"), elapsed


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
    return image.convert("RGB"), elapsed


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
    inputs = processor(
        text=[prompt],
        images=image.convert("RGB"),
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    with torch.no_grad():
        # One forward pass returns the *projected* embeddings in the shared
        # space (text_embeds / image_embeds). This is stable across the
        # transformers 4.x and 5.x APIs, unlike get_text_features(), whose
        # return type changed.
        outputs = model(**inputs)
        text_emb = outputs.text_embeds
        image_emb = outputs.image_embeds

    text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
    image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
    cosine = (text_emb * image_emb).sum(dim=-1)
    return float(cosine.item())


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
    if st.sidebar.button("Randomise seed"):
        st.session_state["seed_box"] = random.randint(0, 2**31 - 1)
        st.rerun()

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
