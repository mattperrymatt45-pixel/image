# Text-to-Image Generation and Semantic Evaluation using Generative AI

**Multimodal Generative AI Laboratory**

A small, working Streamlit application that generates images from natural-language prompts and
calculates a raw CLIP text–image cosine similarity score. It supports **local Diffusers** generation
and an optional **hosted Hugging Face Inference Providers** backend. Generated images and scores are
real runtime outputs; the app never substitutes a random image or a fabricated score.

---

## Experiment title and aim

**Title:** Text-to-Image Generation and Semantic Evaluation using Generative AI

**Aim:** Generate an image from a text prompt using a pretrained diffusion model, then measure the
embedding-space similarity between the prompt and generated image using a pretrained CLIP model.

---

## Implementation choice: local first, hosted as an alternative

For a no-per-image-fee laboratory demo, the default is local inference:

- **Image generation:** Diffusers with `stabilityai/sd-turbo`, a distilled diffusion model that is
  intended to work in 1–4 denoising steps. The app restricts the step control to 1–4 and the size
  controls to 256, 384, or 512 pixels; 512×512 is the model-card recommendation.
- **Semantic evaluation:** Transformers + PyTorch with `openai/clip-vit-base-patch32`. CLIP runs
  locally on CPU in either backend to avoid using extra GPU memory.
- **Cost:** No hosted image-generation charge and no API key in local mode. Model downloads consume
  disk/network bandwidth; local compute consumes your own hardware and electricity.

**Hardware:** An NVIDIA GPU with about 8 GB or more VRAM and 16 GB system RAM is a practical target,
not a guarantee. CPU inference is supported but can be very slow; allow at least 16 GB system RAM and
several GB of free disk space. The first local run downloads several GB of model weights plus the CLIP
checkpoint. Exact memory and download sizes vary with model files, package versions, and device.

If local inference is too slow or the computer has no suitable GPU, select **Hosted Hugging Face
Inference API** in the sidebar:

- Image generation is remote through Hugging Face Inference Providers with
  `stabilityai/stable-diffusion-3-medium-diffusers`; the prompt and generation parameters are sent to
  the provider.
- CLIP scoring remains local and still needs PyTorch, Transformers, and the CLIP model download.
- A Hugging Face User Access Token with **Inference Providers** permission is required. Accept any
  model terms if the model page prompts you. The token can be entered into the masked sidebar field or
  set as `HF_TOKEN`.
- Hosted inference is usage-metered and may require paid credits. Prices, credits, limits, and
  provider/model availability change; check the current Hugging Face billing page before use. Do not
  assume hosted calls are free.

A GitHub token is not an HF token. Never put either token in source files or commit it to Git.

The project keeps dependencies and source files to a small set: one Python app, one requirements file,
and this documentation. Streamlit's built-in chart is used, so no separate plotting library is required.

---

## Theory

### Diffusion models

A diffusion model is trained to reverse a gradual noising process. During training, noise is added to
training images at different timesteps and the network learns to estimate/remove that noise. At
generation time, a text-conditioned model starts from a noise tensor and repeatedly denoises it, guided
by the text representation, until an image is decoded. Inference steps control the number of denoising
updates: more steps can improve results for many diffusion models but cost more time. SD-Turbo is
distilled for a small number of steps, so this app limits it to the model's 1–4-step range.

### CLIP and cosine similarity

CLIP has an image encoder and a text encoder trained to map images and text into a shared embedding
space. This app processes the generated image and its exact prompt, obtains the projected embeddings,
L2-normalizes them, and calculates:

```
cosine_similarity = normalized_text_embedding · normalized_image_embedding
```

The raw cosine value is bounded by −1 and 1. It is an **embedding similarity** — not a probability, a
percent-correct, or a guarantee that the image satisfies every prompt detail. It can be affected by
wording, style, image preprocessing, subject prominence, model bias, and content outside CLIP's training
distribution. The app reports the raw similarity and does not apply softmax or claim a correctness
probability.

---

## System architecture

```
Prompt + settings
       |
       +--> Local mode: Streamlit --> Diffusers / SD-Turbo on this computer --+
       |                                                                     |
       +--> Hosted mode: Streamlit --> HF Inference Providers / SD 3 Medium -+--> generated image
                                                                             |
Prompt + generated image --> Transformers / CLIP (local CPU) --> normalized embeddings
                                                                  --> raw cosine similarity
                                                                             |
                                      Streamlit results, comparison chart, PNG download
```

Small codebase:

| File | Purpose |
| --- | --- |
| `app.py` | Contains the full application source. |
| `requirements.txt` | Lists installable dependencies. |
| `.gitignore` | Excludes virtual environments, local secrets, and model/output caches. |
| `test_pipeline.py` | Self-check: real tiny-pipeline generation + real CLIP scoring + regression tests. |

---

## Algorithm and workflow

1. Read one prompt or a pair of prompts and the selected backend/settings.
2. Check that the Python packages required by that backend are importable.
3. For local mode, lazily load and cache SD-Turbo through Diffusers; use FP16 on CUDA and FP32 on CPU.
   For hosted mode, call the configured Hugging Face Inference Provider with the user's HF token.
4. Generate one image for each prompt using the chosen step count, dimensions, and seed.
5. Pass the original prompt and generated image through the pretrained CLIP processor/model.
6. L2-normalize both projected embeddings and compute their dot product, yielding raw cosine similarity.
7. Store the latest results in the Streamlit session, render images/prompts/settings/scores, plot
   scored results, and offer PNG downloads.
8. If a generation or evaluation step fails, show a clear error or mark the score unavailable. Do not
   use a replacement image or value.

> **Note on concurrency and shape validation.** The Diffusers pipeline is cached once per server
> process (`@st.cache_resource`) and shared by every Streamlit session, and its scheduler is
> stateful (`timesteps`, `sigmas`, `step_index`). Generation is therefore serialised with a
> `threading.Lock`, and the scheduler is put into a verified state before each run
> (`len(timesteps) == steps`, `len(sigmas) == steps + 1`) and repaired afterwards. Generator output
> is coerced through `to_rgb_pil()`, which accepts PIL images or arrays in CHW/HWC layout, with or
> without a batch dimension, and 1/3/4 channels (RGBA is converted, not indexed at channel 3).
> CLIP scoring validates that `pixel_values` is `(batch, 3, H, W)`, that the two embeddings are
> 2-D, same-dimension, non-degenerate and finite, and that the result lies in [-1, 1].

> **Note on the CLIP call.** The app uses a single forward pass
> (`outputs = model(**inputs)`, then `outputs.text_embeds` / `outputs.image_embeds`) rather than
> `get_text_features()` / `get_image_features()`, because the return type of those helpers changed
> between Transformers 4.x and 5.x. The forward pass returns the same projected embeddings on both.

---

## Setup and execution on Windows (PowerShell + VS Code)

### 1. Install Python and open the project

Install 64-bit Python 3.11 (Python 3.10+ is supported) and VS Code with the Microsoft Python
extension. Open this folder in VS Code using **File → Open Folder**. In VS Code, open **Terminal → New
Terminal** and select PowerShell.

### 2. Create and activate a virtual environment

Run from the folder containing `app.py`:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

The execution-policy command only changes the current PowerShell process. If Python 3.11 is not
installed, use an installed supported version, for example `py -3.12 -m venv .venv`.

### 3. Install PyTorch and the project dependencies

For a simple CPU-compatible install, run:

```powershell
python -m pip install torch
python -m pip install -r requirements.txt
```

This is the easiest path and supports CPU inference. If you have an NVIDIA GPU and want CUDA
acceleration, install the CUDA-enabled PyTorch build selected for your Windows driver using the official
PyTorch install selector **before** installing the remaining requirements. Do not guess a CUDA wheel
version if the selector specifies a different one. Then run:

```powershell
python -m pip install -r requirements.txt
```

Check which device PyTorch can see with:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 4. Run the app

```powershell
python -m streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) and opens the app in a browser. In
VS Code, select the `.venv` interpreter using **Ctrl+Shift+P → Python: Select Interpreter** so the
editor and terminal use the same environment.

---

## Hosted-mode token setup (optional)

Create a Hugging Face User Access Token with the **Inference Providers** permission. Either paste it into
the password-masked field in the app sidebar, or set it only for the current PowerShell window before
launching:

```powershell
$env:HF_TOKEN = Read-Host "Paste Hugging Face token"
python -m streamlit run app.py
```

Do not commit a token, add it to `app.py`, or reuse a GitHub token. For local mode, no HF token is
needed. Review current usage prices/credits before making hosted calls.

---

## Complete workflow test

1. Launch the app and keep **Local Diffusers (SD-Turbo)** selected for a local run.
2. In **Generate**, use the pre-filled sample prompt or type your own. Leave the default 512×512 size
   and choose 1–4 steps.
3. Click **Generate image and calculate CLIP score**. The first local run downloads the model weights
   and can take a while; subsequent runs reuse the cached pipeline. Wait for both generation and CLIP
   evaluation to complete.
4. Confirm the page shows the generated image, exact prompt, inference settings, and a numeric CLIP
   cosine similarity. The score is computed from actual embeddings. If scoring fails, the app says it is
   unavailable rather than inventing a value.
5. Open **Compare prompts**, enter two prompts, and click **Generate both images and compare**. Both
   images, prompts, settings, and scores appear side by side with the same seed/settings.
6. Open **Results dashboard** to inspect saved session results and the comparison bar chart. Download
   any generated image with its PNG button.
7. To test hosted mode, select it in the sidebar, provide an HF token, and run the same flow. Check
   billing/credits before calling the provider.

---

## Self-check / tests

```powershell
python test_pipeline.py
```

This runs 12 checks with **real** model code - nothing is stubbed or faked:

- **Real diffusion generation** through `app.generate_local()` using the tiny
  `hf-internal-testing/tiny-sd-pipe` checkpoint (a few MB, CPU-friendly). Prompt encoding, latent
  preparation, the scheduler loop, VAE decode and postprocessing all execute.
- **Concurrency**: 2-step and 4-step generations run simultaneously on one shared pipeline - the
  situation that produced the scheduler `IndexError` - and both must succeed.
- **Lock regression**: two threads entering the pipeline must never overlap (max concurrent
  entries = 1).
- **Real CLIP scoring** of a real image; the score must be finite and within [-1, 1].
- **Shape validation**: CHW, HWC, batched, RGBA, grayscale and malformed arrays, plus the scheduler
  overrun reproduction and its repair.

Expected output ends with `12/12 checks passed`.

---

## Sample input prompts

1. A small red bicycle parked beside a blue brick wall on a rainy city street at dusk, editorial
   photograph.
2. A small red bicycle leaning against a blue brick wall in a quiet city alley at sunrise, photograph.
3. A yellow kayak floating on a calm mountain lake at sunrise, watercolor illustration.

---

## Actual output examples

No model inference was run while packaging this project, so **no images or numeric similarity values
are bundled or presented as observed results**. When you run the workflow, the dashboard shows the
actual image and measured score for that session. Record those runtime values in your lab submission
rather than copying an invented example.

---

## Evaluation methodology

- **Input:** the exact text prompt submitted to the image generator.
- **Image:** the returned PIL image, converted to RGB for CLIP preprocessing.
- **Evaluator:** `openai/clip-vit-base-patch32` from Transformers.
- **Metric:** raw cosine similarity between L2-normalized text and image embeddings.
- **Comparison:** compare scores cautiously only under a consistent CLIP checkpoint, preprocessing path,
  and comparable prompts/settings. A higher score is evidence of closer embedding alignment, not proof
  of correctness.
- **No fabricated data:** generation failures and scoring failures are exposed in the interface. Results
  are session-local and are not silently substituted.

---

## Results and observations

The application records the prompt, generated image, backend/model, seed, dimensions, inference steps,
runtime, and CLIP score. The dashboard plots scores for successful evaluations. SD-Turbo is optimized
for very few steps and is suitable for a quick lab demonstration, but image fidelity and prompt
alignment can be lower than larger, slower diffusion models. CLIP gives a useful semantic signal but can
miss fine details, negation, spatial relations, text rendered in images, or subtle composition
differences.

For a report, record at least two actual prompts, the chosen backend/settings, the generated outputs,
the observed raw cosine scores, and whether the score ranking agrees with visual inspection. **Do not
report a score as a probability.**

---

## Limitations

- Local generation can require several GB of disk, substantial RAM, and a CUDA-capable GPU for a
  responsive experience. CPU generation is supported but slow.
- Hosted generation requires a separate HF token, internet access, an available provider/model route,
  and possibly paid credits. Availability, limits, and cost may change.
- Hosted mode sends prompts to the provider; avoid sensitive prompts. CLIP still downloads/runs locally.
- SD-Turbo has lower quality/prompt alignment than larger models, has known difficulty rendering legible
  text and some faces, and is trained for a small step count and a 512×512 target.
- CLIP similarity is not calibrated as a correctness score and does not evaluate every detail or
  aesthetic property. Bias and training-data limits apply.
- The dashboard retains only the latest 12 outputs in the current Streamlit session. It does not provide
  persistent database storage.

Read model licenses/terms before using either checkpoint outside a teaching demonstration.

---

## Conclusion

This project demonstrates a full text-to-image workflow with pretrained generative and multimodal
models: prompt entry, diffusion inference, CLIP embedding extraction, cosine similarity, comparison,
visualization, and image download. It provides a local no-per-image-fee path and clearly separates it
from an optional hosted, usage-metered path. The measured score is an interpretable research signal, not
a guarantee of image correctness.

---

## Viva questions and answers

**What is the purpose of the denoising steps in a diffusion model?**
Each step updates a noisy latent toward a sample that better matches the learned image distribution and
text conditioning. More steps often cost more compute; distilled models such as SD-Turbo are designed to
use fewer.

**Why is SD-Turbo limited to 1–4 inference steps in this app?**
The pretrained checkpoint is distilled for very low-step inference. The interface constrains the
parameter to the documented range instead of implying that arbitrary step counts are appropriate.

**How does CLIP compare a prompt and image?**
Separate encoders map text and image into a common embedding space. After L2 normalization, their dot
product is cosine similarity; nearby embeddings indicate stronger learned association.

**Why is the CLIP score not a probability or correctness percentage?**
It is an uncalibrated geometric similarity in a learned representation space. It is not normalized over a
set of alternatives and does not certify that every object, attribute, or relation is correct.

**What is the difference between local and hosted inference in this project?**
Local mode downloads SD-Turbo and runs generation on the lab computer, using its hardware and no
per-image hosted API call. Hosted mode sends the prompt to a remote Hugging Face provider and may be
billed; CLIP evaluation remains local in both modes.

---

## Troubleshooting

- **Missing dependency message:** activate `.venv`, rerun `python -m pip install -r requirements.txt`,
  then restart Streamlit.
- **CUDA is unavailable:** the app runs locally on CPU; this may be slow. If you have an NVIDIA GPU,
  reinstall the PyTorch build recommended by the official selector and re-check
  `torch.cuda.is_available()`.
- **Out-of-memory error:** close GPU applications, use 512×512 or smaller, or select the hosted
  generator. The app does not silently retry with a different image.
- **Hugging Face 401/403:** check that the token is an HF token with Inference Providers permission and
  that you accepted any model terms. Never use a GitHub PAT in that field.
- **Hosted billing/429/provider error:** review current credits, rate limits, provider availability, and
  account billing; try later or use local mode.
- **`Generation failed (IndexError): index 3 is out of bounds for dimension 0 with size 3`:** the
  diffusion scheduler was stepped more times than its schedule allows. A scheduler configured for
  N steps holds N+1 sigma values, so the default 2 steps give 3 sigmas and one extra step asks for
  `sigmas[3]`. This is scheduler *state corruption*, caused by two generations sharing the single
  cached pipeline at the same time (two browser tabs, two students, or a rerun during generation).
  The app now serialises every pipeline call with a lock, verifies the scheduler state before each
  run and repairs it afterwards. If it still appears, restart the Streamlit server to discard the
  cached pipeline.
- **First run appears slow:** model downloads are large. Keep the terminal open and wait for the
  progress/error message. A failed request is surfaced, not replaced by a placeholder.

---

## Repository history note

This repository previously contained a single-file `index.html` demo that showed fixed SVG images
("Demo image · not AI-generated") alongside a keyword-overlap score. It performed no diffusion or CLIP
inference, so it has been **removed** and replaced by this real Streamlit + Diffusers + CLIP
implementation, in line with the requirement that images and scores must be genuine runtime outputs.
