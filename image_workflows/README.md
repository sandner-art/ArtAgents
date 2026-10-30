# Image workflows

ArtAgents can generate images through three local backends (Image Generation tab):

| Backend | Default URL | Notes |
|---|---|---|
| **ComfyUI** | `http://127.0.0.1:8188` | Runs the workflow files in `image_workflows/comfyui/`. Any model ComfyUI can run (SD 1.5, SDXL, FLUX, Z-Image, LoRAs, ControlNet…). |
| **Forge / A1111 / SD.Next** | `http://127.0.0.1:7860` | Start the WebUI with `--api`. ArtAgents also defaults to port 7860, so whichever starts second gets 7861. Check the URL. |
| **Ollama (experimental)** | `http://localhost:11434` | Uses `/v1/images/generations`. Ollama removed its experimental image generation in v0.32.6, so current versions refuse. It works again if Ollama restores it, and also with other OpenAI-compatible image servers. |

## ComfyUI workflows

Workflows must be in ComfyUI's **API format**, not the normal saved workflow:

1. Build or open the workflow in ComfyUI and check that it runs.
2. Export it with **Workflow → Export (API)**. In older versions, enable *Dev mode* in the settings and use **Save (API Format)**.
3. Put the `.json` file in `image_workflows/comfyui/`. It then appears in the *Workflow* dropdown after you click *Refresh*.

A normal (UI-format) workflow file is rejected with a message that tells you to re-export it.

### Placeholders (recommended)

Type these into the workflow's widget values before exporting. ArtAgents replaces them on every run:

| Placeholder | Replaced with |
|---|---|
| `%prompt%` | Prompt text. It can be part of a longer text, e.g. `photo of %prompt%, film grain`. |
| `%negative%` | Negative prompt, which can also be embedded. |
| `%seed%` | Seed. A random one is used when the UI seed is -1. |
| `%width%`, `%height%` | Image size |
| `%steps%`, `%cfg%` | Sampler steps / CFG. Defaults are 20 / 7.0 when left empty in the UI. |
| `%sampler%` | Sampler name (default `euler`) |
| `%batch_size%` | Images per run |
| `%checkpoint%` | Checkpoint chosen in the Model dropdown (required when used) |

Numeric placeholders must be the *whole* value. ComfyUI gets real numbers, not strings. `basic_txt2img.json` is a complete example (checkpoint → prompts → KSampler → SaveImage).

### Automatic detection (no placeholders)

If a workflow contains no `%prompt%`, ArtAgents tries to find the right nodes:

- **Sampler**: `KSampler`, `KSamplerAdvanced`, `SamplerCustom`, and `SamplerCustomAdvanced` (via its guider, noise, sigmas and sampler nodes). The seed, steps, CFG and sampler name are set there.
- **Prompt**: it follows the sampler's *positive*/*negative* inputs back to the text encoder: `CLIPTextEncode`, `CLIPTextEncodeSDXL` (`text_g`/`text_l`), `CLIPTextEncodeFlux` (`clip_l`/`t5xxl`), and others. It passes through nodes like `FluxGuidance` or `ConditioningCombine`. A `ConditioningZeroOut` negative is left alone.
- **Size**: any `Empty…Latent…` node (`width`, `height`, `batch_size`).
- **Checkpoint**: `CheckpointLoaderSimple` (`ckpt_name`) when a model is selected.

Values left empty in the UI (steps, CFG, sampler, model) keep the workflow's own settings. In this mode an empty negative prompt also keeps the workflow's negative text. Use placeholders if detection picks the wrong node, for example in workflows with several samplers.

The workflow must end in a `SaveImage` (or `PreviewImage`) node. ArtAgents downloads those images and saves its own copy in the output folder set in the Image Generation tab, in a `<date>/` subfolder. The prompt is embedded in the PNG, and an optional `.txt` caption is saved next to it.
