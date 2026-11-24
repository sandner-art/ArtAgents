# Training Captions: Guide

The **Image Captions Editor** tab prepares caption files for LoRA and fine-tune training
(kohya-ss / sd-scripts, OneTrainer, SimpleTuner, ai-toolkit, diffusion-pipe and similar).
Every image `name.png` gets a caption file `name.txt` next to it.

The tab is built around one rule from the training literature: **text explains away variance.**
Whatever a caption names is credited to that word; whatever is always there and never named is
learned by the trigger word. So caption what should stay changeable, and leave the thing you
are teaching to the trigger (plus a class noun).

## 1. Load a folder

Enter the folder path and press **Load Folder**. The gallery shows the images; click one to
edit its caption. The editor shows:

* **Token count** for the selected target model, e.g. `CLIP: 62/75 (max 225) ✅`. CLIP counts
  are exact (the CLIP tokenizer ships with the app and works offline). T5 and LLM-encoder counts
  are estimates and are marked with `≈`.
* **Provenance**, when the caption was written by AI: model, agent, time, retries, and
  `edited since` if the file changed after the model wrote it.
* A note when a file holds **several caption variants** (one per line, see below).

If some caption files contain old multi-caption blocks (lines separated by `---`), the status
says so; use **Batch Edit & Export → Normalize** to fix them.

## 2. Pick the target model and dataset type

* **Target model** sets the caption format (booru *tags* or natural-language *prose*), the
  token budget, the tag order and an optional quality-tag prefix. Profiles include SD 1.5 / SDXL,
  Pony V6, Illustrious, NoobAI, Anima, Neta Lumina, Pony V7, FLUX.1 / Chroma, FLUX.2,
  SD 3.5, Qwen-Image, Z-Image, HiDream, Wan and Lumina 2.
* **Dataset type** (Style, Concept / effect, Character / person, Object / product,
  Pose / composition, Multi-concept) sets the trigger prefix template, what the captioner must
  leave out, and the caption rules.
* **Trigger word** and **Class noun** fill the prefix, e.g. `txcl painting, `.

The **Guidance** panel explains the choices. Every tip carries an evidence badge:
🟢 measured in a study · 🔵 official docs / model card · 🟡 side-by-side image test ·
🟠 practitioner advice · ⚪ reasoning from the evidence. Most LoRA caption advice is not
measured, so treat it as a starting point and test with your own data.

Picking prose captions for a model trained on tags (or the reverse) shows a warning; it is
allowed, because comparing formats is a valid experiment.

## 3. Generate captions (AI Captioning)

1. Choose a **Vision Model** and **Agent / Team**. `(Direct Agent Call)` sends the instruction
   straight to the model, with no persona; this is usually best for training captions.
2. Adjust **Caption format**, **Length target**, **Caption rules** and **Never mention**.
   The **Caption Instruction** box is built from these settings; you can edit it freely.
3. **Prefix (trigger)** and **Suffix** are added to every caption without doubling a trigger
   that the model already wrote. Tick the quality-tag box to add the profile's quality tags.
4. **If a caption already exists:** Overwrite, Skip, Append, Prepend, or
   **Add as caption variant** (adds a new line, see below). Append and Prepend always stay on
   one line, joined with the separator you choose.
5. Press **Generate Caption for SELECTED Image** or **Generate Captions for ALL Loaded Images**.
   **⏹ Stop** ends a batch.

**Reproducibility** (accordion): temperature `0` and a fixed seed give the same caption for the
same image, model and instruction. Large images are downscaled before they are sent (default
1024 px). Glitches (empty output, refusals, "This image shows…" openers, repeated words, cut-off
text, sentences where tags were asked for) are retried with a new seed.

Captions are cleaned before they are saved: markdown and openers are removed, duplicate tags are
dropped, and tags you listed under **Never mention** are removed (in prose they are flagged).

### Caption sources

**Caption source** (top of AI Captioning) decides where captions come from:

| Source | What happens |
|---|---|
| Vision model | The vision model (or agent / team) writes the caption, as above. |
| Booru tagger only | A WD-style ONNX tagger writes Danbooru tags; no vision model is used. |
| Tagger tags as hints for the vision model | Tags the tagger is confident about are added to the instruction as hints ("mention only what is clearly visible"). Grounds the vision model, e.g. for character names. |
| Tagger tags + vision-model caption | Tagger tags first, then the vision model's caption: merged tags for tag models, or `tags. Sentence.` for prose / hybrid models. |
| Structured fields (JSON) → caption | The model fills fields (subject, clothing, setting, …) as JSON; the caption is assembled from a template. See **Attributes** below. |

**Booru tagger.** Choose a tagger (WD SwinV2 v3 by default; EVA02-Large is slower but more accurate). It needs the
optional package `onnxruntime` (`pip install onnxruntime`, or `onnxruntime-gpu` / `onnxruntime-directml`), and
downloads the model from Hugging Face on first use. Thresholds default to 0.35 (general) and 0.85 (character); MCut
picks a cut-off per image instead. Output follows the target model: tag order, rating tag names (Pony
`rating_safe`, Illustrious `general/sensitive/nsfw/explicit`, NoobAI `safe/…`), spaces instead of underscores. Raw
scores are stored in `.artagents/tagger`, so changing thresholds does not run the model again. Taggers are listed in
`dataset_profiles/taggers.yaml`; you can add a local ONNX tagger there with its own preprocessing.

**Vision backend.** Besides Ollama you can use any **OpenAI-compatible server** (LM Studio, vLLM, llama.cpp's
`llama-server`, KoboldCpp…). This is the dependable way to run captioners such as JoyCaption Beta One or ToriiGate.
Enter the URL (e.g. `http://localhost:1234/v1`), press **🔄 List models**, pick the model. An API key is optional
and never saved (or set `ARTAGENTS_VISION_API_KEY`). With this backend, and with structured fields, the instruction
goes straight to the model; agents and teams run on Ollama only.

## 4. Tags & Terms

The term table counts how many images contain each tag (or word / phrase in prose datasets).

* **Click a term** to show only the images that contain it. Bulk edits can then target just these.
* Terms in at least half of the images (and at least 4 images) are marked
  `⚠ over-represented: may bind to the trigger`. A tag present in every image tends to be absorbed
  by the trigger; delete it or make sure it varies.
* Your trigger and class noun are marked `trigger/class` and never flagged.
* **Bulk edit:** rename a tag, delete a tag, or find & replace (optionally with a regular
  expression) across all images or only the filtered ones. A snapshot is saved first.

## 5. Batch Edit & Export

* **Prepend Text / Append Text** add text to every caption on the same line.
* **Add as Caption Variant** adds the text as an extra line. OneTrainer and SimpleTuner read each
  line as a separate caption and pick one per step; kohya-ss needs `enable_wildcard` (and then
  does not shuffle tags). A common use: one long content caption plus a short trigger-only line.
* **Normalize** fixes files with several lines or old `---` blocks: merge them into one caption
  (duplicate tags are dropped) or keep one clean caption per line.
* **Export** writes `metadata.jsonl` or `metadata.csv` for training tools. Files with several
  variants export their first line, and the status lists them.

**Trainer config** writes a dataset config for kohya-ss sd-scripts, musubi-tuner (FLUX, Qwen-Image, Wan…),
ai-toolkit, OneTrainer, SimpleTuner or diffusion-pipe into `.artagents/export`. Repeats go into the config (your
folders are never renamed); SimpleTuner's repeats are converted (kohya 10 = SimpleTuner 9). Keep tokens are counted
from the trigger prefix. The file notes when captions have several lines, and for kohya adds `enable_wildcard`.

**Repeats calculator**: repeats = max(1, round(target / images)) per concept (kohya's rule), with steps per epoch.
Add other concept folders as `name: count` lines to balance a multi-concept dataset.

## 6. Attributes (structured fields)

After captioning with **Structured fields**, **🔄 Load sheet** shows every image's fields side by side. Edit cells and
**Save sheet**. The summary flags fields whose value is the same in most images: a constant, unnamed detail is learned
by the trigger, a captioned one stays changeable. Untick fields and **Render captions** to rebuild every caption
from the stored fields without calling the model (a snapshot is saved first). **Export CSV** downloads the sheet.
Typical use: for a character dataset, leave hair and eye colour out so the trigger learns them, keep clothing and
setting in so they stay promptable.

## 7. Health & Audit

**Scan dataset** checks every image on your computer, without AI: resolution against the training resolution,
extreme aspect ratios, colour mode and transparency, exact duplicates, near duplicates (perceptual hash), blur
relative to the rest of the dataset, very dark or bright images, and caption problems (missing, empty, several
lines, over the token budget, identical captions, trigger missing). **Show flagged images** filters the gallery.
**Quarantine selected image** moves an image and its caption to `.artagents/quarantine`; **Restore quarantined**
moves them back. Nothing is deleted. Results are cached in `.artagents/cache`.

**Bucket simulator** replicates kohya's aspect-ratio buckets: which bucket each image lands in, how much is cropped
and from which edge, and buckets that don't fill whole batches.

**Caption consistency audit** lists contradictory terms (`dataset_profiles/contradictions.yaml`), missing or repeated
triggers and triggers not at the start, several triggers in one caption (multi-concept leakage), words the dataset
type says to leave out, and captions much longer or shorter than usual. With **Run tagger for comparison** it also
lists caption tags the tagger scored very low, person counts that differ, and confident tagger tags the caption lacks.

**Cross-model verifier (experimental)** asks a second vision model yes / no / unsure for each statement of a caption,
mixed with two statements about objects that are not there. A "yes" to such a control means that model agrees too
easily on that image. Use a model from a different family than the captioner. Disputed statements are listed;
captions are never changed automatically.

## 8. Edit Pairs

For FLUX.1 Kontext, FLUX.2 edit and Qwen-Image-Edit LoRAs. Enter the **result** folder and the **input** folder(s)
(up to 3); files are matched by name, any image extension. **Load pairs** checks for missing inputs, aspect-ratio
mismatches and input sizes far from ~1 MP. **Write instructions** sends the input(s) and the result to the vision
model and saves an imperative instruction ("Make the car red") next to each result; inputs get no captions. **Extra
phrasings** adds alternative wordings as extra lines (BFL suggests 5–10 phrasings in total). Export for ai-toolkit
(`control_path`), musubi-tuner (TOML or JSONL with `control_path` / `control_path_N`) or fal (zip of `_start` / `_end`
pairs).

## 9. Caption Sets

Caption sets are alternative captions for the same images, for caption A/B tests. `img.txt` is the active set
(what trainers read by default); each other set is `img.<set>.txt`. kohya-ss and musubi-tuner can train on a set
directly with `caption_extension = ".<set>.txt"`.

- **Save active captions as a new set** copies the current captions, e.g. before you experiment.
- **Make the selected set active** copies a set over `img.txt` (a snapshot is saved first).
- **Move the selected set out** moves its files to `.artagents/deleted_sets` (nothing is deleted).
- **Compare all sets** shows per set: captions, average and maximum tokens, share over the token budget, average
  words, trigger coverage, distinct terms, files with several lines, and verifier disputes (when the verifier
  checked exactly that text). Selecting a set shows its length histogram.

**Caption Sweep** makes one set per strategy with the AI Captioning settings, without touching your active captions:

| Strategy | What it makes |
|---|---|
| Trigger only | just the trigger (no description) |
| Tags | tags from the tagger (if the caption source uses it) or the vision model |
| Short / long prose | ~15 / ~60-word natural-language captions |
| Structured fields | captions assembled from structured fields |
| Random-length mix | per image, randomly short, long or trigger only (random-length captions did best in Brack et al. 2025) |
| Long + trigger-only line | long caption plus a trigger-only variant line (fal's trick) |
| 95% / 5% blend | model captions with ~5% of your current captions kept (DALL-E 3 kept 5% original captions) |
| Quality tags on | your current captions with the profile's quality tags after the trigger |

Sets record how they were made in `.artagents/caption_sets.json`; the LoRA Lab's experiment records pick this up.

**Write a training copy** writes the selected set, with the images hard-linked (or copied), into
`.artagents/export/<set>-<mode>/`:

- one caption per line, for SimpleTuner and OneTrainer;
- kohya's `---===---` separators, with `enable_multiline_caption = true`;
- duplicated images with one caption each, for kohya with cached text-encoder outputs (which ignores random line
  selection) and for trainers whose multi-caption support is unverified.

Then train one LoRA per set and compare them in the **LoRA Lab** tab (see `lora-lab-guide.md`).

## 10. Snapshots and undo

Every batch change (AI batch, bulk edit, append/prepend, normalize, restore) first saves a snapshot
of all caption files. In **Snapshots** you can **Compare with current** (a diff per file) or
**Restore**. Restoring is exact: captions that did not exist at snapshot time are removed again,
and the current state is saved as a new snapshot first, so a restore can be undone. The 50 newest
snapshots are kept.

## Where ArtAgents stores its data

All of it lives in a hidden `.artagents` folder inside your dataset, so trainers never read it:

```
my_dataset/
├── 001.png
├── 001.txt
└── .artagents/
    ├── snapshots/<time>-<operation>/   # caption files + manifest.json
    ├── provenance/001.json             # model, digest, instruction, seed, options, version
    ├── tagger/001.json                 # tagger scores (re-threshold without re-running)
    ├── structured/001.json             # structured fields
    ├── verify/001.json                 # verifier answers
    ├── cache/health.json               # health-scan measurements
    ├── quarantine/                     # images moved out by the health scan
    └── export/                         # trainer configs, attributes.csv
```

Delete `.artagents` when you no longer need the history. Exclude it when you copy the dataset
to a training machine if your trainer scans subfolders.

## Editing profiles and presets

Target models and dataset types are defined in `dataset_profiles/models.yaml` and
`dataset_profiles/presets.yaml`; taggers in `taggers.yaml`, contradictory terms in `contradictions.yaml`. Both files explain their fields at the top. Edits to an existing
entry apply on the next action; new or renamed entries appear in the lists after a restart.
If a file is missing or invalid, built-in defaults are used and a warning is printed.
