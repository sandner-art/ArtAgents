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

## 6. Snapshots and undo

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
    └── provenance/001.json             # model, digest, instruction, seed, options, version
```

Delete `.artagents` when you no longer need the history. Exclude it when you copy the dataset
to a training machine if your trainer scans subfolders.

## Editing profiles and presets

Target models and dataset types are defined in `dataset_profiles/models.yaml` and
`dataset_profiles/presets.yaml`. Both files explain their fields at the top. Edits to an existing
entry apply on the next action; new or renamed entries appear in the lists after a restart.
If a file is missing or invalid, built-in defaults are used and a warning is printed.
