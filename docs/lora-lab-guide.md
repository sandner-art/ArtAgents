# LoRA Lab: Guide

The **LoRA Lab** tab tests what a training run actually learned. The loop is: make caption variants (Captions tab →
Caption Sets), train one LoRA per variant in your trainer, then render the same prompts and seeds with every LoRA,
strength and a no-LoRA baseline, compare the grids, vote blind, and keep a record.

No published standard exists for LoRA caption A/B tests. This workflow follows the protocol synthesized in the
research report (DreamBooth-style prompt sets, a baseline, fixed seeds, blinded votes, intervals on every number).
Treat any difference smaller than the seed-to-seed variation as noise.

## 1. Validation Prompts

A validation set has four buckets:

| Bucket | Tests | Example |
|---|---|---|
| training-like | fidelity: does it reproduce the training data? | taken from your captions |
| held-out | generalization: the subject or style in new contexts | `[trigger] painting in a snowy forest` |
| trigger-less | leakage: does the LoRA change images without its trigger? | `painting in a snowy forest` |
| unrelated | prior preservation: does the base model still work? | `a red sports car on a mountain road` |

- **Fill starter prompts** builds all four buckets from your trigger, class noun and the dataset's captions. Tick
  **Style LoRA** for style datasets (content prompts with the style appended).
- **Draft held-out prompts** asks a text model for new recontextualization prompts. Review them before saving.
- Use `[trigger]` where the trigger goes. Per-prompt flags use the kohya / FluxGym syntax: `--w 832 --h 1216 --l 4.5
  --s 28 --d 7` (width, height, CFG, steps, seed).
- **Save as a new version** never overwrites: `validation_sets/<name>.v1.json`, `v2`, … so results stay comparable.
- **Export kohya / FluxGym sample prompts** writes one line per prompt and seed, ready for a trainer's sample prompts.

The DreamBooth protocol used about 25 prompts × 4 seeds. Start smaller (a few prompts per bucket, 2–4 seeds); the
Test Sweep shows the image count before you run.

## 2. Test Sweep

1. Pick **ComfyUI** or **Forge / A1111 / SD.Next** and press **🔄 Connect** (URLs come from App Settings).
2. **ComfyUI** needs a workflow with `%lora_name%` and `%lora_strength%` placeholders: use the included
   `lora_txt2img.json` or add a `LoraLoader` to your own (see `image_workflows/README.md`). The baseline reuses the LoRA
   slot at strength 0, which ComfyUI treats as "no LoRA". **Forge / A1111** needs nothing: ArtAgents adds
   `<lora:name:strength>` to the prompt, and the baseline has no tag.
3. Choose the checkpoint, the LoRA files (e.g. one per saved epoch), strengths (`0.6, 0.8, 1.0`), the validation set
   and the buckets. **Max prompts** keeps a first run small.
4. **Run test sweep.** Images, `run.json` and one labelled grid per prompt (rows = baseline and each LoRA/strength,
   columns = seeds) go to `outputs/lora_tests/<time>-<name>/`. **Stop** keeps what is finished.

**Copy a LoRA into ComfyUI** copies a `.safetensors` file into `ComfyUI/models/loras` (never overwrites).

For multi-concept LoRAs, put each trigger alone and the triggers together into the validation set; the grids then
show leakage between concepts.

## 3. Score & Vote

**Blind pairwise votes.** Two images with the same prompt and seed from different conditions, labels hidden, sides
random. Votes are fitted with a Bradley-Terry model; win rates show 95% Wilson intervals. Votes are stored in the run
folder (`votes.jsonl`). About a thousand comparisons gave a stable ranking in a 2026 identity benchmark; a few dozen
already separate clearly different LoRAs.

**Vision-model judge (experimental).** A vision model votes on each pair twice, with the order swapped. If its answer
changes with the order it is position-biased on that pair and the vote is not counted. It judges pairs you already
voted on first and reports how often it agrees with you; 50% is chance. Trust it only when agreement is high.

**Metric scoring (experimental).** Needs PyTorch plus `open_clip_torch` (CLIP) or `transformers` (DINOv2); models
download on first use. For each condition and bucket:

- **Fidelity** is the mean cosine similarity to the training images, with a bootstrap 95% interval. DINO agrees with
  human preference only weakly (r ≈ 0.32).
- **Δ vs baseline** on trigger-less and unrelated prompts is **leakage**: the LoRA changing images without its
  trigger.
- **Adherence (CLIP-T)** is the prompt–image similarity (CLIP models only). CLIP behaves like a bag of words.
- **CMMD** is the distance to the training set (lower = closer), usable at small sample sizes, unlike FID.

Metrics are signals, not verdicts. Aesthetic predictors are deliberately not included: they mostly measure
photorealism and would penalize non-photographic styles.

## 4. Experiments

One record per training experiment, stored in `experiments/artagents_experiments.sqlite`:

- dataset folder and caption snapshot;
- caption set and how it was made (filled from the set's manifest);
- trainer and its settings (typed or pasted);
- LoRA files, validation set and version, and the test run (filled from the run);
- metrics and vote results (taken from the run folder when you save);
- notes and conclusions.

**Export report** writes Markdown and HTML (with the grids) into the run folder. Shared reports let others repeat
the test: LoRA-scale caption comparisons are exactly what the community lacks.

## Caption sets (Captions tab)

See `captions-guide.md` → *Caption Sets*. In short:

- `img.txt` is the active set and `img.<set>.txt` holds each alternative.
- The **Caption Sweep** makes one set per strategy: trigger only, tags, short or long prose, structured, a
  random-length mix, long + trigger-only variant, a 95/5 model/human blend, or quality tags on.
- **Write a training copy** materializes caption variants correctly for each trainer.
