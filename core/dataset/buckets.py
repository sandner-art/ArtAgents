# ArtAgent/core/dataset/buckets.py
"""
Aspect-ratio bucket simulator.

Replicates kohya-ss sd-scripts (library/model_util.make_bucket_resolutions and
library/dataset.BucketManager.select_bucket) to show, before training, which
bucket each image lands in, how much of it is cropped away, and which buckets
hold fewer images than the batch size.
"""
from __future__ import annotations

import math
from collections import defaultdict


def make_bucket_resolutions(max_reso: tuple[int, int], min_size: int = 256, max_size: int = 1024,
                            divisible: int = 64) -> list[tuple[int, int]]:
    """Same algorithm as kohya's model_util.make_bucket_resolutions."""
    max_width, max_height = max_reso
    max_area = max_width * max_height
    resos = set()
    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))
    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))
        width += divisible
    return sorted(resos)


def select_bucket(width: int, height: int, resos: list[tuple[int, int]], no_upscale: bool = False,
                  max_area: int | None = None, steps: int = 64):
    """
    Returns (bucket (w, h), resized size (w, h)) like kohya's BucketManager.select_bucket.
    The image is resized to `resized` and then cropped to the bucket.
    """
    aspect = width / height
    if not no_upscale:
        reso = (width, height)
        if reso not in resos:
            errors = [abs(w / h - aspect) for w, h in resos]
            reso = resos[errors.index(min(errors))]
        ar_reso = reso[0] / reso[1]
        scale = reso[1] / height if aspect > ar_reso else reso[0] / width
        resized = (int(width * scale + 0.5), int(height * scale + 0.5))
        return reso, resized

    def round_to_steps(x):
        x = int(x + 0.5)
        return x - x % steps

    if max_area and width * height > max_area:
        resized_w = math.sqrt(max_area * aspect)
        resized_h = max_area / resized_w
        b_w = round_to_steps(resized_w)
        b_h_in_w = round_to_steps(b_w / aspect)
        ar_w = b_w / b_h_in_w if b_h_in_w else float("inf")
        b_h = round_to_steps(resized_h)
        b_w_in_h = round_to_steps(b_h * aspect)
        ar_h = b_w_in_h / b_h if b_h else float("inf")
        if abs(ar_w - aspect) < abs(ar_h - aspect):
            resized = (b_w, int(b_w / aspect + 0.5))
        else:
            resized = (int(b_h * aspect + 0.5), b_h)
    else:
        resized = (width, height)
    bucket = (resized[0] - resized[0] % steps, resized[1] - resized[1] % steps)
    return bucket, resized


def simulate(sizes: dict[str, tuple[int, int]], resolution: int = 1024, min_bucket: int = 256, max_bucket: int = 2048,
             steps: int = 64, batch_size: int = 1, repeats: int = 1, no_upscale: bool = False) -> dict:
    """
    sizes: {filename: (width, height)}. Returns {"rows": per-image rows, "buckets": per-bucket rows, "summary": str}.
    Per-image row: [image, size, bucket, scale, cropped %, cut from, note].
    """
    resos = make_bucket_resolutions((resolution, resolution), min_bucket, max_bucket, steps)
    rows, members = [], defaultdict(list)
    crops = []
    for name in sorted(sizes):
        w, h = sizes[name]
        if not w or not h:
            continue
        bucket, resized = select_bucket(w, h, resos, no_upscale, resolution * resolution, steps)
        lost_w, lost_h = max(0, resized[0] - bucket[0]), max(0, resized[1] - bucket[1])
        area = resized[0] * resized[1]
        cropped = 100.0 * (area - bucket[0] * bucket[1]) / area if area else 0.0
        cut = "left/right" if lost_w > lost_h else ("top/bottom" if lost_h > 0 else "")
        scale = resized[0] / w
        note = []
        if scale > 1.05:
            note.append(f"upscaled ×{scale:.2f}")
        if cropped >= 10:
            note.append("⚠ large crop")
        rows.append([name, f"{w}×{h}", f"{bucket[0]}×{bucket[1]}", round(scale, 3), round(cropped, 1), cut, "; ".join(note)])
        members[bucket].append(name)
        crops.append(cropped)

    bucket_rows = []
    lonely = 0
    for bucket in sorted(members, key=lambda b: (-len(members[b]), b)):
        count = len(members[bucket])
        per_epoch = count * max(1, repeats)
        note = ""
        if batch_size > 1 and per_epoch % batch_size:
            note = f"⚠ {per_epoch} images per epoch: the last batch in this bucket is smaller than {batch_size}"
            lonely += 1
        bucket_rows.append([f"{bucket[0]}×{bucket[1]}", round(bucket[0] / bucket[1], 3), count, per_epoch, note])

    n = len(rows)
    summary = (f"{n} image(s) in {len(members)} bucket(s) at {resolution}px "
               f"({'no upscale' if no_upscale else 'upscale allowed'}, buckets {min_bucket}–{max_bucket}, step {steps}).")
    if crops:
        summary += f" Crop: average {sum(crops) / len(crops):.1f}%, max {max(crops):.1f}%."
    if lonely:
        summary += f" {lonely} bucket(s) do not fill whole batches of {batch_size}."
    return {"rows": rows, "buckets": bucket_rows, "summary": summary, "resolutions": resos}
