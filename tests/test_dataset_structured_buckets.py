# ArtAgent/tests/test_dataset_structured_buckets.py
"""Tests for core/dataset/structured.py (fields → captions, attribute sheet) and core/dataset/buckets.py."""
import csv
import io
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import buckets as bk, structured as st


# ======================= structured =======================

def test_json_schema_known_fields_only():
    schema = st.json_schema(["subject", "bogus", "hair"])
    assert schema == {"type": "object", "properties": {"subject": {"type": "string"}, "hair": {"type": "string"}},
                      "required": ["subject", "hair"]}


def test_json_schema_defaults_when_empty():
    assert st.json_schema([])["required"] == st.DEFAULT_FIELDS
    assert st.json_schema(["nope"])["required"] == st.DEFAULT_FIELDS


def test_instruction_lists_fields_and_style():
    text = st.instruction(["subject", "text"], "tags", "Mention the logo.")
    assert '- "subject": ' in text and '- "text": legible text quoted exactly' in text
    assert "booru-style tags" in text and "Mention the logo." in text
    assert text.rstrip().endswith("Answer with the JSON object only.")
    assert "short plain phrases" in st.instruction(["subject"], "prose")


def test_normalize_fields():
    data = {"Subject": " a cat ", "objects": ["a cup", "", "a book"], "hair": None, "lighting": 3,
            "camera": "N/A", "setting": "none", "text": '"OPEN"', "other": "x"}
    out = st.normalize_fields(data)
    assert out["subject"] == "a cat"
    assert out["objects"] == "a cup, a book"
    assert out["hair"] == "" and out["camera"] == "" and out["setting"] == ""
    assert out["lighting"] == "3"
    assert out["other"] == "x"
    assert st.normalize_fields(data, ["subject"]) == {"subject": "a cat"}
    assert st.normalize_fields(None) == {}


def test_field_label_roundtrip():
    for fid in st.FIELDS:
        assert st.label_to_id(st.field_label(fid)) == fid
    assert st.field_label("custom_thing") == "Custom_Thing"
    assert st.label_to_id("unknown") == "unknown"


def test_default_template():
    assert st.default_template(["subject", "action", "setting"], "tags") == "{subject}, {action}, {setting}"
    assert st.default_template(["subject", "action", "setting"], "prose") == "{subject} {action}, {setting}"
    assert st.default_template(["subject", "setting"], "prose") == "{subject}, {setting}"


FIELDS = {"subject": "a girl", "action": "standing", "expression": "", "hair": "long red hair",
          "clothing": "blue dress", "setting": "a park", "lighting": "soft daylight"}


def test_render_prose_lead_ins_and_capitalization():
    tpl = st.default_template(["subject", "action", "expression", "hair", "clothing", "setting", "lighting"], "prose")
    assert st.render(FIELDS, tpl, "prose") == \
        "A girl standing, with long red hair, wearing blue dress, a park, soft daylight."


def test_render_prose_keeps_existing_final_punctuation():
    assert st.render({"subject": "is it a cat?"}, "{subject}", "prose") == "Is it a cat?"


def test_render_drops_empty_and_unknown_fields():
    assert st.render(FIELDS, "{expression}, {subject}, {missing}, {setting}", "tags") == "a girl, a park"


def test_render_include_filter():
    out = st.render(FIELDS, "{subject}, {hair}, {setting}", "tags", include=["subject", "setting"])
    assert out == "a girl, a park"
    assert st.render(FIELDS, "{subject}", "tags", include=[]) == ""


def test_render_tags_dedupes_and_splits_values():
    fields = {"subject": "1girl, solo", "objects": "Solo, umbrella", "hair": "red hair"}
    assert st.render(fields, "{subject}, {objects}, {hair}", "tags") == "1girl, solo, umbrella, red hair"


def test_render_tags_has_no_lead_ins():
    assert st.render({"clothing": "dress", "hair": "red hair"}, "{clothing}, {hair}", "tags") == "dress, red hair"


def test_render_empty_template():
    assert st.render(FIELDS, "", "prose") == ""
    assert st.render({}, "{subject}", "prose") == ""


def test_render_visible_text_keeps_quotes():
    fields = st.normalize_fields({"subject": "a shop sign", "text": '"OPEN"'})
    assert st.render(fields, "{subject}, {text}", "prose") == 'A shop sign, with the text "OPEN".'


def test_save_and_load_fields(tmp_path):
    image = str(tmp_path / "a.png")
    st.save_fields(image, {"subject": "a cat", "hair": ""}, model="m", source="model")
    path = tmp_path / ".artagents" / "structured" / "a.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["image"] == "a.png" and data["model"] == "m" and data["source"] == "model"
    assert st.load_fields(image) == {"subject": "a cat", "hair": ""}
    assert st.load_fields(str(tmp_path / "missing.png")) == {}


def test_load_fields_broken_json(tmp_path):
    (tmp_path / ".artagents" / "structured").mkdir(parents=True)
    (tmp_path / ".artagents" / "structured" / "a.json").write_text("{nope", encoding="utf-8")
    assert st.load_fields(str(tmp_path / "a.png")) == {}


@pytest.fixture
def sheet(tmp_path):
    paths = {n: str(tmp_path / n) for n in ("a.png", "b.png", "c.png", "d.png", "e.png")}
    st.save_fields(paths["a.png"], {"subject": "a girl", "setting": "a park", "zzz_custom": "x"})
    st.save_fields(paths["b.png"], {"subject": "a girl", "setting": "a beach"})
    st.save_fields(paths["c.png"], {"subject": "A Girl", "hair": "red"})
    st.save_fields(paths["d.png"], {"subject": "a girl", "setting": "a city"})
    return paths  # e.png has no fields


def test_sheet_columns_and_rows(sheet):
    cols = st.sheet_columns(sheet)
    assert cols == ["subject", "hair", "setting", "zzz_custom"]  # FIELDS order, then extras
    rows = st.sheet_rows(sheet, cols)
    assert [r[0] for r in rows] == ["a.png", "b.png", "c.png", "d.png"]
    assert rows[0] == ["a.png", "a girl", "", "a park", "x"]


def test_save_sheet_writes_edits(sheet):
    cols = st.sheet_columns(sheet)
    rows = st.sheet_rows(sheet, cols)
    rows[1][3] = "a lake"      # b.png setting
    rows.append(["unknown.png", "x", "", "", ""])
    rows.append([])
    assert st.save_sheet(sheet, cols, rows) >= 1
    assert st.load_fields(sheet["b.png"])["setting"] == "a lake"
    data = json.loads(open(st.fields_path(sheet["b.png"]), encoding="utf-8").read())
    assert data["source"] == "edited"
    assert not os.path.exists(st.fields_path(sheet["e.png"]))


def test_save_sheet_identical_fields_unchanged(tmp_path):
    paths = {"a.png": str(tmp_path / "a.png")}
    st.save_fields(paths["a.png"], {"subject": "a cat", "setting": "a sofa"}, model="m")
    cols = st.sheet_columns(paths)
    assert st.save_sheet(paths, cols, st.sheet_rows(paths, cols)) == 0


def test_save_sheet_unedited_rows_are_not_rewritten(sheet):
    cols = st.sheet_columns(sheet)
    assert st.save_sheet(sheet, cols, st.sheet_rows(sheet, cols)) == 0


def test_save_sheet_keeps_model_name(tmp_path):
    paths = {"a.png": str(tmp_path / "a.png")}
    st.save_fields(paths["a.png"], {"subject": "a cat"}, model="gemma4:e4b")
    st.save_sheet(paths, ["subject"], [["a.png", "a dog"]])
    data = json.loads(open(st.fields_path(paths["a.png"]), encoding="utf-8").read())
    assert data["fields"]["subject"] == "a dog"
    assert data["model"] == "gemma4:e4b"


def test_save_sheet_none_cells_become_empty(sheet):
    cols = ["subject", "setting"]
    assert st.save_sheet(sheet, cols, [["a.png", None, "a park"]]) == 1
    assert st.load_fields(sheet["a.png"])["subject"] == ""


def test_field_summary_flags_constant_values(sheet):
    cols = st.sheet_columns(sheet)
    summary = {row[0]: row for row in st.field_summary(cols, st.sheet_rows(sheet, cols))}
    subject = summary["Subject"]
    assert subject[1:5] == [4, 1, "a girl", 100.0]      # Case-insensitive: all 4 are 'a girl'
    assert subject[5].startswith("⚠ mostly the same")
    assert summary["Setting"][1:3] == [3, 3] and summary["Setting"][5] == ""
    assert summary["Hair"][1] == 1


def test_field_summary_empty_and_varied():
    rows = [[f"{i}.png", f"value {i}", ""] for i in range(5)]
    summary = st.field_summary(["setting", "hair"], rows)
    assert summary[0][5] == "varies in every image"
    assert summary[1] == ["Hair", 0, 0, "", 0.0, "empty"]


def test_field_summary_needs_four_images_to_flag():
    rows = [["a", "x"], ["b", "x"], ["c", "x"]]
    assert st.field_summary(["subject"], rows)[0][5] == ""


def test_sheet_csv():
    text = st.sheet_csv(["subject", "hair"], [["a.png", "a girl, smiling", "red"]])
    rows = list(csv.reader(io.StringIO(text)))
    assert rows == [["image", "Subject", "Hair"], ["a.png", "a girl, smiling", "red"]]


# ======================= buckets =======================
# Reference transcribed from kohya-ss sd-scripts (library/model_util.make_bucket_resolutions and
# library/dataset.BucketManager.select_bucket / round_to_steps), kept independent of the app code.

def ref_make_bucket_resolutions(max_reso, min_size=256, max_size=1024, divisible=64):
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
    resos = list(resos)
    resos.sort()
    return resos


def ref_select_bucket(image_width, image_height, resos, no_upscale, max_area, reso_steps=64):
    def round_to_steps(x):
        x = int(x + 0.5)
        return x - x % reso_steps

    aspect_ratio = image_width / image_height
    if not no_upscale:
        reso = (image_width, image_height)
        if reso not in set(resos):
            ars = np.array([w / h for w, h in resos])
            reso = resos[np.abs(ars - aspect_ratio).argmin()]
        ar_reso = reso[0] / reso[1]
        scale = reso[1] / image_height if aspect_ratio > ar_reso else reso[0] / image_width
        resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
    else:
        if image_width * image_height > max_area:
            resized_width = math.sqrt(max_area * aspect_ratio)
            resized_height = max_area / resized_width
            b_width_rounded = round_to_steps(resized_width)
            b_height_in_wr = round_to_steps(b_width_rounded / aspect_ratio)
            ar_width_rounded = b_width_rounded / b_height_in_wr
            b_height_rounded = round_to_steps(resized_height)
            b_width_in_hr = round_to_steps(b_height_rounded * aspect_ratio)
            ar_height_rounded = b_width_in_hr / b_height_rounded
            if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
            else:
                resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
        else:
            resized_size = (image_width, image_height)
        reso = (resized_size[0] - resized_size[0] % reso_steps, resized_size[1] - resized_size[1] % reso_steps)
    return reso, resized_size


@pytest.mark.parametrize("res, lo, hi", [(1024, 256, 1024), (1024, 256, 2048), (512, 256, 1024), (768, 320, 1536)])
def test_bucket_resolutions_match_kohya(res, lo, hi):
    assert bk.make_bucket_resolutions((res, res), lo, hi, 64) == ref_make_bucket_resolutions((res, res), lo, hi, 64)


def test_bucket_resolutions_known_values():
    resos = bk.make_bucket_resolutions((1024, 1024), 256, 2048, 64)
    assert (1024, 1024) in resos and (256, 2048) in resos and (2048, 256) in resos
    assert (576, 1792) in resos and (1792, 576) in resos
    assert resos == sorted(resos)


SIZES = [(1024, 1024), (768, 512), (512, 768), (3000, 500), (500, 3000), (2000, 1500), (600, 400), (1023, 1025),
         (4000, 4000), (100, 100), (1920, 1080), (1080, 1920), (777, 333)]


@pytest.mark.parametrize("no_upscale", [False, True])
@pytest.mark.parametrize("res, lo, hi", [(1024, 256, 2048), (512, 256, 1024)])
def test_select_bucket_matches_kohya(no_upscale, res, lo, hi):
    resos = bk.make_bucket_resolutions((res, res), lo, hi, 64)
    for w, h in SIZES:
        assert bk.select_bucket(w, h, resos, no_upscale, res * res, 64) == \
            ref_select_bucket(w, h, resos, no_upscale, res * res), (w, h)


def test_select_bucket_hand_checked_768x512_at_512():
    # The commonly cited example: 768×512 at 512px lands in 640×384 after resizing to 640×427 (~10% cropped)
    resos = bk.make_bucket_resolutions((512, 512), 256, 1024, 64)
    assert bk.select_bucket(768, 512, resos) == ((640, 384), (640, 427))


def test_select_bucket_square_exact():
    resos = bk.make_bucket_resolutions((1024, 1024), 256, 2048, 64)
    assert bk.select_bucket(1024, 1024, resos) == ((1024, 1024), (1024, 1024))
    assert bk.select_bucket(2048, 2048, resos) == ((1024, 1024), (1024, 1024))


def test_select_bucket_no_upscale_large_and_small():
    resos = bk.make_bucket_resolutions((1024, 1024), 256, 2048, 64)
    assert bk.select_bucket(2000, 1500, resos, True, 1024 * 1024) == ((1088, 832), (1109, 832))
    assert bk.select_bucket(600, 400, resos, True, 1024 * 1024) == ((576, 384), (600, 400))


def test_simulate_rows_and_summary():
    sizes = {"wide.png": (768, 512), "square.png": (512, 512), "small.png": (256, 256)}
    result = bk.simulate(sizes, resolution=512, min_bucket=256, max_bucket=1024)
    rows = {r[0]: r for r in result["rows"]}
    assert rows["wide.png"] == ["wide.png", "768×512", "640×384", 0.833, 10.1, "top/bottom", "⚠ large crop"]
    assert rows["square.png"][2:6] == ["512×512", 1.0, 0.0, ""]
    assert rows["small.png"][6] == "upscaled ×2.00"
    assert "3 image(s) in 2 bucket(s) at 512px (upscale allowed, buckets 256–1024, step 64)." in result["summary"]
    assert "Crop: average 3.4%, max 10.1%." in result["summary"]


def test_simulate_cut_from_matches_lost_edge():
    sizes = {"tall.png": (500, 3000), "wide.png": (3000, 500), "photo.png": (1920, 1080)}
    result = bk.simulate(sizes, resolution=1024)
    resos = ref_make_bucket_resolutions((1024, 1024), 256, 2048)
    for row in result["rows"]:
        (bw, bh), (rw, rh) = ref_select_bucket(*sizes[row[0]], resos, False, 1024 * 1024)
        assert row[2] == f"{bw}×{bh}"
        lost_w, lost_h = rw - bw, rh - bh
        expected = "left/right" if lost_w > lost_h else ("top/bottom" if lost_h > 0 else "")
        assert row[5] == expected, row
        assert row[4] == round(100.0 * (rw * rh - bw * bh) / (rw * rh), 1)


def test_simulate_lonely_buckets():
    sizes = {"a.png": (1024, 1024), "b.png": (1024, 1024), "c.png": (768, 512)}
    result = bk.simulate(sizes, resolution=1024, batch_size=2)
    by_bucket = {r[0]: r for r in result["buckets"]}
    assert by_bucket["1024×1024"][2:] == [2, 2, ""]
    lonely = [r for r in result["buckets"] if r[2] == 1][0]
    assert lonely[4].startswith("⚠ 1 images per epoch")
    assert "1 bucket(s) do not fill whole batches of 2." in result["summary"]
    # Repeats can fill the batch
    assert "do not fill" not in bk.simulate(sizes, resolution=1024, batch_size=2, repeats=2)["summary"]


def test_simulate_buckets_sorted_by_count_and_skips_zero_sizes():
    sizes = {"a.png": (1024, 1024), "b.png": (1024, 1024), "c.png": (768, 512), "bad.png": (0, 100)}
    result = bk.simulate(sizes, resolution=1024)
    assert result["buckets"][0][2] == 2
    assert [r[0] for r in result["rows"]] == ["a.png", "b.png", "c.png"]


def test_simulate_empty():
    result = bk.simulate({})
    assert result["rows"] == [] and result["buckets"] == []
    assert result["summary"].startswith("0 image(s) in 0 bucket(s)")
