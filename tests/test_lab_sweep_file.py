# ArtAgent/tests/test_lab_sweep_file.py
"""Sweep files out, results folders back (GRADIENT on another machine)."""
import hashlib
import json
import os
from unittest.mock import patch

import pytest
from PIL import Image

from core import lab_ui
from core.lab import lora_sweep as ls
from core.lab import sweep_file as sf
from core.lab import validation

VSET = {"trigger": "txcl", "seeds": [1, 2], "defaults": {"width": 64, "height": 64, "steps": 4, "cfg": None, "negative": "bad"},
        "prompts": [{"bucket": "held-out", "prompt": "[trigger] cat", "negative": "", "flags": ""},
                    {"bucket": "unrelated", "prompt": "a car", "negative": "", "flags": "--s 9"}]}
META = {"name": "my test", "validation_set": "vs", "validation_version": 3, "trigger": "txcl"}


@pytest.fixture
def lora_file(tmp_path):
    path = tmp_path / "train" / "style_e04.safetensors"
    path.parent.mkdir()
    path.write_bytes(b"fake lora weights")
    return str(path)


def export(tmp_path, loras, options=None, model="flux2-klein"):
    jobs = ls.build_jobs(VSET, loras, [0.5, 1.0])
    run_dir = ls.new_run_dir(str(tmp_path / "runs"), "my test")
    path, sha = sf.export_sweep(run_dir, jobs, META, model, options)
    return jobs, run_dir, path, sha


def test_export_writes_sweep_and_waiting_run(tmp_path, lora_file):
    jobs, run_dir, path, sha = export(tmp_path, [lora_file, "style_e04.safetensors", "other.safetensors"],
                                      {"scheduler": "Euler", "quantization": "bf16", "vae_id": "", "guidance": "linear(9,3)"})
    raw = open(path, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == sha and os.path.basename(path) == sf.SWEEP_FILE
    sweep = json.loads(raw)
    assert sweep["format"] == sf.FORMAT and sweep["version"] == 1
    assert sweep["run"] == {"name": "my test", "folder": os.path.basename(run_dir), "validation_set": "vs",
                            "validation_version": 3, "trigger": "txcl"}
    assert sweep["model"] == {"model_id": "flux2-klein", "scheduler_id": "Euler", "quantization": "bf16", "vae_id": None,
                              "guidance": "linear(9,3)"}
    loras = {lo["key"]: lo for lo in sweep["loras"]}
    assert list(loras) == ["style_e04", "style_e04-2", "other"]                    # Same stem twice: unique keys
    assert loras["style_e04"]["sha256"] == hashlib.sha256(b"fake lora weights").hexdigest()
    assert loras["style_e04"]["name"] == lora_file
    assert loras["style_e04-2"]["sha256"] is None and loras["other"]["sha256"] is None  # Names, not files here
    assert len(sweep["jobs"]) == len(jobs) == 2 * 2 * (1 + 3 * 2)
    first, second = sweep["jobs"][0], sweep["jobs"][1]
    assert first["lora"] is None and first["condition"] == ls.BASELINE and first["prompt"] == "txcl cat"
    assert second["lora"] == "style_e04" and second["strength"] == 0.5 and second["steps"] == 4
    assert {j["steps"] for j in sweep["jobs"] if j["prompt_index"] == 1} == {9}
    data = ls.load_run(run_dir)
    assert data["status"] == "exported" and data["sweep_sha256"] == sha and data["results"] == []
    assert data["backend"] == "GRADIENT (sweep file)" and data["sampler"] == "Euler" and data["model"] == "flux2-klein"
    assert sf.find_export(str(tmp_path / "runs"), sha) == run_dir


def test_export_needs_model_and_jobs(tmp_path):
    with pytest.raises(sf.SweepFileError, match="model"):
        export(tmp_path, [], model="")
    run_dir = ls.new_run_dir(str(tmp_path), "x")
    with pytest.raises(sf.SweepFileError, match="no images"):
        sf.export_sweep(run_dir, [], META, "m")


def png(path, color=(200, 10, 10), size=(64, 64), fmt="PNG"):
    Image.new("RGB", size, color).save(path, fmt)


def results_folder(tmp_path, sha, jobs, *, skip=(), fail=(), skipped_lora=(), files=None, extra=None):
    folder = tmp_path / "from_gradient"
    folder.mkdir(exist_ok=True)
    results = []
    for job in jobs:
        if job.index in skip:
            continue
        name = (files or {}).get(job.index, f"img_{job.index}.png")
        entry = {"index": job.index, "file": name, "seed": job.seed, "output_id": 100 + job.index, "error": None,
                 "run_uuid": "group-run"}
        if job.index in fail:
            entry.update(file=None, error="CUDA out of memory")
        elif job.index in skipped_lora:
            entry["loras_skipped"] = ["style_e04"]
        elif name and not os.path.isabs(name) and ".." not in name and not (folder / name).exists():
            png(folder / name, color=(job.index * 10 % 255, 0, 0))
        results.append(entry)
    data = {"format": sf.RESULTS_FORMAT, "version": 1, "sweep_sha256": sha,
            "renderer": {"app": "GRADIENT", "version": "0.9.1", "model_id": "flux2-klein", "nested": {"x": 1}},
            "results": results}
    data.update(extra or {})
    (folder / sf.RESULTS_FILE).write_text(json.dumps(data), encoding="utf-8")
    return str(folder)


def test_import_results_into_the_exported_run(tmp_path, lora_file):
    jobs, run_dir, _, sha = export(tmp_path, [lora_file])
    folder = results_folder(tmp_path, sha, jobs, skip={3}, fail={4}, skipped_lora={5})
    got_dir, counts = sf.import_results(str(tmp_path / "runs"), f' "{folder}" ')
    assert got_dir == run_dir and counts == {"images": len(jobs) - 3, "errors": 2, "missing": 1}
    data = ls.load_run(run_dir)
    assert data["status"] == "imported" and data["renderer"]["app"] == "GRADIENT" and "nested" not in data["renderer"]
    assert data["results_folder"] == os.path.abspath(folder)
    by_job = {r["job"]: r for r in data["results"]}
    assert len(by_job) == len(jobs)
    assert by_job[3]["error"].startswith("Not in the results") and by_job[3]["path"] is None
    assert by_job[4]["error"] == "CUDA out of memory"
    assert "LoRA not applied" in by_job[5]["error"] and "style_e04" in by_job[5]["error"]
    ok = by_job[0]
    assert ok["path"] == ls._file_name(jobs[0]) and ok["remote"] == {"output_id": 100, "run_uuid": "group-run"}
    assert Image.open(os.path.join(run_dir, ok["path"])).size == (64, 64)
    grids = [f for f in os.listdir(run_dir) if f.startswith("grid_")]
    assert len(grids) == 2                                                       # One per prompt
    # Importing again replaces the results (e.g. after GRADIENT rendered the missing job)
    folder = results_folder(tmp_path, sha, jobs)
    _, counts = sf.import_results(str(tmp_path / "runs"), folder)
    assert counts == {"images": len(jobs), "errors": 0, "missing": 0}


def test_untrusted_files_are_checked(tmp_path, lora_file):
    jobs, run_dir, _, sha = export(tmp_path, [lora_file])
    outside = tmp_path / "secret.png"
    png(outside)
    folder = tmp_path / "from_gradient"
    folder.mkdir()
    (folder / "not_image.png").write_text("<html>not an image</html>")
    png(folder / "gif.gif", fmt="GIF")
    files = {0: "../secret.png", 1: str(outside), 2: "not_image.png", 6: "gif.gif", 7: ""}
    results_folder(tmp_path, sha, jobs, files=files)
    _, counts = sf.import_results(str(tmp_path / "runs"), str(folder))
    by_job = {r["job"]: r for r in ls.load_run(run_dir)["results"]}
    for index in (0, 1, 7):
        assert by_job[index]["path"] is None and "outside the results folder" in by_job[index]["error"]
    assert "could not be read" in by_job[2]["error"] and "could not be read" in by_job[6]["error"]
    assert counts["errors"] == 5 and not any(f.startswith("secret") for f in os.listdir(run_dir))


def test_import_refusals(tmp_path, lora_file):
    jobs, run_dir, path, sha = export(tmp_path, [lora_file])
    root = str(tmp_path / "runs")
    with pytest.raises(sf.SweepFileError, match="Not a folder"):
        sf.import_results(root, str(tmp_path / "nope"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(sf.SweepFileError, match="No artagents-results.json"):
        sf.import_results(root, str(empty))
    other = results_folder(tmp_path, "f" * 64, jobs)
    with pytest.raises(sf.SweepFileError, match="didn't export"):          # Results of another sweep file
        sf.import_results(root, other)
    for extra, message in (({"format": "x"}, "not an ArtAgents"), ({"version": 2}, "version 2"),
                           ({"results": "x"}, "no results list")):
        folder = results_folder(tmp_path, sha, jobs, extra=extra)
        with pytest.raises(sf.SweepFileError, match=message):
            sf.import_results(root, folder)
    (tmp_path / "from_gradient" / sf.RESULTS_FILE).write_text("{broken", encoding="utf-8")
    with pytest.raises(sf.SweepFileError, match="could not be read"):
        sf.import_results(root, str(tmp_path / "from_gradient"))
    folder = results_folder(tmp_path, sha, jobs)
    with open(path, "ab") as f:                                              # The exported file was edited
        f.write(b" ")
    with pytest.raises(sf.SweepFileError, match="was changed"):
        sf.import_results(root, folder)


def test_lora_keys():
    assert sf.lora_keys(["a/x.safetensors", "b/x.safetensors", "y.pt", "a/x.safetensors", "weird name!"]) == {
        "a/x.safetensors": "x", "b/x.safetensors": "x-2", "y.pt": "y", "weird name!": "weird_name"}
    assert sf.file_sha256("relative.safetensors") is None and sf.file_sha256("") is None


# ---------- LoRA Lab buttons ----------

@pytest.fixture
def lab(tmp_path):
    root = tmp_path / "app"
    root.mkdir()

    def gap(rel):
        return str(root / rel)
    settings = {"image_generation": {"output_dir": str(tmp_path / "outputs" / "images"), "gradient_url": "http://127.0.0.1:9"}}
    with patch("core.lab.validation.get_absolute_path", side_effect=gap), patch("core.lab_ui.get_absolute_path", side_effect=gap):
        validation.save_new_version("vs", [["held-out", "[trigger] cat", "", ""], ["unrelated", "a car", "", ""]],
                                    "1, 2", "txcl", {"width": 64, "height": 64})
        yield settings


def test_lab_export_and_import(lab, tmp_path, lora_file):
    status, download, runs, _ = lab_ui.export_gradient_sweep(
        lab, "flux2-klein", [], f"{lora_file}\nname-only.safetensors", "1", True, "vs", "1", None, 0, "lab run",
        "Euler", "(model default)", "(model default)", "")
    assert status.startswith("✅ Sweep file for 12 image(s)") and "1 LoRA(s) are names" in status
    assert download.endswith(f"{runs['value']}.lora-sweep.json") and os.path.isfile(download)
    run_dir = os.path.join(lab_ui.runs_root(lab), runs["value"])
    sha = ls.load_run(run_dir)["sweep_sha256"]
    assert hashlib.sha256(open(download, "rb").read()).hexdigest() == sha    # The download is the exact file
    jobs = [ls.Job(**j) for j in ls.load_run(run_dir)["jobs"]]
    folder = results_folder(tmp_path, sha, jobs)
    status, grids, runs2, _ = lab_ui.import_gradient_results(lab, folder)
    assert status.startswith("✅ Imported 12 image(s)") and runs2["value"] == runs["value"]
    assert len(grids["value"]) == 2
    assert lab_ui.import_gradient_results(lab, str(tmp_path / "nope"))[0].startswith("⚠️")


def test_lab_export_refusals(lab):
    assert lab_ui.export_gradient_sweep(lab, "m", [], "", "1", False, "vs", "1", None, 0, "x")[0].startswith("Pick at least one")
    status, download, *_ = lab_ui.export_gradient_sweep(lab, "", [], "", "1", True, "vs", "1", None, 0, "x")
    assert status.startswith("⚠️") and "model" in status and download is None
    assert not ls.list_runs(lab_ui.runs_root(lab))                            # No empty run left behind
