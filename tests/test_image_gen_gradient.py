# ArtAgent/tests/test_image_gen_gradient.py
"""GRADIENT backend: payloads, job submission, polling, image fetch, cancelling, token, and the LoRA sweep's queue window."""
import io
import json
import os
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

from core.image_gen import BACKENDS, DEFAULT_URLS, GradientBackend
from core.image_gen import gradient as gmod
from core.image_gen.base import GeneratedImage, GenProgress, GenResult, ImageGenError, ImageGenRequest
from core.lab import lora_sweep as ls

BASE = "http://127.0.0.1:8000"
API = BASE + "/api/v1"


class FakeResp:
    def __init__(self, status=200, data=None, content=b"", text=None):
        self.status_code = status
        self._data = data
        self.content = content
        self.text = text if text is not None else (json.dumps(data) if data is not None else "")

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


def png(color=(10, 20, 30), size=(16, 16)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class FakeGradient:
    """A tiny in-memory GRADIENT: a queue of jobs that finish after `steps` polls."""

    def __init__(self, polls_to_finish=1, fail=None, skipped=None, token=None):
        self.jobs = {}
        self.order = []
        self.calls = []
        self.polls_to_finish = polls_to_finish
        self.fail = fail or set()          # Payload seeds whose jobs fail
        self.skipped = skipped or {}       # LoRA name -> reported as not applied
        self.token = token
        self.local_calls = 0
        self.mock = MagicMock()
        self.mock.exceptions = requests.exceptions
        self.mock.get.side_effect = lambda url, **kw: self.route("GET", url, **kw)
        self.mock.post.side_effect = lambda url, **kw: self.route("POST", url, **kw)

    def route(self, method, url, **kw):
        path = url.replace(API, "")
        self.calls.append((method, path, kw))
        if self.token and kw.get("headers", {}).get("Authorization") != f"Bearer {self.token}":
            return FakeResp(401, {"detail": "Unauthorized"})
        if method == "GET" and path == "/status":
            return FakeResp(200, {"status": "ok", "version": "0.9.0"})
        if method == "GET" and path == "/system/info":
            return FakeResp(200, {"version": "0.9.0", "device": {"name": "cpu"}})
        if method == "GET" and path == "/system/models/diffusion":
            return FakeResp(200, ["tiny-sd", "flux-dev"])
        if method == "GET" and path == "/system/models/loras":
            return FakeResp(200, ["style_e04.safetensors"])
        if method == "GET" and path == "/models/local":
            kind = (kw.get("params") or {}).get("type")
            self.local_calls += 1
            models = {"diffusers": [{"name": "tiny-sd", "defaults": None},
                                    {"name": "flux-dev", "defaults": {"steps": 28, "guidance_scale": 3.5, "width": None}}],
                      "checkpoints": [{"name": "turbo.safetensors", "defaults": {"steps": 4, "guidance_scale": 0.0}}]}
            return FakeResp(200, models.get(kind, []))
        if method == "GET" and path == "/system/models/vae":
            return FakeResp(200, ["vae-ft-mse"])
        if method == "GET" and path == "/system/schedulers":
            return FakeResp(200, ["Default", "EulerDiscreteScheduler"])
        if method == "GET" and path == "/system/quantization":
            model = (kw.get("params") or {}).get("model_id")
            return FakeResp(200, {"options": [
                {"value": "bf16", "label": "BF16", "available": True},
                {"value": "bnb-4bit", "label": "4-bit (bitsandbytes)", "available": model == "flux-dev"},
                {"value": "", "label": "broken", "available": True}]})
        if method == "POST" and path == "/queue":
            body = kw["json"]
            job_id = f"job{len(self.jobs) + 1}"
            self.jobs[job_id] = {"id": job_id, "status": "queued", "label": body.get("label"), "body": body, "polls": 0}
            self.order.append(job_id)
            return FakeResp(200, {"jobs": [{"id": job_id}]})
        if method == "GET" and path == "/queue":
            return FakeResp(200, {"jobs": [{"id": j["id"], "status": j["status"], "label": j["label"]} for j in self.jobs.values()]})
        if method == "POST" and path.startswith("/queue/") and path.endswith("/cancel"):
            job = self.jobs.get(path.split("/")[2])
            if not job:
                return FakeResp(404, {"detail": "No job"})
            if job["status"] in ("queued", "running"):
                job["status"] = "cancelled"
            return FakeResp(200, {"id": job["id"], "status": job["status"]})
        if method == "GET" and path.startswith("/queue/"):
            job = self.jobs.get(path.split("/")[2])
            if not job:
                return FakeResp(404, {"detail": "No job"})
            return FakeResp(200, self.poll(job))
        if method == "GET" and path.startswith("/history/outputs/") and path.endswith("/image"):
            return FakeResp(200, content=png())
        return FakeResp(404, {"detail": "Not Found"})

    def poll(self, job):
        if job["status"] in ("queued", "running"):
            job["polls"] += 1
            payload = job["body"]["payload"]
            if job["polls"] == 1 and self.polls_to_finish > 1:
                job["status"] = "queued"
                return {"id": job["id"], "status": "queued", "position": 2}
            if job["polls"] < self.polls_to_finish:
                job["status"] = "running"
                return {"id": job["id"], "status": "running", "progress": {"step": 3, "total": 4, "phase": "denoising"}}
            if payload["seed"] in self.fail:
                job["status"] = "failed"
                job["error"] = "CUDA out of memory"
            else:
                job["status"] = "completed"
                skipped = [l["name"] for l in payload["loras"] if l["name"] in self.skipped]
                job["result"] = {"run_uuid": "run-" + job["id"], "outputs": [{"id": 100 + len(self.order), "seed": payload["seed"]}],
                                 "performance_ms": 5, **({"loras_skipped": skipped} if skipped else {})}
        view = {k: v for k, v in job.items() if k not in ("body", "polls")}
        return view


@pytest.fixture(autouse=True)
def no_cached_defaults():
    gmod._defaults_cache.clear()
    yield
    gmod._defaults_cache.clear()


@pytest.fixture
def fake():
    server = FakeGradient()
    with patch.object(gmod, "requests", server.mock):
        yield server


def backend(**kw):
    b = GradientBackend(BASE + "/api/v1/", **kw)
    b.poll_interval = 0
    return b


def req(**kw):
    base = dict(prompt="a  red\ncat", model="tiny-sd", width=63, height=100, seed=7)
    base.update(kw)
    return ImageGenRequest(**base)


# ---------- registry and payloads ----------

def test_registered_with_default_url():
    assert BACKENDS["GRADIENT"] is GradientBackend and DEFAULT_URLS["GRADIENT"] == BASE


def test_base_url_suffix_stripped():
    assert backend().base_url == BASE
    assert GradientBackend(BASE + "/docs").base_url == BASE


def test_round8():
    assert [gmod.round8(v) for v in (63, 64, 7, 0, 1023)] == [56, 64, 8, 8, 1016]


def test_build_payload_basics():
    p = backend().build_payload(req(loras=[("style.safetensors", 0.8), ("/abs/x.safetensors", 1)], steps=12, cfg_scale=4.5,
                                    sampler="EulerDiscreteScheduler", negative_prompt="blurry"))
    assert p["model_id"] == "tiny-sd" and p["width"] == 56 and p["height"] == 96 and p["seed"] == 7
    assert p["loras"] == [{"name": "style.safetensors", "weight": 0.8}, {"name": "/abs/x.safetensors", "weight": 1.0}]
    assert p["num_inference_steps"] == 12 and p["guidance_scale"] == 4.5 and p["scheduler_id"] == "EulerDiscreteScheduler"
    assert p["num_outputs"] == 1 and p["negative_prompt"] == "blurry"
    assert [backend().build_payload(req(batch_size=n))["num_outputs"] for n in (0, 3, 9)] == [1, 3, 4]
    assert "quantization" not in p and "vae_id" not in p


def test_build_payload_defaults_left_to_gradient():
    p = backend().build_payload(req(steps=None, cfg_scale=None, sampler=None))
    assert "num_inference_steps" not in p and "guidance_scale" not in p and "scheduler_id" not in p
    assert p["loras"] == []                    # A real no-LoRA baseline


def test_build_payload_extra_options():
    p = backend().build_payload(req(cfg_scale=5, extra={"guidance": " linear(9,3) ", "quantization": "bnb-4bit", "vae_id": "vae-ft-mse"}))
    assert p["guidance_scale"] == "linear(9,3)" and p["quantization"] == "bnb-4bit" and p["vae_id"] == "vae-ft-mse"
    assert backend().build_payload(req(cfg_scale=5, extra={"guidance": "3.5"}))["guidance_scale"] == 3.5
    assert backend().build_payload(req(cfg_scale=5, extra={"guidance": ""}))["guidance_scale"] == 5.0


def test_build_payload_needs_model():
    with pytest.raises(ImageGenError, match="Pick a GRADIENT model"):
        backend().build_payload(req(model=None))


def test_random_seed_resolved():
    p = backend().build_payload(req(seed=-1))
    assert isinstance(p["seed"], int) and p["seed"] >= 0


# ---------- information ----------

def test_check_and_lists(fake):
    b = backend()
    ok, msg = b.check()
    assert ok and msg == "GRADIENT 0.9.0 · cpu · 2 model(s)"
    assert b.list_models() == ["tiny-sd", "flux-dev"] and b.list_loras() == ["style_e04.safetensors"]
    assert b.list_samplers()[0] == "Default" and b.list_vaes() == ["vae-ft-mse"]
    assert b.list_precisions("tiny-sd") == [("BF16", "bf16")]
    assert b.list_precisions("flux-dev") == [("BF16", "bf16"), ("4-bit (bitsandbytes)", "bnb-4bit")]


def test_check_unreachable():
    mock = MagicMock()
    mock.exceptions = requests.exceptions
    mock.get.side_effect = requests.exceptions.ConnectionError("refused")
    with patch.object(gmod, "requests", mock):
        ok, msg = backend().check()
    assert not ok and msg


def test_http_error_detail(fake):
    fake.mock.get.side_effect = lambda url, **kw: FakeResp(500, {"detail": "model folder missing"})
    with pytest.raises(ImageGenError, match="HTTP 500.*model folder missing"):
        backend().list_models()


# ---------- token ----------

def test_token_sent_and_refusal_explained(monkeypatch):
    server = FakeGradient(token="s3cret")
    monkeypatch.delenv(gmod.TOKEN_ENV, raising=False)
    with patch.object(gmod, "requests", server.mock):
        ok, msg = backend().check()
        assert not ok and gmod.TOKEN_ENV in msg and "token" in msg
        assert backend(token="s3cret").check()[0]
        monkeypatch.setenv(gmod.TOKEN_ENV, " s3cret ")
        assert backend().check()[0]


def test_no_token_no_header(fake, monkeypatch):
    monkeypatch.delenv(gmod.TOKEN_ENV, raising=False)
    backend().list_models()
    assert "Authorization" not in fake.calls[-1][2]["headers"]


def test_openai_key_never_used(fake, monkeypatch):
    monkeypatch.delenv(gmod.TOKEN_ENV, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-nope")
    backend().list_models()
    assert "Authorization" not in fake.calls[-1][2]["headers"]


# ---------- jobs ----------

def test_submit_body(fake):
    b = backend()
    job_id = b.submit(req(extra={"label": "L", "group": "g1", "tags": ["t"], "origin": {"app": "ArtAgents"}, "quantization": "bf16"}))
    body = fake.jobs[job_id]["body"]
    assert body["kind"] == "text_to_image" and body["count"] == 1 and body["vary_seed"] is False
    assert body["label"] == "L" and body["group"] == "g1" and body["tags"] == ["t"] and body["origin"] == {"app": "ArtAgents"}
    assert body["payload"]["quantization"] == "bf16" and "quantization" not in body
    assert b._active == [job_id]


def test_submit_default_label(fake):
    job_id = backend().submit(req())
    body = fake.jobs[job_id]["body"]
    assert body["label"] == "ArtAgents · a red cat" and "group" not in body and "tags" not in body


def test_submit_without_job_id(fake):
    fake.mock.post.side_effect = lambda url, **kw: FakeResp(200, {"jobs": []})
    with pytest.raises(ImageGenError, match="no job id"):
        backend().submit(req())


def test_generate_iter_progress_and_image():
    server = FakeGradient(polls_to_finish=3)
    with patch.object(gmod, "requests", server.mock):
        b = backend()
        events = list(b.generate_iter(req()))
    progress = [e for e in events if isinstance(e, GenProgress)]
    assert progress[0].message == "Queued in GRADIENT"
    assert any("position 3" in p.message for p in progress)
    assert any(p.fraction == 0.75 and "denoising 3/4" in p.message for p in progress)
    result = events[-1]
    assert isinstance(result, GenResult) and len(result.images) == 1
    img = result.images[0]
    assert img.image.size == (16, 16) and img.seed == 7
    assert img.info["run_uuid"] == "run-job1" and img.info["gradient_job"] == "job1" and img.info["output_id"]
    assert img.info["loras_skipped"] == [] and b._active == []


def test_generate_iter_failed_job():
    server = FakeGradient(fail={7})
    with patch.object(gmod, "requests", server.mock):
        b = backend()
        with pytest.raises(ImageGenError, match="CUDA out of memory"):
            list(b.generate_iter(req()))
    assert b._active == []


def test_generate_iter_cancelled_elsewhere(fake):
    b = backend()
    gen = b.generate_iter(req())
    next(gen)                                    # Queued
    fake.jobs["job1"]["status"] = "cancelled"
    with pytest.raises(ImageGenError, match="cancelled"):
        list(gen)


def test_generate_iter_closed_early_cancels(fake):
    fake.polls_to_finish = 5
    b = backend()
    gen = b.generate_iter(req())
    next(gen)
    next(gen)
    gen.close()
    assert fake.jobs["job1"]["status"] == "cancelled" and b._active == []


def test_wait_iter_timeout_cancels(fake):
    fake.polls_to_finish = 99
    b = backend(generation_timeout=-1)
    job_id = b.submit(req())
    with pytest.raises(ImageGenError, match="in time"):
        list(b.wait_iter(job_id))
    assert fake.jobs[job_id]["status"] == "cancelled"


def test_finished_without_outputs(fake):
    b = backend()
    job_id = b.submit(req())
    fake.jobs[job_id].update(status="completed", result={"outputs": []})
    with pytest.raises(ImageGenError, match="no image"):
        list(b.wait_iter(job_id))


def test_unreadable_image(fake):
    route = fake.route

    def bad_image(method, url, **kw):
        resp = route(method, url, **kw)
        return FakeResp(200, content=b"<html>not an image</html>") if url.endswith("/image") else resp
    fake.mock.get.side_effect = lambda url, **kw: bad_image("GET", url, **kw)
    with pytest.raises(ImageGenError, match="could not be read"):
        list(backend().generate_iter(req()))


def test_image_fetch_http_error(fake):
    route = fake.route
    fake.mock.get.side_effect = lambda url, **kw: (FakeResp(404, {"detail": "gone"}) if url.endswith("/image")
                                                   else route("GET", url, **kw))
    with pytest.raises(ImageGenError, match="Could not fetch GRADIENT image"):
        list(backend().generate_iter(req()))


def test_loras_skipped_reported():
    server = FakeGradient(skipped={"flux_style.safetensors"})
    with patch.object(gmod, "requests", server.mock):
        result = list(backend().generate_iter(req(loras=[("flux_style.safetensors", 1.0)])))[-1]
    assert result.images[0].info["loras_skipped"] == ["flux_style.safetensors"]


def test_cancel_unknown_job_is_quiet(fake):
    backend().cancel("nope")


def test_interrupt_cancels_only_artagents_jobs(fake):
    fake.polls_to_finish = 9
    b = backend()
    mine = b.submit(req())
    other = backend().submit(req(extra={"label": "ArtAgents · image tab"}))
    user = backend().submit(req(extra={"label": "my own render"}))
    fake.jobs["done"] = {"id": "done", "status": "completed", "label": "ArtAgents · old", "body": {}, "polls": 0}
    b.interrupt()
    assert fake.jobs[mine]["status"] == "cancelled" and fake.jobs[other]["status"] == "cancelled"
    assert fake.jobs[user]["status"] == "queued" and fake.jobs["done"]["status"] == "completed"
    assert b._active == []


def test_interrupt_unreachable_is_quiet():
    mock = MagicMock()
    mock.exceptions = requests.exceptions
    mock.get.side_effect = mock.post.side_effect = requests.exceptions.ConnectionError("down")
    with patch.object(gmod, "requests", mock):
        backend().interrupt()


# ---------- LoRA sweep through GRADIENT ----------

VSET = {"trigger": "txcl", "seeds": [1, 2], "defaults": {"width": 63, "height": 64, "steps": 4, "cfg": 5.0, "negative": "bad"},
        "prompts": [{"bucket": "held-out", "prompt": "[trigger] cat", "negative": "", "flags": ""},
                    {"bucket": "unrelated", "prompt": "a car", "negative": "", "flags": ""}]}


def test_gradient_extra(tmp_path):
    job = ls.build_jobs(VSET, ["a.safetensors"], [0.5])[1]
    extra = ls.gradient_extra(job, str(tmp_path / "20260101-120000-mine") + "/", {"name": "mine"},
                              {"quantization": "bf16", "vae_id": "", "guidance": None})
    assert extra["quantization"] == "bf16" and "vae_id" not in extra and "guidance" not in extra
    assert extra["label"].startswith("ArtAgents · LoRA test · mine · ") and extra["label"].endswith(f"seed {job.seed}")
    assert extra["group"] == "ArtAgents · LoRA test · 20260101-120000-mine" and extra["tags"] == ["ArtAgents/LoRA test/mine"]
    assert extra["origin"] == {"app": "ArtAgents", "run": "20260101-120000-mine", "job_index": job.index,
                               "prompt_index": job.prompt_index, "bucket": job.bucket, "condition": job.condition}


def test_request_for_gradient_baseline_has_no_lora():
    base, lora = ls.build_jobs(VSET, ["a.safetensors"], [0.5], max_prompts=1)[:2]
    r = ls.request_for(base, "GRADIENT", "tiny-sd", None, "Euler", "a.safetensors", {"quantization": "bf16"})
    assert r.loras == [] and r.sampler == "Euler" and r.extra == {"quantization": "bf16"}
    assert ls.request_for(lora, "GRADIENT", "tiny-sd", None, None, "a.safetensors").loras == [("a.safetensors", 0.5)]


def _sweep(tmp_path, server, loras=("b.safetensors", "a.safetensors"), close_after=None, options=None):
    jobs = ls.build_jobs(VSET, list(loras), [0.5, 1.0])
    run_dir = ls.new_run_dir(str(tmp_path / "runs"), "g")
    with patch.object(gmod, "requests", server.mock):
        b = backend()
        gen = ls.run_iter(b, "GRADIENT", jobs, run_dir, {"name": "g"}, model="tiny-sd", sampler="Euler",
                          options=options or {"quantization": "bf16"})
        events = []
        for event in gen:
            events.append(event)
            if close_after and len(events) == close_after:
                gen.close()
                break
    return jobs, run_dir, b, events


def test_sweep_through_gradient(tmp_path):
    server = FakeGradient()
    jobs, run_dir, b, events = _sweep(tmp_path, server)
    assert [e[0] for e in events] == list(range(1, len(jobs) + 1)) and len(server.jobs) == len(jobs)
    data = ls.load_run(run_dir)
    assert data["backend"] == "GRADIENT" and data["sampler"] == "Euler" and data["options"] == {"quantization": "bf16"}
    assert all(r["path"] and r["error"] is None for r in data["results"])
    assert {r["job"] for r in data["results"]} == {j.index for j in jobs}
    assert all(r["remote"]["gradient_job"] and r["remote"]["run_uuid"] and r["remote"]["output_id"] for r in data["results"])
    bodies = [server.jobs[i]["body"] for i in server.order]
    # Ordered by condition: baselines first (no LoRAs), then a.safetensors, then b.safetensors
    names = [(p["payload"]["loras"] or [{"name": None}])[0]["name"] for p in bodies]
    assert names == [None] * 4 + ["a.safetensors"] * 8 + ["b.safetensors"] * 8
    assert all(p["payload"]["width"] == 56 and p["payload"]["quantization"] == "bf16" for p in bodies)
    assert all(p["group"] == f"ArtAgents · LoRA test · {os.path.basename(run_dir)}" and p["label"].startswith("ArtAgents · LoRA test · g") for p in bodies)
    assert len(os.listdir(run_dir)) == len(jobs) + 1   # Images + run.json


def test_sweep_keeps_a_small_window_queued(tmp_path):
    server = FakeGradient()
    seen = []
    real_submit = GradientBackend.submit

    def spy(self, request):
        waiting = sum(j["status"] in ("queued", "running") for j in server.jobs.values())
        seen.append(waiting)
        return real_submit(self, request)
    with patch.object(GradientBackend, "submit", spy):
        _sweep(tmp_path, server)
    assert max(seen) < ls.QUEUE_AHEAD and len(seen) == 20


def test_sweep_stopped_cancels_what_is_queued(tmp_path):
    server = FakeGradient()
    jobs, run_dir, b, events = _sweep(tmp_path, server, close_after=3)
    statuses = [server.jobs[i]["status"] for i in server.order]
    assert statuses[:3] == ["completed"] * 3
    assert statuses[3:] and set(statuses[3:]) == {"cancelled"} and len(statuses) <= 3 + ls.QUEUE_AHEAD
    data = ls.load_run(run_dir)
    assert len(data["results"]) == 3 and b._active == []


def test_sweep_failed_and_skipped_jobs_are_errors(tmp_path):
    server = FakeGradient(fail={2}, skipped={"b.safetensors"})
    jobs, run_dir, b, events = _sweep(tmp_path, server)
    data = ls.load_run(run_dir)
    errors = [r for r in data["results"] if r["error"]]
    failed = [r for r in errors if "CUDA out of memory" in r["error"]]
    skipped = [r for r in errors if "LoRA not applied by GRADIENT" in r["error"]]
    assert failed and skipped and all(r["path"] is None for r in errors)
    assert all("b.safetensors" in r["error"] for r in skipped)
    assert sum(1 for e in events if "⚠" in e[2]) == len(errors)
    assert len(data["results"]) == len(jobs)


def test_sweep_submit_error_recorded_and_continues(tmp_path):
    server = FakeGradient()
    route = server.route
    count = {"n": 0}

    def flaky(method, url, **kw):
        if method == "POST" and url.endswith("/queue"):
            count["n"] += 1
            if count["n"] == 2:
                return FakeResp(422, {"detail": "width must be a multiple of 8"})
        return route(method, url, **kw)
    server.mock.post.side_effect = lambda url, **kw: flaky("POST", url, **kw)
    jobs, run_dir, b, events = _sweep(tmp_path, server)
    data = ls.load_run(run_dir)
    errors = [r for r in data["results"] if r["error"]]
    assert len(errors) == 1 and "multiple of 8" in errors[0]["error"] and len(data["results"]) == len(jobs)


def test_sequential_backends_unchanged(tmp_path):
    class Seq:
        def __init__(self):
            self.requests = []

        def generate_iter(self, request):
            self.requests.append(request)
            yield GenResult([GeneratedImage(Image.new("RGB", (8, 8)), request.seed)])
    jobs = ls.build_jobs(VSET, ["a.safetensors"], [1.0])
    run_dir = ls.new_run_dir(str(tmp_path), "s")
    backend_ = Seq()
    list(ls.run_iter(backend_, "Forge / A1111 / SD.Next", jobs, run_dir, {}))
    assert [r.seed for r in backend_.requests] == [j.seed for j in jobs]     # Job order kept
    assert all(r.extra == {} for r in backend_.requests)


# ---------- LoRA Lab UI with GRADIENT ----------

from core import lab_ui  # noqa: E402
from core.lab import validation  # noqa: E402


@pytest.fixture
def lab(tmp_path, fake, monkeypatch):
    monkeypatch.delenv(gmod.TOKEN_ENV, raising=False)
    root = tmp_path / "app"
    root.mkdir()

    def gap(rel):
        return str(root / rel)
    settings = {"image_generation": {"output_dir": str(tmp_path / "outputs" / "images"), "gradient_url": BASE}}
    with patch("core.lab.validation.get_absolute_path", side_effect=gap), patch("core.lab_ui.get_absolute_path", side_effect=gap):
        validation.save_new_version("vs", [["held-out", "[trigger] cat", "", ""], ["unrelated", "a car", "", ""]],
                                    "1, 2", "txcl", {"width": 500, "height": 512})
        yield settings


def test_lab_refresh_gradient(lab, fake):
    models, loras, workflow, status, group, schedulers, precisions, vaes = lab_ui.refresh_lab_backend("GRADIENT", lab)
    assert models["value"] == "tiny-sd" and loras["choices"] == ["style_e04.safetensors"] and not workflow["visible"]
    assert status.startswith("✅ GRADIENT at http://127.0.0.1:8000: GRADIENT 0.9.0 · cpu") and group["visible"]
    assert schedulers["choices"] == [lab_ui.DEFAULT_OPTION, "Default", "EulerDiscreteScheduler"]
    assert precisions["choices"] == [lab_ui.DEFAULT_OPTION, "BF16"] and vaes["choices"] == [lab_ui.DEFAULT_OPTION, "vae-ft-mse"]
    assert all(c["value"] == lab_ui.DEFAULT_OPTION for c in (schedulers, precisions, vaes))


def test_lab_refresh_gradient_token(lab):
    server = FakeGradient(token="tok")
    with patch.object(gmod, "requests", server.mock):
        assert "token" in lab_ui.refresh_lab_backend("GRADIENT", lab)[3]
        assert lab_ui.refresh_lab_backend("GRADIENT", lab, "tok")[3].startswith("✅")


def test_lab_precisions_follow_the_checkpoint(lab):
    flux = lab_ui.gradient_precisions("GRADIENT", "flux-dev", lab)
    assert flux["choices"] == [lab_ui.DEFAULT_OPTION, "BF16", "4-bit (bitsandbytes)"]
    assert lab_ui.gradient_precisions("ComfyUI", "flux-dev", lab) == {"__type__": "update"}
    assert lab_ui.gradient_precisions("GRADIENT", None, lab) == {"__type__": "update"}


def test_lab_gradient_options():
    assert lab_ui.gradient_options(lab_ui.DEFAULT_OPTION, None, "", "  ") == {
        "scheduler": "", "quantization": "", "vae_id": "", "guidance": ""}
    assert lab_ui.gradient_options("Euler", "BF16", "vae", " linear(9,3) ") == {
        "scheduler": "Euler", "quantization": "BF16", "vae_id": "vae", "guidance": "linear(9,3)"}


def test_lab_sweep_gradient(lab, fake):
    events = list(lab_ui.iter_lora_sweep(lab, "GRADIENT", "flux-dev", "ignored.json", ["style_e04.safetensors"], "/abs/e08.safetensors",
                                         "1", True, "vs", "1", None, 0, "g run", "EulerDiscreteScheduler",
                                         "4-bit (bitsandbytes)", "vae-ft-mse", "linear(9,3)"))
    note = events[0][0]
    assert note.startswith("Note:") and "500×512 will be rounded down" in note
    assert "28 steps (recommended for flux-dev)" in note and "CFG" not in note        # The CFG override is set
    assert events[-1][0].startswith("Finished: 12 job(s), 0 error(s).")
    bodies = [fake.jobs[i]["body"] for i in fake.order]
    assert len(bodies) == 12
    p = bodies[0]["payload"]
    assert p["width"] == 496 and p["quantization"] == "bnb-4bit" and p["vae_id"] == "vae-ft-mse"
    assert p["scheduler_id"] == "EulerDiscreteScheduler" and p["guidance_scale"] == "linear(9,3)" and p["model_id"] == "flux-dev"
    assert p["num_inference_steps"] == 28                                             # From the model's defaults
    run = ls.load_run(os.path.join(lab_ui.runs_root(lab), ls.list_runs(lab_ui.runs_root(lab))[0]))
    assert run["workflow"] is None and run["options"]["quantization"] == "bnb-4bit" and run["sampler"] == "EulerDiscreteScheduler"
    assert {(b["payload"]["loras"] or [{"name": None}])[0]["name"] for b in bodies} == {None, "style_e04.safetensors", "/abs/e08.safetensors"}


def test_lab_sweep_gradient_no_note_when_set(lab, fake):
    validation.save_new_version("vs2", [["held-out", "[trigger] cat", "", ""]], "1", "txcl",
                                {"width": 512, "height": 512, "steps": 4, "cfg": 2})
    events = list(lab_ui.iter_lora_sweep(lab, "GRADIENT", "tiny-sd", None, [], "", "", True, "vs2", None, None, 0, "b"))
    assert not any(e[0].startswith("Note:") for e in events) and events[-1][0].startswith("Finished: 1 job(s)")


def test_lab_stop_gradient(lab, fake):
    fake.polls_to_finish = 9
    job = backend().submit(req(extra={"label": "ArtAgents · LoRA test · x"}))
    message = lab_ui.stop_lab_backend("GRADIENT", lab)
    assert fake.jobs[job]["status"] == "cancelled" and message.startswith("⏹ Stopped.")



# ---------- model defaults ----------

def test_model_defaults(fake):
    b = backend()
    assert b.model_defaults("flux-dev") == {"steps": 28, "guidance_scale": 3.5}
    assert b.model_defaults("turbo.safetensors") == {"steps": 4, "guidance_scale": 0.0}
    assert b.model_defaults("tiny-sd") == {} and b.model_defaults("missing") == {} and b.model_defaults(None) == {}
    calls = fake.local_calls
    assert b.model_defaults("flux-dev") == {"steps": 28, "guidance_scale": 3.5} and fake.local_calls == calls  # Cached


def test_build_payload_with_defaults():
    d = {"steps": 4, "guidance_scale": 0.0}
    p = backend().build_payload(req(steps=None, cfg_scale=None), d)
    assert p["num_inference_steps"] == 4 and p["guidance_scale"] == 0.0
    p = backend().build_payload(req(steps=9, cfg_scale=2.0), d)                 # The request wins
    assert p["num_inference_steps"] == 9 and p["guidance_scale"] == 2.0
    p = backend().build_payload(req(steps=None, cfg_scale=None, extra={"guidance": "linear(9,3)"}), d)
    assert p["guidance_scale"] == "linear(9,3)" and p["num_inference_steps"] == 4


def test_submit_uses_model_defaults_only_when_needed(fake):
    b = backend()
    job = b.submit(req(model="turbo.safetensors", steps=None, cfg_scale=None))
    payload = fake.jobs[job]["body"]["payload"]
    assert payload["num_inference_steps"] == 4 and payload["guidance_scale"] == 0.0
    gmod._defaults_cache.clear()
    calls = fake.local_calls
    b.submit(req(model="turbo.safetensors", steps=6, cfg_scale=1.0))
    assert fake.local_calls == calls                                             # Nothing to fill: no lookup
    job = b.submit(req(model="tiny-sd", steps=None, cfg_scale=None))            # No recommendation: GRADIENT decides
    payload = fake.jobs[job]["body"]["payload"]
    assert "num_inference_steps" not in payload and "guidance_scale" not in payload


def test_model_defaults_unreachable_is_empty():
    mock = MagicMock()
    mock.exceptions = requests.exceptions
    mock.get.side_effect = requests.exceptions.ConnectionError("down")
    with patch.object(gmod, "requests", mock):
        assert backend().model_defaults("flux-dev") == {}


def test_lab_defaults_note(fake):
    b = backend()
    note = lab_ui.gradient_defaults_note(b, "flux-dev", True, True)
    assert "28 steps (recommended for flux-dev)" in note and "CFG 3.5 (recommended for flux-dev)" in note
    assert "turbo" not in note
    note = lab_ui.gradient_defaults_note(b, "tiny-sd", True, False)
    assert "25 steps (GRADIENT's default)" in note and "CFG" not in note and "distilled or turbo" in note
