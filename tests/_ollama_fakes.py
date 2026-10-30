# ArtAgent/tests/_ollama_fakes.py
"""
Shared fakes for tests that talk to Ollama through core.ollama_client.

FakeOllama is installed as the side effect of a patched
`core.ollama_client.requests.request` and answers by URL path, recording
every request so tests can inspect payloads.
"""
import json
from unittest.mock import MagicMock

import requests

REQUEST_PATH = "core.ollama_client.requests.request"


def make_response(status_code=200, json_data=None, lines=None, text=None):
    """A MagicMock requests.Response with json(), iter_lines() and close()."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.reason = "OK" if status_code < 400 else "Error"
    if json_data is not None:
        resp.json.return_value = json_data
        resp.text = json.dumps(json_data)
    else:
        resp.json.side_effect = ValueError("no json")
        resp.text = text or ""
    resp.iter_lines.return_value = iter(lines or [])
    return resp


def chat_lines(*contents, thinking=None, done=True):
    """NDJSON /api/chat lines: optional thinking chunks, then content chunks, then done."""
    lines = []
    for t in thinking or []:
        lines.append(json.dumps({"message": {"role": "assistant", "content": "", "thinking": t}, "done": False}))
    for c in contents:
        lines.append(json.dumps({"message": {"role": "assistant", "content": c}, "done": False}))
    if done:
        lines.append(json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}))
    return lines


class FakeOllama:
    """Routes fake HTTP calls by path. Configure the attributes, then use as side_effect."""

    def __init__(self, chat=None, capabilities=None, tags=None, ps=None, chat_status=200, chat_json=None):
        self.chat = chat if chat is not None else chat_lines("Hello", " world")
        self.capabilities = capabilities if capabilities is not None else {}
        self.tags = tags if tags is not None else []
        self.ps = ps if ps is not None else []
        self.chat_status = chat_status
        self.chat_json = chat_json
        self.calls = []        # (method, url, kwargs)
        self.chat_responses = []

    def payloads(self, path):
        return [kw.get("json") for m, url, kw in self.calls if url.endswith(path)]

    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/api/show"):
            model = (kwargs.get("json") or {}).get("model")
            return make_response(json_data={"capabilities": self.capabilities.get(model, ["completion"])})
        if url.endswith("/api/chat"):
            if self.chat_status >= 400:
                resp = make_response(status_code=self.chat_status, json_data=self.chat_json)
            else:
                resp = make_response(lines=list(self.chat))
            self.chat_responses.append(resp)
            return resp
        if url.endswith("/api/tags"):
            return make_response(json_data={"models": self.tags})
        if url.endswith("/api/ps"):
            if isinstance(self.ps, Exception):
                raise self.ps
            return make_response(json_data={"models": self.ps})
        if url.endswith("/api/generate"):
            return make_response(json_data={"done": True, "done_reason": "unload"})
        if url.endswith("/api/version"):
            return make_response(json_data={"version": "0.40.1"})
        return make_response(status_code=404, json_data={"error": "not found"})
