# ArtAgent/core/lab/experiments.py
"""
Experiment records (SQLite): what was trained and how it scored, so LoRA tests accumulate
into evidence instead of one-off grids.

One row per experiment links: the dataset folder and its caption snapshot, the caption set and
how it was made, the trainer and its settings (entered by hand), the LoRA files, the validation
set and version, the LoRA Lab run folder (grids), metric results, vote results and notes.
Reports export as Markdown and HTML into the run folder (or experiments/reports/).
"""
from __future__ import annotations

import html
import json
import os
import sqlite3
import time

from ..utils import get_absolute_path
from ..version import APP_VERSION

DB_FILE = os.path.join("experiments", "artagents_experiments.sqlite")
FIELDS = ["name", "dataset", "snapshot", "caption_set", "caption_set_settings", "trainer", "trainer_settings",
          "lora_paths", "validation_set", "validation_version", "run_dir", "metrics", "votes", "notes"]
LIST_HEADERS = ["ID", "Name", "Created", "Caption set", "Trainer", "LoRA(s)", "Validation", "Top vote"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT NOT NULL, updated TEXT NOT NULL, app_version TEXT,
    name TEXT NOT NULL, dataset TEXT, snapshot TEXT, caption_set TEXT, caption_set_settings TEXT,
    trainer TEXT, trainer_settings TEXT, lora_paths TEXT, validation_set TEXT, validation_version TEXT,
    run_dir TEXT, metrics TEXT, votes TEXT, notes TEXT
);
"""


def db_path() -> str:
    path = get_absolute_path(DB_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _connect(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or db_path())
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def save(record: dict, exp_id: int | None = None, path: str | None = None) -> int:
    """Creates (exp_id None) or updates an experiment. Returns its id."""
    if not (record.get("name") or "").strip():
        raise ValueError("Give the experiment a name.")
    values = {k: _text(record.get(k)) for k in FIELDS}
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with _connect(path) as conn:
        if exp_id:
            given = {k: v for k, v in values.items() if k in record}  # Fields not given keep their stored value
            sets = ", ".join(f"{k} = ?" for k in given)
            cur = conn.execute(f"UPDATE experiments SET {sets}, updated = ? WHERE id = ?", [*given.values(), now, int(exp_id)])
            if cur.rowcount == 0:
                raise ValueError(f"No experiment with id {exp_id}.")
            return int(exp_id)
        cols = ", ".join(FIELDS)
        marks = ", ".join("?" for _ in FIELDS)
        cur = conn.execute(f"INSERT INTO experiments (created, updated, app_version, {cols}) VALUES (?, ?, ?, {marks})",
                           [now, now, APP_VERSION, *values.values()])
        return int(cur.lastrowid)


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def get(exp_id: int, path: str | None = None) -> dict:
    with _connect(path) as conn:
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (int(exp_id),)).fetchone()
    if not row:
        raise ValueError(f"No experiment with id {exp_id}.")
    return dict(row)


def delete(exp_id: int, path: str | None = None) -> None:
    with _connect(path) as conn:
        conn.execute("DELETE FROM experiments WHERE id = ?", (int(exp_id),))


def list_rows(path: str | None = None) -> list[list]:
    with _connect(path) as conn:
        rows = conn.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
    out = []
    for r in rows:
        top = ""
        try:
            votes = json.loads(r["votes"] or "[]")
            if votes:
                top = f"{votes[0][0]} ({votes[0][1]})"
        except (ValueError, IndexError, TypeError):
            pass
        loras = ", ".join(os.path.basename(p) for p in (r["lora_paths"] or "").splitlines() if p.strip())
        validation = f"{r['validation_set']} v{r['validation_version']}" if r["validation_set"] else ""
        out.append([r["id"], r["name"], r["created"], r["caption_set"], r["trainer"], loras, validation, top])
    return out


def _table_md(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "_none_"
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def report_markdown(exp: dict, grids: list[str] | None = None, base_dir: str | None = None) -> str:
    from .metrics import SCORE_HEADERS
    from .voting import RESULT_HEADERS

    def js(field):
        try:
            return json.loads(exp.get(field) or "[]")
        except ValueError:
            return []

    lines = [f"# {exp['name']}", "", f"Experiment #{exp['id']} · created {exp['created']} · ArtAgents {exp.get('app_version') or APP_VERSION}", ""]
    lines += ["## Setup", "",
              f"- **Dataset:** {exp.get('dataset') or '—'}" + (f" (caption snapshot `{exp['snapshot']}`)" if exp.get("snapshot") else ""),
              f"- **Caption set:** {exp.get('caption_set') or '—'}",
              f"- **Trainer:** {exp.get('trainer') or '—'}",
              f"- **LoRA files:** {', '.join(os.path.basename(p) for p in (exp.get('lora_paths') or '').splitlines() if p.strip()) or '—'}",
              f"- **Validation prompts:** {exp.get('validation_set') or '—'}" + (f" v{exp['validation_version']}" if exp.get("validation_version") else ""),
              f"- **Test run:** {exp.get('run_dir') or '—'}", ""]
    if exp.get("caption_set_settings"):
        lines += ["### How the caption set was made", "", "```json", exp["caption_set_settings"], "```", ""]
    if exp.get("trainer_settings"):
        lines += ["### Trainer settings", "", "```", exp["trainer_settings"], "```", ""]
    metrics = js("metrics")
    lines += ["## Metrics (experimental)", "", _table_md(SCORE_HEADERS, metrics), ""]
    votes = js("votes")
    lines += ["## Blind votes", "", _table_md(RESULT_HEADERS, votes), ""]
    if exp.get("notes"):
        lines += ["## Notes", "", exp["notes"], ""]
    if grids:
        lines += ["## Grids", ""]
        for g in grids:
            rel = os.path.relpath(g, base_dir) if base_dir else g
            lines.append(f"![{os.path.basename(g)}]({rel.replace(os.sep, '/')})")
        lines.append("")
    return "\n".join(lines)


def markdown_to_html(md: str, title: str) -> str:
    """Small converter for the report's own Markdown (headings, lists, tables, code, images)."""
    out, in_code, in_table, in_list = [], False, False, False

    def inline(t: str) -> str:
        import re
        t = html.escape(t)
        t = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2" style="max-width:100%">', t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        return re.sub(r"(?<![\w/])_([^_]+)_(?!\w)", r"<em>\1</em>", t)  # Not inside file names

    for line in md.splitlines():
        if line.startswith("```"):
            out.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if in_list and not line.startswith("- "):
            out.append("</ul>")
            in_list = False
        if line.startswith("|"):
            cells = [c.strip().replace("\\|", "|") for c in line.strip().strip("|").split(" | ")]
            if set(line.replace("|", "").strip()) <= {"-"}:
                continue
            if not in_table:
                out.append("<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
        elif line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    style = ("body{font-family:system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}"
             "table{border-collapse:collapse;margin:1em 0}td,th{border:1px solid #ccc;padding:4px 8px;text-align:left}"
             "pre{background:#f5f5f5;padding:1em;overflow:auto}")
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{style}</style></head><body>\n" + "\n".join(out) + "\n</body></html>\n")


def export_report(exp_id: int, path: str | None = None) -> tuple[str, str]:
    """Writes report.md and report.html (into the run folder when there is one). Returns their paths."""
    exp = get(exp_id, path)
    run_dir = exp.get("run_dir") or ""
    out_dir = run_dir if run_dir and os.path.isdir(run_dir) else get_absolute_path(os.path.join("experiments", "reports"))
    os.makedirs(out_dir, exist_ok=True)
    grids = sorted(os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.startswith("grid_") and f.endswith(".png")) \
        if run_dir and os.path.isdir(run_dir) else []
    md = report_markdown(exp, grids, out_dir)
    stem = f"experiment_{exp['id']}_report"
    md_path, html_path = os.path.join(out_dir, stem + ".md"), os.path.join(out_dir, stem + ".html")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(markdown_to_html(md, exp["name"]))
    return md_path, html_path
