#!/usr/bin/env python3
"""Fasst Footprint-Proben (results/mem/*.json) und Mehrnutzer-Läufe (results/multi/*.json) als Markdown- und
HTML-Tabellen zusammen.

  python3 bench/report_speed.py            # Markdown nach stdout
  python3 bench/report_speed.py --html     # HTML-Tabellen (für docs/benchmarks.html)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MEM = ROOT / "bench" / "results" / "mem"
MULTI = ROOT / "bench" / "results" / "multi"


def fmt(v, digits=1, suffix=""):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}{suffix}".replace(".", ",")
    return f"{v}{suffix}"


def mem_rows():
    rows = []
    for p in sorted(MEM.glob("*.json")):
        d = json.loads(p.read_text())
        m = re.search(r"Verbrauch ([\d.]+) GiB.*Peak GTT ([\d.]+) GiB", d.get("guard_summary", ""))
        rows.append({
            "name": d["name"], "preset": d.get("preset"), "load_s": d.get("load_s"), "pp": d.get("pp"), "tg": d.get("tg"),
            "n_gen": d.get("n_gen"), "draft_n": d.get("draft_n"), "draft_acc": d.get("draft_acc"),
            "footprint": float(m.group(1)) if m else None, "gtt": float(m.group(2)) if m else None,
            "estimate": d.get("estimate_gib"), "error": d.get("error") or d.get("req_error"),
        })
    return rows


def multi_rows():
    out = []
    for p in sorted(MULTI.glob("*.json")):
        d = json.loads(p.read_text())
        crash = ""
        log = p.with_suffix(".log")
        if log.is_file():
            m = re.search(r"GGML_ASSERT\(([^)]*)\) failed", log.read_text(errors="replace"))
            if m and all(l.get("errors") for l in d.get("levels", [])):
                crash = f"Absturz: GGML_ASSERT({m.group(1)})"
        out.append({"tag": d["tag"], "levels": d.get("levels", []), "crash": crash})
    return out


def dsum(l):
    return l.get("decode_sum_tps") if l.get("decode_sum_tps") is not None else (l["per_user_tps"] * l["users"])


def acc(r):
    return f"{r['draft_acc'] / r['draft_n']:.0%}" if r.get("draft_n") else "–"


def md():
    lines = ["### Footprint-Proben (eine Anfrage, 300 Token)", "",
             "| Lauf | Preset | Ladezeit | Prefill | Decode | Draft-Akzeptanz | Bedarf (Δ MemAvailable) | Peak GTT | geschätzt |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in mem_rows():
        lines.append(f"| {r['name']} | {r['preset']} | {fmt(r['load_s'], 0, ' s')} | {fmt(r['pp'], 1, ' t/s')} | {fmt(r['tg'], 1, ' t/s')} | "
                     f"{acc(r)} | {fmt(r['footprint'], 1, ' GiB')} | {fmt(r['gtt'], 1, ' GiB')} | {fmt(r['estimate'], 1, ' GiB')} |"
                     + (f" Fehler: {r['error']}" if r["error"] else ""))
    lines += ["", "### Mehrnutzer (gleichzeitige Streams, je 8k Prompt und 512 Token Ausgabe)", ""]
    for m in multi_rows():
        if m["crash"]:
            lines += [f"**{m['tag']}**: {m['crash']}", ""]
            continue
        lines += [f"**{m['tag']}**", "", "| Streams | Σ Decode t/s | je Stream t/s | Prefill je Stream t/s | TTFT | inkl. Prefill t/s | Draft-Akzeptanz | Mix-ups | Fehler |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for l in m["levels"]:
            lines.append(f"| {l['users']} | {fmt(dsum(l))} | {fmt(l['per_user_tps'])} | {fmt(l.get('pp_tps'))} | {fmt(l['ttft_s'], 1, ' s')} | {fmt(l['total_tps'])} | "
                         f"{(f'{l[chr(97)+chr(99)+chr(99)+chr(101)+chr(112)+chr(116)]:.0%}' if l.get('accept') is not None else '–')} | {l['mixups']} | {l['errors']} |")
        lines.append("")
    return "\n".join(lines)


def html():
    out = ['<div class="tablewrap"><table>',
           "<tr><th>Lauf</th><th>Preset</th><th class=\"num\">Ladezeit</th><th class=\"num\">Prefill</th><th class=\"num\">Decode</th>"
           "<th class=\"num\">Akzeptanz</th><th class=\"num\">Bedarf</th><th class=\"num\">Peak GTT</th><th class=\"num\">geschätzt</th></tr>"]
    for r in mem_rows():
        out.append(f"<tr><td>{r['name']}</td><td>{r['preset']}</td><td class=\"num\">{fmt(r['load_s'], 0, ' s')}</td>"
                   f"<td class=\"num\">{fmt(r['pp'], 1, ' t/s')}</td><td class=\"num\">{fmt(r['tg'], 1, ' t/s')}</td><td class=\"num\">{acc(r)}</td>"
                   f"<td class=\"num\">{fmt(r['footprint'], 1, ' GiB')}</td><td class=\"num\">{fmt(r['gtt'], 1, ' GiB')}</td><td class=\"num\">{fmt(r['estimate'], 1, ' GiB')}</td></tr>")
    out.append("</table></div>")
    for m in multi_rows():
        if m["crash"]:
            out += [f"<h3>{m['tag']}</h3>", f'<p class="bad">{m["crash"]}</p>']
            continue
        out += [f"<h3>{m['tag']}</h3>", '<div class="tablewrap"><table>',
                "<tr><th class=\"num\">Streams</th><th class=\"num\">Σ Decode t/s</th><th class=\"num\">je Stream</th><th class=\"num\">Prefill je Stream</th><th class=\"num\">TTFT</th><th class=\"num\">inkl. Prefill</th>"
                "<th class=\"num\">Akzeptanz</th><th class=\"num\">Mix-ups</th><th class=\"num\">Fehler</th></tr>"]
        for l in m["levels"]:
            a = f"{l['accept']:.0%}" if l.get("accept") is not None else "–"
            out.append(f"<tr><td class=\"num\">{l['users']}</td><td class=\"num\">{fmt(dsum(l))}</td><td class=\"num\">{fmt(l['per_user_tps'])}</td><td class=\"num\">{fmt(l.get('pp_tps'))}</td>"
                       f"<td class=\"num\">{fmt(l['ttft_s'], 1, ' s')}</td><td class=\"num\">{fmt(l['total_tps'])}</td><td class=\"num\">{a}</td><td class=\"num\">{l['mixups']}</td><td class=\"num\">{l['errors']}</td></tr>")
        out.append("</table></div>")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", action="store_true")
    a = ap.parse_args()
    print(html() if a.html else md())
