"""Kommandozeile: python3 -m glm53 <befehl>

  presets                         alle Presets
  models                          gefundene Quants, Varianten, Engines, Projektoren
  show   --preset P [Overrides]   Speicherbilanz, Warnungen und fertige Kommandozeile
  cmd    --preset P [Overrides]   nur die Kommandozeile (für Skripte)
  run    --preset P [Overrides]   Server unter bench/memguard.py starten (Log in state/logs)
  variants [--create] [--quant Q] Shard-Sätze je Namensschema anlegen (models/variants)
  gguf-info DATEI                 Header eines GGUF: Architektur, Größen, Chat-Template-Optionen
  rename-shard1 SRC DST --arch A  Shard-1-Header mit anderem Architekturnamen schreiben
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .config import PRESETS, ServerConfig, build_command, get_preset, load_profile, save_profile
from .discovery import PROJECT, discover_all, create_variant, VARIANTS
from .hardware import probe
from .memory import GIB

OVERRIDES = {
    "quant": str, "engine": str, "backend": str, "ctx": ("ctx_size", int), "np": ("n_parallel", int),
    "spec-n": ("spec_draft_n_max", int), "spec-p": ("spec_draft_p_min", float), "reasoning": ("reasoning_effort", str),
    "host": str, "port": int, "ub": ("ubatch", int), "b": ("batch", int), "threads": int, "ctk": str, "ctv": str,
    "cache-ram": ("cache_ram_mib", int), "load-mode": str, "alias": str, "temp": float, "top-p": ("top_p", float),
    "top-k": ("top_k", int), "min-p": ("min_p", float), "spec-extra": ("spec_extra_types", str),
    "mem-guard": ("mem_guard_gib", float),
}


def add_overrides(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--preset", default="unsloth-agent")
    ap.add_argument("--profile", default="", help="gespeichertes Profil statt Preset (state/profiles)")
    for k, spec in OVERRIDES.items():
        typ = spec[1] if isinstance(spec, tuple) else spec
        ap.add_argument("--" + k, type=typ, default=None)
    ap.add_argument("--mtp", dest="mtp", action="store_true", default=None)
    ap.add_argument("--no-mtp", dest="mtp", action="store_false")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--kv-unified", action="store_true")
    ap.add_argument("--reasoning-preserve", action="store_true")
    ap.add_argument("--webui", action="store_true")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE für den Server (z.B. ROCBLAS_USE_HIPBLASLT=1)")
    ap.add_argument("--extra", default="", help="weitere Argumente für llama-server (ein String)")
    ap.add_argument("--budget-gib", type=float, default=None,
                    help="Speicherbudget statt aktuellem MemAvailable (Planung, während ein anderer Server läuft)")
    ap.add_argument("--save-profile", default="", help="Konfiguration unter diesem Namen speichern")


def config_from(a: argparse.Namespace) -> ServerConfig:
    if a.profile:
        cfg = load_profile(a.profile)
    else:
        pr = get_preset(a.preset)
        if not pr:
            sys.exit(f"Unbekanntes Preset: {a.preset} (python3 -m glm53 presets)")
        cfg = pr.apply()
    over: dict = {}
    for k, spec in OVERRIDES.items():
        v = getattr(a, k.replace("-", "_"))
        if v is not None:
            over[spec[0] if isinstance(spec, tuple) else k] = v
    if a.mtp is not None:
        over["mtp_enabled"] = a.mtp
    if a.no_thinking:
        over["thinking"] = False
    if a.vision:
        over["vision"] = True
    if a.kv_unified:
        over["kv_unified"] = True
    if a.reasoning_preserve:
        over["reasoning_preserve"] = True
    if a.webui:
        over["webui"] = True
    if a.env:
        over["env"] = {**cfg.env, **dict(e.split("=", 1) for e in a.env if "=" in e)}
    if a.extra:
        import shlex
        over["extra_args"] = list(cfg.extra_args) + shlex.split(a.extra)
    cfg = cfg.copy(**over)
    if a.save_profile:
        print("Profil gespeichert:", save_profile(a.save_profile, cfg), file=sys.stderr)
    return cfg


def cmd_presets(a) -> int:
    print(f"{'Preset':16} {'Engine':10} {'Quant':16} {'Kontext':>8} {'MTP':>5}  Beschreibung")
    for p in PRESETS:
        c = p.apply()
        print(f"{p.name:16} {c.engine:10} {c.quant:16} {c.ctx_size:>8} {('n' + str(c.spec_draft_n_max)) if c.mtp_enabled else 'aus':>5}  {p.purpose}")
    return 0


def cmd_models(a) -> int:
    inv, hw = discover_all(), probe()
    print(f"Maschine: MemTotal {hw.mem_total / GIB:.1f} GiB, MemAvailable {hw.mem_available / GIB:.1f} GiB, "
          f"GTT {hw.gtt_used / GIB:.1f}/{hw.gtt_total / GIB:.0f} GiB, VRAM-Carve-out {hw.vram_total / GIB:.0f} GiB "
          f"(belegt {hw.vram_used / GIB:.2f}), ROCm {hw.rocm_version or '?'}, {hw.gfx or '?'}")
    for n in hw.notes:
        print("  Hinweis:", n)
    print("\nModelle:")
    print(f"  {'Label':18} {'Schema':10} {'Quelle':8} {'Größe':>10} {'NextN':>7} {'Experten':>9}  Shard 1")
    for m in sorted(inv.models, key=lambda m: (m.label, m.arch, m.source)):
        print(f"  {m.label:18} {m.arch:10} {m.source:8} {m.size_gib:8.2f} G {m.nextn_bytes / GIB:5.2f} G {m.experts_bytes / GIB:7.2f} G  {m.first_shard}")
    if not inv.models:
        print("  (keine – models/fetch.sh)")
    print("\nEngines:")
    for e in inv.engines:
        print(f"  {e.label:16} Schema {e.naming:10} MTP {'ja' if e.has_mtp else 'nein':4}  {e.bindir}")
    if not inv.engines:
        print("  (keine gebaut – engine/build.sh)")
    print("\nVision-Projektoren:")
    for m in inv.mmproj:
        print(f"  {m.label:40} Schema {m.naming:10} {m.size_bytes / GIB:5.2f} GiB  {m.path}")
    return 0


def _build(a):
    cfg = config_from(a)
    inv, hw = discover_all(), probe()
    budget = int(a.budget_gib * GIB) if a.budget_gib else None
    return cfg, inv, hw, build_command(cfg, inv, hw, budget=budget)


def cmd_show(a) -> int:
    cfg, inv, hw, cmd = _build(a)
    r = cmd.resolved
    print("== Konfiguration")
    print(f"   Engine        {r.engine.label if r.engine else cfg.engine + ' (fehlt)'}")
    print(f"   Modell        {cfg.quant}" + (f"  ({r.model.size_gib:.1f} GiB, Schema {r.model.arch}, {r.model.source})" if r.model else "  (fehlt)"))
    print(f"   MTP           {'draft-mtp n' + str(cfg.spec_draft_n_max) + ' p' + str(cfg.spec_draft_p_min) if cfg.mtp_enabled else 'aus'}")
    print(f"   Kontext       {cfg.ctx_size} gesamt, {cfg.ctx_size // max(1, cfg.n_parallel)} je Slot ({cfg.n_parallel} Slots)")
    print(f"   Thinking      {'reasoning_effort ' + cfg.reasoning_effort if cfg.thinking else 'aus'}")
    if cmd.estimate:
        print("== Speicherbilanz" + ("" if a.budget_gib is None else f" (Budget {a.budget_gib:.1f} GiB)"))
        for k, v in cmd.estimate.rows():
            print(f"   {k:34} {v}")
        print(f"   Bewertung     {cmd.estimate.verdict}")
        for n in cmd.estimate.notes:
            print("   Hinweis:", n)
    for w in cmd.warnings:
        print("   WARNUNG:", w)
    for e in cmd.errors:
        print("   FEHLER:", e)
    print("== Kommando")
    print("$ " + cmd.shell())
    return 0 if cmd.ok else 1


def cmd_cmd(a) -> int:
    cfg, inv, hw, cmd = _build(a)
    for e in cmd.errors:
        print("FEHLER:", e, file=sys.stderr)
    print(cmd.shell())
    return 0 if cmd.ok else 1


def cmd_run(a) -> int:
    cfg, inv, hw, cmd = _build(a)
    for w in cmd.warnings:
        print("WARNUNG:", w, file=sys.stderr)
    if not cmd.ok:
        for e in cmd.errors:
            print("FEHLER:", e, file=sys.stderr)
        return 1
    logdir = PROJECT / "state" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = logdir / f"server-{stamp}.log"
    guard = [sys.executable, str(PROJECT / "bench" / "memguard.py"), "--min-avail-gib", str(cfg.mem_guard_gib),
             "--csv", str(logdir / f"server-mem-{stamp}.csv"), "--"] + cmd.argv
    env = dict(os.environ)
    env.update(cmd.env)
    print(f"Server-Log: {log}\nWächter: SIGKILL bei MemAvailable < {cfg.mem_guard_gib} GiB\n$ {cmd.shell()}", file=sys.stderr)
    with log.open("w", buffering=1) as fh:
        fh.write("# " + cmd.shell() + "\n")
        proc = subprocess.Popen(guard, env=env, stdout=fh, stderr=subprocess.STDOUT)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(2)
            return proc.wait()


def cmd_variants(a) -> int:
    inv = discover_all()
    hf = [m for m in inv.models if m.source == "hf" and (not a.quant or m.label == a.quant)]
    if not hf:
        print("keine Modelle im HF-Cache gefunden")
        return 1
    for m in hf:
        for naming in ("glm5-next", "glm5next"):
            exists = inv.model(m.label, naming) is not None and (inv.model(m.label, naming).source == "variant" or m.arch == naming)
            if a.create:
                try:
                    p, how = create_variant(m, naming, force=a.force)
                    print(f"{m.label:18} {naming:10} -> {p.parent.name}: {how}")
                except Exception as e:  # noqa: BLE001
                    print(f"{m.label:18} {naming:10} FEHLER: {e}")
            else:
                print(f"{m.label:18} {naming:10} {'vorhanden' if exists else 'fehlt (--create)'}")
    return 0


def cmd_gguf_info(a) -> int:
    from .gguf import read_gguf, model_stats
    g = read_gguf(a.file, max_array=8)
    arch = g.arch
    print(f"Datei: {a.file}\nArchitektur: {arch}   GGUF v{g.version}   Tensoren in diesem Shard: {len(g.tensors)}")
    for k in sorted(g.metadata):
        if k.startswith(arch + ".") or k.startswith("general.") or k.startswith("split."):
            v = g.metadata[k]
            s = str(v)
            print(f"  {k} = {s[:120]}")
    tpl = str(g.get("tokenizer.chat_template", ""))
    if tpl:
        opts = sorted(set(re.findall(r"\b(reasoning_effort|enable_thinking|clear_thinking|preserve_thinking|reasoning_content)\b", tpl)))
        m = re.search(r"reasoning_effort in \[([^\]]*)\]", tpl)
        print(f"  Chat-Template: {len(tpl)} Zeichen, Optionen: {', '.join(opts)}" + (f"; reasoning_effort erlaubt {m.group(1)} (sonst 'max')" if m else ""))
    if a.stats:
        st = model_stats(a.file)
        print(f"Alle Shards: {st.total_bytes / GIB:.2f} GiB in {st.n_tensors} Tensoren; NextN {st.nextn_bytes / GIB:.2f} GiB; "
              f"geroutete Experten {st.experts_bytes / GIB:.2f} GiB")
        print("  nach Kategorie:", {k: round(v / GIB, 2) for k, v in sorted(st.by_category.items(), key=lambda x: -x[1])})
        print("  nach Typ:", {k: round(v / GIB, 2) for k, v in sorted(st.by_type.items(), key=lambda x: -x[1])})
    return 0


def cmd_rename(a) -> int:
    from .gguf import rewrite_arch
    print(json.dumps(rewrite_arch(a.src, a.dst, a.arch), indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="glm53", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("presets").set_defaults(fn=cmd_presets)
    sub.add_parser("models").set_defaults(fn=cmd_models)
    for name, fn in (("show", cmd_show), ("cmd", cmd_cmd), ("run", cmd_run)):
        p = sub.add_parser(name)
        add_overrides(p)
        p.set_defaults(fn=fn)
    p = sub.add_parser("variants")
    p.add_argument("--create", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--quant", default="")
    p.set_defaults(fn=cmd_variants)
    p = sub.add_parser("gguf-info")
    p.add_argument("file")
    p.add_argument("--stats", action="store_true", help="alle Shards lesen: Größen nach Kategorie und Typ")
    p.set_defaults(fn=cmd_gguf_info)
    p = sub.add_parser("rename-shard1")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--arch", required=True, choices=["glm5next", "glm5-next"])
    p.set_defaults(fn=cmd_rename)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
