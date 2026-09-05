"""Server-Konfiguration, Presets und Kommandozeilen-Erzeugung für llama-server."""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

from .discovery import Inventory, ModelInfo, Engine, Mmproj, PROJECT
from .hardware import Hardware
from .memory import estimate, Estimate, GIB

DEFAULT_HOST = os.environ.get("GLM53_HOST", "10.50.4.9")
REASONING_LEVELS = ("low", "high", "max")   # das Chat-Template kennt genau diese; alles andere wird zu "max"


@dataclass
class ServerConfig:
    engine: str = "tk-mtp"
    backend: str = "hip"
    quant: str = "UD-IQ2_XXS"
    ctx_size: int = 131072
    n_parallel: int = 1
    kv_unified: bool = False           # bei -np > 1 auf #27773 Pflicht
    mtp_enabled: bool = True
    spec_draft_n_max: int = 2
    spec_draft_n_min: int = 0
    spec_draft_p_min: float = 0.75
    spec_extra_types: str = ""         # z.B. "ngram-mod"
    flash_attn: str = "on"
    ctk: str = "q8_0"
    ctv: str = "q8_0"
    batch: int = 4096
    ubatch: int = 1024
    threads: int = 8
    load_mode: str = "none"            # mmap+mlock stürzte bei langen Generierungen ab (HF-Diskussion #4)
    cache_ram_mib: int = 512            # Obergrenze Prompt-Cache; 8 GiB Default kostete auf 128 GB die Reserve (Spark-README)
    reasoning_effort: str = "high"     # low | high | max (Template-Default: max)
    thinking: bool = True
    reasoning_preserve: bool = False
    temp: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    vision: bool = False
    host: str = DEFAULT_HOST
    port: int = 8080
    alias: str = "glm-5.3-flash"
    api_key: str = ""
    metrics: bool = True
    webui: bool = False
    verbosity: int = 3
    env: dict[str, str] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)
    mem_guard_gib: float = 5.0

    def copy(self, **over) -> "ServerConfig":
        return replace(self, **over)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)


@dataclass
class Preset:
    name: str
    title: str
    purpose: str
    over: dict

    def apply(self) -> ServerConfig:
        return ServerConfig().copy(**self.over)


PRESETS: list[Preset] = [
    Preset("tkmtp-agent", "timkhronos #27773+#27917, MTP, 128k",
           "Standard für Agenten (ein Nutzer): PR #27917 (MTP) auf #27773, UD-IQ2_XXS, Draft-Tiefe 2",
           {"engine": "tk-mtp", "mtp_enabled": True, "spec_draft_n_max": 2}),
    Preset("tkmtp-agent-n3", "wie tkmtp-agent, Draft-Tiefe 3", "Vergleich der Draft-Tiefe",
           {"engine": "tk-mtp", "mtp_enabled": True, "spec_draft_n_max": 3}),
    Preset("merged-agent", "#27917 auf #27773-HEAD (Patch 0001), MTP, 128k",
           "Experimentell: neueste Basis (Sparse-FA-Prefill, Multi-Stream) plus MTP",
           {"engine": "tk-merged", "mtp_enabled": True, "spec_draft_n_max": 2}),
    Preset("unsloth-agent", "unsloth #27754, MTP, 128k",
           "Zweite Engine: unsloth-Branch (glm5next-Schema) mit MTP – A/B-Vergleich",
           {"engine": "unsloth", "mtp_enabled": True, "spec_draft_n_max": 2}),
    Preset("tk-plain", "timkhronos #27773 ohne MTP, 128k", "Basislinie ohne spekulatives Decoding",
           {"engine": "tk", "mtp_enabled": False}),
    Preset("chat-vision", "MTP + Vision-Projektor, 64k, reasoning high",
           "Chat mit Bildern (mmproj F16/Q8_0 im glm5-next-Schema)",
           {"engine": "tk-mtp", "mtp_enabled": True, "vision": True, "ctx_size": 65536, "webui": True, "ubatch": 512, "batch": 2048, "cache_ram_mib": 256}),
    Preset("chat-fast", "MTP, 32k, reasoning low, ngram-mod", "Schneller Chat, kurze Antworten",
           {"engine": "tk-mtp", "mtp_enabled": True, "ctx_size": 32768, "reasoning_effort": "low",
            "spec_extra_types": "ngram-mod", "webui": True}),
    Preset("multiuser-4", "4 Slots, kv-unified, ohne MTP, 4×64k",
           "Mehrere Nutzer gleichzeitig (MTP lohnt nur bei einem Nutzer)",
           {"engine": "tk", "mtp_enabled": False, "n_parallel": 4, "kv_unified": True, "ctx_size": 262144}),
    Preset("q2kxl-agent", "UD-Q2_K_XL + MTP (nur mit kleinem VRAM-Carve-out)",
           "Bester unsloth-Quant, der mit 1 GiB Carve-out passt; mit 16 GiB Carve-out zu groß",
           {"engine": "tk-mtp", "mtp_enabled": True, "quant": "UD-Q2_K_XL"}),
    Preset("spark-agent", "Spark-Q2XL-MTP (2,80 bpw) + MTP (nur kleiner Carve-out, unsloth-Engine)",
           "Quality-first-Quant von sayyidfareed (Attention Q8_0, Experten IQ2_XS/IQ3_XXS)",
           {"engine": "unsloth", "mtp_enabled": True, "quant": "Spark-Q2XL-MTP"}),
]


def get_preset(name: str) -> Preset | None:
    return next((p for p in PRESETS if p.name == name), None)


@dataclass
class Resolved:
    engine: Engine | None = None
    model: ModelInfo | None = None
    mmproj: Mmproj | None = None


@dataclass
class Command:
    argv: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    resolved: Resolved = field(default_factory=Resolved)
    estimate: Estimate | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def shell(self) -> str:
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
        return (env + " " if env else "") + " ".join(shlex.quote(a) for a in self.argv)


def build_command(cfg: ServerConfig, inv: Inventory, hw: Hardware, *, budget: int | None = None) -> Command:
    cmd = Command()
    r = cmd.resolved
    r.engine = inv.engine(cfg.engine, cfg.backend)
    if not r.engine:
        cmd.errors.append(f"Engine {cfg.engine}/{cfg.backend} nicht gebaut (engine/build.sh {cfg.engine} {cfg.backend})")
        naming = "glm5next" if cfg.engine.startswith("unsloth") else "glm5-next"
    else:
        naming = r.engine.naming
        if cfg.mtp_enabled and not r.engine.has_mtp:
            cmd.errors.append(f"Engine {cfg.engine} hat keinen MTP-Pfad – Preset ohne MTP oder Engine tk-mtp/tk-merged/unsloth")
    r.model = inv.model(cfg.quant, naming)
    if not r.model:
        any_ = inv.model(cfg.quant)
        if any_:
            cmd.errors.append(f"Quant {cfg.quant} liegt nur im Schema {any_.arch} vor, Engine {cfg.engine} braucht {naming}: "
                              f"python3 -m glm53 variants --create")
        else:
            cmd.errors.append(f"Quant {cfg.quant} nicht gefunden (models/fetch.sh)")
    if r.model and cfg.mtp_enabled and not r.model.has_nextn:
        cmd.errors.append(f"{cfg.quant} enthält keine NextN-Tensoren – MTP nicht möglich")
    if cfg.vision:
        r.mmproj = inv.mmproj_for(naming)
        if not r.mmproj:
            cmd.errors.append(f"kein Vision-Projektor im Schema {naming} (models/fetch.sh mmproj)")
    if cfg.reasoning_effort not in REASONING_LEVELS:
        cmd.errors.append(f"reasoning_effort {cfg.reasoning_effort!r}: das Template kennt nur {', '.join(REASONING_LEVELS)}")
    if cfg.n_parallel > 1 and not cfg.kv_unified and not cfg.engine.startswith("unsloth"):
        cmd.warnings.append("mehrere Slots auf #27773 brauchen --kv-unified – wird gesetzt")
        cfg = cfg.copy(kv_unified=True)
    if cfg.n_parallel > 1 and cfg.mtp_enabled:
        cmd.warnings.append("MTP mit mehreren Slots: im Qwen-Projekt sank der Gesamtdurchsatz; hier ungemessen")
    if cfg.ctx_size > 262144 * max(1, cfg.n_parallel):
        cmd.warnings.append("mehr als 256k je Slot: Prompt-Processing wird sehr langsam, Nutzen ungemessen")

    b = int(budget if budget is not None else hw.mem_available)
    if r.model:
        cmd.estimate = estimate(total_bytes=r.model.total_bytes, nextn_bytes=r.model.nextn_bytes, mtp=cfg.mtp_enabled,
                                ctx=cfg.ctx_size, n_parallel=cfg.n_parallel, ctk=cfg.ctk, ctv=cfg.ctv, ubatch=cfg.ubatch,
                                cache_ram_mib=cfg.cache_ram_mib, mmproj_bytes=(r.mmproj.size_bytes if r.mmproj else 0), budget=b)
        if cmd.estimate.verdict == "zu groß":
            cmd.errors.append(f"passt nicht: Bedarf {cmd.estimate.total / GIB:.1f} GiB + {cmd.estimate.reserve / GIB:.0f} GiB Reserve "
                              f"> Budget {b / GIB:.1f} GiB")
        elif cmd.estimate.verdict == "knapp":
            cmd.warnings.append(f"knapp: nur {cmd.estimate.headroom / GIB:.1f} GiB Spielraum")

    argv: list[str] = [str(r.engine.server) if r.engine else "llama-server"]
    if r.model:
        argv += ["-m", str(r.model.path)]
    argv += ["-ngl", "99", "-c", str(cfg.ctx_size), "-np", str(cfg.n_parallel)]
    if cfg.kv_unified:
        argv += ["--kv-unified"]
    argv += ["-fa", cfg.flash_attn, "-ctk", cfg.ctk, "-ctv", cfg.ctv, "-b", str(cfg.batch), "-ub", str(cfg.ubatch),
             "-t", str(cfg.threads), "--load-mode", cfg.load_mode, "--cache-ram", str(cfg.cache_ram_mib)]
    if cfg.mtp_enabled:
        types = "draft-mtp" + ("," + cfg.spec_extra_types if cfg.spec_extra_types else "")
        argv += ["--spec-type", types, "--spec-draft-n-max", str(cfg.spec_draft_n_max),
                 "--spec-draft-p-min", str(cfg.spec_draft_p_min)]
        if cfg.spec_draft_n_min:
            argv += ["--spec-draft-n-min", str(cfg.spec_draft_n_min)]
    if r.mmproj:
        argv += ["--mmproj", r.mmproj.path]
    kwargs = {"reasoning_effort": cfg.reasoning_effort}
    if not cfg.thinking:
        kwargs["enable_thinking"] = False
    argv += ["--jinja", "--chat-template-kwargs", json.dumps(kwargs)]
    if cfg.reasoning_preserve:
        argv += ["--reasoning-preserve"]
    argv += ["--temp", str(cfg.temp), "--top-p", str(cfg.top_p), "--top-k", str(cfg.top_k), "--min-p", str(cfg.min_p)]
    argv += ["--host", cfg.host, "--port", str(cfg.port), "-a", cfg.alias]
    if cfg.api_key:
        argv += ["--api-key", cfg.api_key]
    if cfg.metrics:
        argv += ["--metrics"]
    if not cfg.webui:
        argv += ["--no-webui"]
    argv += ["-lv", str(cfg.verbosity)]
    argv += list(cfg.extra_args)
    cmd.argv = argv
    cmd.env = dict(cfg.env)
    return cmd


def load_profile(name: str) -> ServerConfig:
    p = PROJECT / "state" / "profiles" / f"{name}.json"
    data = json.loads(p.read_text())
    return ServerConfig(**{k: v for k, v in data.items() if k in ServerConfig.__dataclass_fields__})


def save_profile(name: str, cfg: ServerConfig) -> Path:
    d = PROJECT / "state" / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(cfg.to_json())
    return p
