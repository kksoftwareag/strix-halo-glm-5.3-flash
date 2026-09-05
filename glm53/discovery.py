"""Findet Modelle (HF-Cache, Varianten-Ordner), Engines (engine/build-*) und Vision-Projektoren."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .gguf import read_gguf, shard_paths, model_stats

PROJECT = Path(__file__).resolve().parents[1]
HF_HUB = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))).expanduser() / "hub"
VARIANTS = PROJECT / "models" / "variants"
CACHE = PROJECT / "state" / "model_cache.json"
GIB = 2**30

# Bekannte Repos: (HF-Repo, Präfix im Label)
KNOWN_REPOS = [
    "unsloth/GLM-5.3-Flash-GGUF",
    "sayyidfareed/GLM-5.3-Flash-Spark-Q2XL-MTP",
    "AesSedai/GLM-5.3-Flash-GGUF",
    "aj9o9/GLM-5.3-Flash-GGUF",
    "vcruz305/GLM-5.3-Flash-GGUF",
]
ARCH_NAMES = ("glm5next", "glm5-next")


@dataclass
class ModelInfo:
    label: str                      # z.B. UD-IQ2_XXS, Spark-Q2XL-MTP, AesSedai-IQ2_S
    repo: str
    first_shard: str
    n_shards: int
    total_bytes: int
    nextn_bytes: int
    experts_bytes: int
    arch: str                       # glm5next | glm5-next
    source: str                     # hf | variant
    mtime: float = 0.0
    complete: bool = True

    @property
    def path(self) -> Path:
        return Path(self.first_shard)

    @property
    def size_gib(self) -> float:
        return self.total_bytes / GIB

    @property
    def has_nextn(self) -> bool:
        return self.nextn_bytes > 0

    @property
    def naming(self) -> str:
        return self.arch


@dataclass
class Engine:
    name: str          # tk | tk-mtp | tk-merged | unsloth
    backend: str       # hip | vulkan
    bindir: str

    @property
    def server(self) -> Path:
        return Path(self.bindir) / "llama-server"

    @property
    def bench(self) -> Path:
        return Path(self.bindir) / "llama-bench"

    @property
    def naming(self) -> str:
        return "glm5next" if self.name.startswith("unsloth") else "glm5-next"

    @property
    def has_mtp(self) -> bool:
        return self.name in ("tk-mtp", "tk-merged", "unsloth")

    @property
    def label(self) -> str:
        return f"{self.name}/{self.backend}"

    def version(self) -> str:
        import subprocess
        try:
            out = subprocess.run([str(self.server), "--version"], capture_output=True, text=True, timeout=20)
            return (out.stdout + out.stderr).strip().splitlines()[0][:80] if (out.stdout + out.stderr).strip() else "?"
        except (OSError, subprocess.SubprocessError, IndexError):
            return "?"


@dataclass
class Mmproj:
    label: str
    path: str
    naming: str        # glm5next (unsloth-Schlüssel, alt) | glm5-next (aktuelle Schlüssel, avar6/AesSedai)
    size_bytes: int


@dataclass
class Inventory:
    models: list[ModelInfo] = field(default_factory=list)
    engines: list[Engine] = field(default_factory=list)
    mmproj: list[Mmproj] = field(default_factory=list)

    def model(self, label: str, naming: str | None = None) -> ModelInfo | None:
        cands = [m for m in self.models if m.label == label and (naming is None or m.arch == naming)]
        # Varianten (models/variants) vor HF-Cache bevorzugen
        cands.sort(key=lambda m: 0 if m.source == "variant" else 1)
        return cands[0] if cands else None

    def engine(self, name: str, backend: str = "hip") -> Engine | None:
        return next((e for e in self.engines if e.name == name and e.backend == backend), None)

    def mmproj_for(self, naming: str) -> Mmproj | None:
        c = [m for m in self.mmproj if m.naming == naming]
        c.sort(key=lambda m: m.size_bytes)  # kleinste zuerst (Q8_0 vor F16)
        return c[0] if c else None


# ---------------------------------------------------------------- Modelle

def _label_for(repo: str, first: Path) -> str:
    d = first.parent.name
    if repo.startswith("unsloth/"):
        return d if d.startswith("UD-") or d in ("Q8_0", "BF16") else d
    if "Spark" in repo:
        return "Spark-Q2XL-MTP"
    if repo.startswith("AesSedai/"):
        return f"AesSedai-{d}"
    if repo.startswith("aj9o9/"):
        return d
    if repo.startswith("vcruz305/"):
        m = re.search(r"GLM-5\.3-Flash-(.+?)\.gguf$", first.name)
        return f"vcruz-{m.group(1)}" if m else first.stem
    return d


def _first_shards(directory: Path) -> list[Path]:
    out = []
    for p in sorted(directory.glob("*.gguf")):
        if p.name.lower().startswith("mmproj"):
            continue
        m = re.match(r".*-(\d{5})-of-(\d{5})\.gguf$", p.name)
        if m and m.group(1) != "00001":
            continue
        out.append(p)
    return out


def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def _save_cache(c: dict) -> None:
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(c, indent=1))
    except OSError:
        pass


def _analyze(first: Path, repo: str, source: str, cache: dict) -> ModelInfo | None:
    shards = shard_paths(first)
    if not all(p.exists() for p in shards):
        return None
    mtime = max(p.stat().st_mtime for p in shards)
    key = str(first)
    c = cache.get(key)
    if c and abs(c.get("mtime", 0) - mtime) < 1:
        c["source"], c["repo"] = source, repo
        return ModelInfo(**{k: v for k, v in c.items() if k in ModelInfo.__dataclass_fields__})
    try:
        head = read_gguf(first, max_array=4)
        st = model_stats(first)
    except (OSError, ValueError) as e:
        return None
    mi = ModelInfo(label=_label_for(repo, first), repo=repo, first_shard=str(first), n_shards=len(shards),
                   total_bytes=st.total_bytes, nextn_bytes=st.nextn_bytes, experts_bytes=st.experts_bytes,
                   arch=head.arch, source=source, mtime=mtime)
    cache[key] = asdict(mi)
    return mi


def scan_models() -> list[ModelInfo]:
    cache = _load_cache()
    out: list[ModelInfo] = []
    for repo in KNOWN_REPOS:
        d = HF_HUB / ("models--" + repo.replace("/", "--")) / "snapshots"
        if not d.is_dir():
            continue
        for snap in d.iterdir():
            dirs = [snap] + [p for p in snap.iterdir() if p.is_dir() and p.name != "Shard_Rewrite"]
            for sub in dirs:
                for first in _first_shards(sub):
                    mi = _analyze(first, repo, "hf", cache)
                    if mi:
                        out.append(mi)
    if VARIANTS.is_dir():
        for v in sorted(VARIANTS.iterdir()):
            for first in _first_shards(v):
                mi = _analyze(first, "variant", "variant", cache)
                if mi:
                    m = re.match(r"(.+)\.(glm5next|glm5-next)$", v.name)
                    mi.label = m.group(1) if m else v.name
                    out.append(mi)
    _save_cache(cache)
    return out


def shard_rewrite_headers() -> dict[str, Path]:
    """unsloth Shard_Rewrite: Shard-1-Header im glm5-next-Schema je Quant (Datei ...gguf_file)."""
    out: dict[str, Path] = {}
    d = HF_HUB / "models--unsloth--GLM-5.3-Flash-GGUF" / "snapshots"
    if d.is_dir():
        for p in d.glob("*/Shard_Rewrite/*.gguf_file"):
            m = re.match(r"GLM-5\.3-Flash-(UD-[A-Z0-9_]+)-00001-of-\d{5}\.gguf_file$", p.name)
            if m:
                out[m.group(1)] = p
    return out


# ---------------------------------------------------------------- Engines / mmproj

def scan_engines() -> list[Engine]:
    out = []
    for d in sorted((PROJECT / "engine").glob("build-*")):
        m = re.match(r"build-(.+)-(hip|vulkan)$", d.name)
        if m and (d / "bin" / "llama-server").is_file():
            out.append(Engine(m.group(1), m.group(2), str(d / "bin")))
    for spec in os.environ.get("GLM53_ENGINES", "").split(":"):
        if "=" in spec:
            name, path = spec.split("=", 1)
            out.append(Engine(name, "hip", str(Path(path).expanduser().parent)))
    return out


def scan_mmproj() -> list[Mmproj]:
    out = []
    d = HF_HUB / "models--unsloth--GLM-5.3-Flash-GGUF" / "snapshots"
    for p in d.glob("*/mmproj-*.gguf") if d.is_dir() else []:
        out.append(Mmproj(f"unsloth {p.name}", str(p), "glm5next", p.stat().st_size))
    d = HF_HUB / "models--AesSedai--GLM-5.3-Flash-GGUF" / "snapshots"
    for p in d.glob("*/mmproj-*.gguf") if d.is_dir() else []:
        out.append(Mmproj(f"AesSedai {p.name}", str(p), "glm5-next", p.stat().st_size))
    d = HF_HUB / "models--avar6--GLM-5.3-Flash-BF16-gguf" / "snapshots"
    for p in d.glob("*/mmproj*.gguf") if d.is_dir() else []:
        out.append(Mmproj(f"avar6 {p.name}", str(p), "glm5-next", p.stat().st_size))
    return out


def discover_all() -> Inventory:
    return Inventory(models=scan_models(), engines=scan_engines(), mmproj=scan_mmproj())


# ---------------------------------------------------------------- Varianten anlegen

def create_variant(model: ModelInfo, naming: str, force: bool = False) -> tuple[Path, str]:
    """Legt models/variants/<label>.<naming>/ an: Shard 1 im gewünschten Namensschema, Rest als Symlink.

    Reihenfolge: passendes Shard_Rewrite-Header-File von unsloth (falls vorhanden), sonst eigener
    Header-Umschreiber (glm53.gguf.rewrite_arch). Gibt (Pfad zu Shard 1, Beschreibung) zurück.
    """
    from .gguf import rewrite_arch
    dest = VARIANTS / f"{model.label}.{naming}"
    shards = shard_paths(model.path)
    first_name = shards[0].name
    dest.mkdir(parents=True, exist_ok=True)
    for p in shards[1:]:
        link = dest / p.name
        if link.is_symlink() or link.exists():
            if force:
                link.unlink()
            else:
                continue
        link.symlink_to(p.resolve())
    target = dest / first_name
    if target.exists() and not force:
        return target, "vorhanden"
    if target.is_symlink() or target.exists():
        target.unlink()
    if model.arch == naming:
        target.symlink_to(shards[0].resolve())
        return target, "Original (Schema passt)"
    hdr = shard_rewrite_headers().get(model.label) if naming == "glm5-next" else None
    if hdr is not None:
        g = read_gguf(hdr, max_array=2)
        if g.arch == naming:
            target.symlink_to(hdr.resolve())
            return target, f"unsloth Shard_Rewrite ({hdr.name})"
    info = rewrite_arch(shards[0], target, naming)
    return target, f"Header umgeschrieben ({info['old_arch']} -> {info['new_arch']}, {info['renamed_keys']} Schlüssel)"
