"""Minimaler GGUF-Header-Reader und -Umschreiber (ohne numpy).

Liest Metadaten und Tensor-Infos, ohne Gewichte anzufassen, und kann den Header eines Shards
mit anderem Architekturnamen neu schreiben (glm5next <-> glm5-next). Die Gewichte-Shards bleiben
unverändert; nur Shard 1 (Metadaten, Vokabular, keine Tensoren) unterscheidet sich zwischen den
Namensschemata der llama.cpp-Branches.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16, _T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL, _T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = range(13)
_SCALAR_FMT = {
    _T_UINT8: "<B", _T_INT8: "<b", _T_UINT16: "<H", _T_INT16: "<h", _T_UINT32: "<I", _T_INT32: "<i",
    _T_FLOAT32: "<f", _T_BOOL: "<?", _T_UINT64: "<Q", _T_INT64: "<q", _T_FLOAT64: "<d",
}
# ggml-Typen: id -> (Name, Blockgröße, Bytes je Block)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18),
}
KV_TYPE_BYTES_PER_ELEMENT: dict[str, float] = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32, "iq4_nl": 18 / 32, "q5_0": 22 / 32, "q5_1": 24 / 32,
}
BIG_KEYS = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type", "tokenizer.ggml.scores")


@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    type_id: int
    offset: int

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.type_id, (f"T{self.type_id}", 1, 1))[0]

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def n_bytes(self) -> int:
        _, block, tsize = GGML_TYPES.get(self.type_id, (None, 1, 1))
        if not self.shape:
            return 0
        rows = 1
        for d in self.shape[1:]:
            rows *= d
        return (self.shape[0] // block) * tsize * rows


@dataclass
class GGUFFile:
    path: Path
    version: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    @property
    def arch(self) -> str:
        return str(self.metadata.get("general.architecture", ""))


class _Reader:
    def __init__(self, f: BinaryIO):
        self.f = f

    def scalar(self, t: int) -> Any:
        fmt = _SCALAR_FMT[t]
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        (n,) = struct.unpack("<Q", self.f.read(8))
        return self.f.read(n).decode("utf-8", errors="replace")

    def value(self, t: int, max_array: int) -> Any:
        if t == _T_STRING:
            return self.string()
        if t == _T_ARRAY:
            (sub,) = struct.unpack("<I", self.f.read(4))
            (n,) = struct.unpack("<Q", self.f.read(8))
            out: list[Any] = []
            for i in range(n):
                v = self.value(sub, max_array)
                if i < max_array:
                    out.append(v)
            if n > max_array:
                out.append(f"...(+{n - max_array})")
            return out
        return self.scalar(t)


def read_gguf(path: str | Path, *, max_array: int = 256, skip_keys: tuple[str, ...] = BIG_KEYS) -> GGUFFile:
    """Liest Header, Metadaten und Tensor-Infos. Große Token-Arrays werden nur gezählt (`<key>#len`)."""
    p = Path(path)
    with p.open("rb") as f:
        if f.read(4) != GGUF_MAGIC:
            raise ValueError(f"{p}: kein GGUF (Magic fehlt)")
        (version,) = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError(f"{p}: GGUF-Version {version} nicht unterstützt")
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        r = _Reader(f)
        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            key = r.string()
            (t,) = struct.unpack("<I", f.read(4))
            if key in skip_keys:
                if t == _T_ARRAY:
                    (sub,) = struct.unpack("<I", f.read(4))
                    (n,) = struct.unpack("<Q", f.read(8))
                    for _i in range(n):
                        r.value(sub, 0)
                    meta[key + "#len"] = n
                else:
                    r.value(t, 0)
                continue
            meta[key] = r.value(t, max_array)
        tensors: list[TensorInfo] = []
        for _ in range(n_tensors):
            name = r.string()
            (n_dims,) = struct.unpack("<I", f.read(4))
            dims = list(struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims)))
            type_id, offset = struct.unpack("<IQ", f.read(12))
            tensors.append(TensorInfo(name, dims, type_id, offset))
    return GGUFFile(p, version, meta, tensors)


def shard_paths(first_shard: str | Path) -> list[Path]:
    """Alle Shards eines gesplitteten GGUF (…-00001-of-00004.gguf) in Reihenfolge."""
    p = Path(first_shard)
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", p.name)
    if not m:
        return [p]
    prefix, total = m.group(1), int(m.group(3))
    return [p.with_name(f"{prefix}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)]


# ---------------------------------------------------------------- Header umschreiben

def _skip_value(buf: bytes, pos: int, t: int) -> int:
    """Endposition eines Werts vom Typ t ab pos."""
    if t == _T_STRING:
        (n,) = struct.unpack_from("<Q", buf, pos)
        return pos + 8 + n
    if t == _T_ARRAY:
        (sub,) = struct.unpack_from("<I", buf, pos)
        (n,) = struct.unpack_from("<Q", buf, pos + 4)
        pos += 12
        if sub in _SCALAR_FMT:
            return pos + n * struct.calcsize(_SCALAR_FMT[sub])
        for _ in range(n):
            pos = _skip_value(buf, pos, sub)
        return pos
    return pos + struct.calcsize(_SCALAR_FMT[t])


def _enc_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


# Schlüssel, die das glm5-next-Schema (#27773) zusätzlich erwartet; unsloth setzt sie in Shard_Rewrite.
ARCH_EXTRA_BOOL_KEYS = {"glm5-next": {"attention.indexer.index_share_mtp": True}}


def rewrite_arch(src: str | Path, dst: str | Path, new_arch: str, *, extra_renames: dict[str, str] | None = None,
                 add_bool: dict[str, bool] | None = None) -> dict[str, Any]:
    """Schreibt die Datei src mit anderem Architekturnamen nach dst.

    Umbenannt werden `general.architecture` und alle Schlüssel mit dem Präfix `<alt>.` -> `<neu>.`;
    alles andere (auch Tensor-Infos und Daten) wird byteweise übernommen. Fehlende Bool-Schlüssel aus
    ARCH_EXTRA_BOOL_KEYS[new_arch] (bzw. add_bool, relativ zum Architektur-Präfix) werden angehängt.
    Ergebnis: kleines Protokoll.
    """
    src, dst = Path(src), Path(dst)
    buf = src.read_bytes()
    if buf[:4] != GGUF_MAGIC:
        raise ValueError(f"{src}: kein GGUF")
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
    pos = 24
    entries: list[tuple[str, int, int, int]] = []  # key, type, value_start, value_end
    old_arch = ""
    for _ in range(n_kv):
        (klen,) = struct.unpack_from("<Q", buf, pos)
        key = buf[pos + 8: pos + 8 + klen].decode("utf-8")
        pos += 8 + klen
        (t,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        vstart = pos
        pos = _skip_value(buf, pos, t)
        entries.append((key, t, vstart, pos))
        if key == "general.architecture":
            (n,) = struct.unpack_from("<Q", buf, vstart)
            old_arch = buf[vstart + 8: vstart + 8 + n].decode("utf-8")
    kv_end = pos
    # Tensor-Infos überspringen, um das Ende des Headers zu finden
    for _ in range(n_tensors):
        (nlen,) = struct.unpack_from("<Q", buf, pos)
        pos += 8 + nlen
        (nd,) = struct.unpack_from("<I", buf, pos)
        pos += 4 + 8 * nd + 12
    header_end = pos
    alignment = 32
    for key, t, a, b in entries:
        if key == "general.alignment" and t == _T_UINT32:
            (alignment,) = struct.unpack_from("<I", buf, a)
    data_start = (header_end + alignment - 1) // alignment * alignment
    renamed = 0
    want = dict(ARCH_EXTRA_BOOL_KEYS.get(new_arch, {}))
    if add_bool:
        want.update(add_bool)
    present = {(new_arch + k[len(old_arch):]) if old_arch and k.startswith(old_arch + ".") else k for k, *_ in entries}
    added = {f"{new_arch}.{k}": v for k, v in want.items() if f"{new_arch}.{k}" not in present}
    out = bytearray(GGUF_MAGIC)
    out += struct.pack("<IQQ", version, n_tensors, n_kv + len(added))
    for key, t, a, b in entries:
        nk = key
        if old_arch and key.startswith(old_arch + "."):
            nk = new_arch + key[len(old_arch):]
        if extra_renames and key in extra_renames:
            nk = extra_renames[key]
        if nk != key:
            renamed += 1
        out += _enc_string(nk) + struct.pack("<I", t)
        if key == "general.architecture":
            out += _enc_string(new_arch)
        else:
            out += buf[a:b]
    for k, v in added.items():
        out += _enc_string(k) + struct.pack("<I", _T_BOOL) + struct.pack("<?", v)
    out += buf[kv_end:header_end]
    new_data_start = (len(out) + alignment - 1) // alignment * alignment
    out += b"\0" * (new_data_start - len(out))
    out += buf[data_start:]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(bytes(out))
    return {"old_arch": old_arch, "new_arch": new_arch, "renamed_keys": renamed, "added_keys": sorted(added), "n_tensors": n_tensors, "bytes": len(out)}


# ---------------------------------------------------------------- Modellanalyse

@dataclass
class ModelStats:
    total_bytes: int = 0
    nextn_bytes: int = 0
    experts_bytes: int = 0
    n_tensors: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    names: set[str] = field(default_factory=set)

    @property
    def resident_without_nextn(self) -> int:
        return self.total_bytes - self.nextn_bytes


def categorize(name: str, n_layer_nextn_first: int | None) -> str:
    if n_layer_nextn_first is not None and name.startswith(f"blk.{n_layer_nextn_first}."):
        return "nextn"
    if "exps" in name:
        return "routed_experts"
    if "shexp" in name:
        return "shared_expert"
    if "ssm" in name:
        return "kda_linear_attn"
    if "attn" in name or "indexer" in name:
        return "mla_attn_indexer"
    if "token_embd" in name or name.startswith("output"):
        return "embd_output"
    return "other"


def model_stats(first_shard: str | Path) -> ModelStats:
    """Größen aller Shards nach Kategorie, NextN-Anteil, Tensortypen (nur Header werden gelesen)."""
    st = ModelStats()
    head = read_gguf(first_shard, max_array=4)
    arch = head.arch
    n_layer = int(head.get(f"{arch}.block_count", 0) or 0)
    n_nextn = int(head.get(f"{arch}.nextn_predict_layers", 0) or 0)
    nextn_first = n_layer - n_nextn if n_nextn else None
    for p in shard_paths(first_shard):
        g = head if p == Path(first_shard) else read_gguf(p, max_array=4)
        for t in g.tensors:
            b = t.n_bytes
            st.total_bytes += b
            st.n_tensors += 1
            st.by_type[t.type_name] = st.by_type.get(t.type_name, 0) + b
            c = categorize(t.name, nextn_first)
            st.by_category[c] = st.by_category.get(c, 0) + b
            if c == "nextn":
                st.nextn_bytes += b
            elif c == "routed_experts":
                st.experts_bytes += b
            st.names.add(re.sub(r"blk\.\d+\.", "blk.N.", t.name))
    return st
