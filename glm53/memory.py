"""Speicherschätzung für GLM-5.3-Flash auf Unified Memory.

Konstanten sind an der Messung vom 29.08.2026 (UD-IQ2_XXS, 131k Kontext, HIP) und an den
Buffer-Zeilen der Server-Logs kalibriert – bis eigene Messreihen vorliegen, sind sie Schätzwerte
und werden im Programm als solche ausgewiesen (docs/RESEARCH.md, Abschnitt Speicher).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .gguf import KV_TYPE_BYTES_PER_ELEMENT

GIB = 2**30
MIB = 2**20

# Modellfakten (aus dem GGUF-Header von UD-IQ2_XXS gelesen)
N_LAYER = 45                # Trunk-Layer (46 = 45 + 1 NextN)
N_MLA = 11                  # Layer mit KV-Cache (DSA/MLA), Rest KDA (rekurrent)
KV_LORA_RANK = 512          # MLA: komprimierter KV je Token und Layer (rope dims = 0)
IDX_KEY_LEN = 128           # Indexer-Key je Token und Layer (f16, plus Gate)
KDA_STATE_BYTES = 34 * (32 * 128 * 128 * 4 + 4 * 4096 * 3 * 2)   # 34 KDA-Layer: S-Matrix f32 + Conv-Zustände (geschätzt)
OS_RESERVE = 4 * GIB        # Betriebssystem, Desktop, Page-Cache-Minimum (Wächter greift bei 5 GiB MemAvailable)
HOST_BUFFERS = 1.2 * GIB    # ROCm_Host-Puffer, Tokenizer, Vokabular (154k Einträge), Server


def kv_bytes_per_token(ctk: str, ctv: str) -> tuple[int, int]:
    """(KV-Cache-Bytes, Indexer-Cache-Bytes) je Token über alle 11 MLA-Layer."""
    k = KV_TYPE_BYTES_PER_ELEMENT.get(ctk, 2.0)
    # MLA-Cache: nur K-Anteil (kv_lora_rank) je Layer; V wird daraus rekonstruiert
    kv = int(N_MLA * KV_LORA_RANK * k)
    # Indexer-Key-Cache bleibt f16 (hält auch die Compressor-Gates): 128 Dims + 4 Byte Gate/Score
    idx = int(N_MLA * (IDX_KEY_LEN * 2 + 4))
    return kv, idx


def compute_bytes(ubatch: int) -> int:
    """Compute-Puffer (GPU) in Abhängigkeit vom ubatch – Schätzung, zu kalibrieren."""
    ub = max(64, ubatch)
    return int(1.0 * GIB + (ub - 512) / (2048 - 512) * 1.5 * GIB) if ub > 512 else int(1.0 * GIB * ub / 512)


@dataclass
class Estimate:
    weights: int = 0
    nextn: int = 0
    kv: int = 0
    idx: int = 0
    rs: int = 0
    compute: int = 0
    host: int = 0
    mtp: int = 0
    mmproj: int = 0
    cache_ram: int = 0
    reserve: int = OS_RESERVE
    budget: int = 0
    calibrated: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.weights + self.nextn + self.kv + self.idx + self.rs + self.compute + self.host + self.mtp + self.mmproj + self.cache_ram

    @property
    def headroom(self) -> int:
        return self.budget - self.reserve - self.total

    @property
    def verdict(self) -> str:
        h = self.headroom / GIB
        if h < 0:
            return "zu groß"
        if h < 2:
            return "knapp"
        return "passt"

    def rows(self) -> list[tuple[str, str]]:
        g = lambda b: f"{b / GIB:6.1f} GiB"
        rows = [("Gewichte (ohne NextN)", g(self.weights))]
        if self.nextn:
            rows.append(("NextN-Layer (MTP)", g(self.nextn)))
        rows += [("KV-Cache (11 MLA-Layer)", g(self.kv)), ("Indexer-Cache", g(self.idx)),
                 ("KDA-Zustand je Slot", g(self.rs)), ("Compute-Puffer", g(self.compute)), ("Host-Puffer", g(self.host))]
        if self.mtp:
            rows.append(("MTP-Draft-Kontext", g(self.mtp)))
        if self.mmproj:
            rows.append(("Vision-Projektor", g(self.mmproj)))
        rows += [("Prompt-Cache (Obergrenze)", g(self.cache_ram)), ("Summe", g(self.total)),
                 ("Budget (MemAvailable)", g(self.budget)), ("Reserve Betriebssystem", g(self.reserve)),
                 ("Spielraum", g(self.headroom))]
        return rows


def estimate(*, total_bytes: int, nextn_bytes: int, mtp: bool, ctx: int, n_parallel: int, ctk: str, ctv: str,
             ubatch: int, cache_ram_mib: int, mmproj_bytes: int, budget: int) -> Estimate:
    e = Estimate()
    e.weights = total_bytes - nextn_bytes
    e.nextn = nextn_bytes if mtp else 0
    kv, idx = kv_bytes_per_token(ctk, ctv)
    e.kv, e.idx = kv * ctx, idx * ctx
    e.rs = KDA_STATE_BYTES * max(1, n_parallel)
    e.compute = compute_bytes(ubatch)
    e.host = int(HOST_BUFFERS)
    e.mtp = int(0.8 * GIB) if mtp else 0          # Draft-Kontext (1 NextN-Layer, eigener KV/Compute)
    e.mmproj = mmproj_bytes + (int(1.0 * GIB) if mmproj_bytes else 0)
    e.cache_ram = cache_ram_mib * MIB
    e.budget = budget
    e.notes.append("Schätzung: Compute-, Host- und Zustandspuffer sind noch nicht auf dieser Maschine kalibriert.")
    return e
