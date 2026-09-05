#!/usr/bin/env python3
"""Parallelitätsgrenzen je Quant: Wie viele gleichzeitige Kontexte welcher Größe passen in den Speicher?
Rechnet mit dem Speichermodell (glm53.memory) gegen ein festes Budget (Default 105 GiB = MemAvailable der
leeren Maschine mit 16-GiB-Carve-out; --budget-gib 120 für 1 GiB Carve-out).

  python3 bench/context_limits.py [--budget-gib 105] [--max-slots 32]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from glm53.discovery import discover_all   # noqa: E402
from glm53.memory import estimate, kv_bytes_per_token, GIB   # noqa: E402

SIZES = (16384, 32768, 65536, 131072, 262144)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gib", type=float, default=105.0)
    ap.add_argument("--max-slots", type=int, default=32)
    ap.add_argument("--ub", type=int, default=1024)
    a = ap.parse_args()
    inv = discover_all()
    seen, models = set(), []
    for m in sorted(inv.models, key=lambda m: m.total_bytes):
        if m.label not in seen:
            seen.add(m.label); models.append(m)
    kv, idx = kv_bytes_per_token("q8_0", "q8_0")
    print(f"KV+Indexer je Token: {kv + idx} Byte = {(kv + idx) * 32768 / GIB:.2f} GiB je 32k Kontext (q8_0)")
    print(f"Budget {a.budget_gib:.1f} GiB, ubatch {a.ub}\n")
    print(f"{'Quant':18} {'Gewichte':>9} | " + " ".join(f"{s // 1024:>4}k" for s in SIZES) + "   (Slots mit MTP / ohne MTP)")

    def max_slots(m, per_slot, mtp):
        best = 0
        for n in range(1, a.max_slots + 1):
            e = estimate(total_bytes=m.total_bytes, nextn_bytes=m.nextn_bytes, mtp=mtp and m.has_nextn, ctx=per_slot * n, n_parallel=n,
                         ctk="q8_0", ctv="q8_0", ubatch=a.ub, cache_ram_mib=512, mmproj_bytes=0, budget=int(a.budget_gib * GIB))
            if e.headroom < 0:
                break
            best = n
        return best

    for m in models:
        cells = [f"{max_slots(m, s, True):2d}/{max_slots(m, s, False):2d}" for s in SIZES]
        print(f"{m.label:18} {m.size_gib:7.1f} G | " + " ".join(f"{c:>5}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
