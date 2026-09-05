"""Hardware-Erkennung: Speicher (MemAvailable ist die einzige verlässliche Größe), GTT/VRAM, ROCm, Kernel."""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

GIB = 2**30


@dataclass
class Hardware:
    mem_total: int = 0
    mem_available: int = 0
    gtt_total: int = 0
    gtt_used: int = 0
    vram_total: int = 0
    vram_used: int = 0
    rocm_version: str = ""
    gfx: str = ""
    cmdline: str = ""
    n_cpu: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def vram_carveout_gib(self) -> float:
        return self.vram_total / GIB


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return out


def _sysfs_int(pattern: str) -> int:
    for p in sorted(Path("/sys/class/drm").glob(pattern)):
        try:
            return int(p.read_text().strip())
        except (OSError, ValueError):
            continue
    return 0


def probe() -> Hardware:
    hw = Hardware()
    mi = _meminfo()
    hw.mem_total, hw.mem_available = mi.get("MemTotal", 0), mi.get("MemAvailable", 0)
    hw.gtt_total = _sysfs_int("card*/device/mem_info_gtt_total")
    hw.gtt_used = _sysfs_int("card*/device/mem_info_gtt_used")
    hw.vram_total = _sysfs_int("card*/device/mem_info_vram_total")
    hw.vram_used = _sysfs_int("card*/device/mem_info_vram_used")
    try:
        hw.cmdline = Path("/proc/cmdline").read_text().strip()
    except OSError:
        pass
    import os
    hw.n_cpu = os.cpu_count() or 0
    if shutil.which("hipcc"):
        try:
            out = subprocess.run(["hipcc", "--version"], capture_output=True, text=True, timeout=10).stdout
            m = re.search(r"HIP version: ([\d.]+)", out)
            hw.rocm_version = m.group(1) if m else ""
        except (OSError, subprocess.SubprocessError):
            pass
    if shutil.which("rocminfo"):
        try:
            out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10).stdout
            m = re.search(r"Name:\s+(gfx\d+\w*)", out)
            hw.gfx = m.group(1) if m else ""
        except (OSError, subprocess.SubprocessError):
            pass
    if "amdgpu.gttsize" not in hw.cmdline:
        hw.notes.append("Kernel-Parameter amdgpu.gttsize fehlt – die GPU darf dann nur einen Teil des RAM nutzen")
    if hw.vram_total >= 8 * GIB:
        hw.notes.append(f"BIOS-VRAM-Carve-out {hw.vram_total / GIB:.0f} GiB: llama.cpp/HIP legt Gewichte in GTT, der Carve-out bleibt ungenutzt "
                        f"(UMA Frame Buffer im BIOS auf 512 MiB–1 GiB stellen bringt ~{hw.vram_total / GIB - 1:.0f} GiB)")
    return hw
