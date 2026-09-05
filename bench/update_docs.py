#!/usr/bin/env python3
"""Setzt die Tabellen aus bench/report_speed.py zwischen die Marker STREAMS:BEGIN/END in docs/benchmarks.html
und schreibt bench/results/SPEED.md (Markdown-Fassung).

  python3 bench/update_docs.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
page = ROOT / "docs" / "benchmarks.html"
html = subprocess.run([sys.executable, str(ROOT / "bench" / "report_speed.py"), "--html"], capture_output=True, text=True, check=True).stdout
md = subprocess.run([sys.executable, str(ROOT / "bench" / "report_speed.py")], capture_output=True, text=True, check=True).stdout
s = page.read_text()
s = re.sub(r"<!-- STREAMS:BEGIN -->.*?<!-- STREAMS:END -->", "<!-- STREAMS:BEGIN -->\n" + html + "\n<!-- STREAMS:END -->", s, flags=re.S)
page.write_text(s)
(ROOT / "bench" / "results" / "SPEED.md").write_text("# Geschwindigkeitsmessungen\n\n" + md)
print("docs/benchmarks.html und bench/results/SPEED.md aktualisiert")
