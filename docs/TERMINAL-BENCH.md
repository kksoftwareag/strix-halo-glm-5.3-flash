# Terminal-Bench-Mini-20 – GLM-5.3-Flash auf Strix Halo

Noch keine Läufe (Stand 2026-09-05). Diese Datei und `tbmini-data.js` werden von `bench/quality/report.py`
aus `state/quality/tbench/` erzeugt, sobald Ergebnisse vorliegen.

Geplant (je ein Slot, MTP an, pass@1, reasoning_effort `high`, 131 072 Token Kontext, 60 Minuten je Aufgabe):

| Lauf | Engine | Quant | Zweck |
| --- | --- | --- | --- |
| tbmini-UD-IQ2_XXS-unsloth-agent-high | unsloth (#27754, MTP) | UD-IQ2_XXS | Hauptkandidat nach dem Stream-Benchmark |
| tbmini-UD-IQ2_XXS-tk-plain-high | tk (#27773, ohne MTP) | UD-IQ2_XXS | Engine-Vergleich bei gleichem Quant |
| tbmini-UD-IQ1_M-unsloth-agent-high | unsloth | UD-IQ1_M | kleinerer Quant, +10 % Decode |
| tbmini-UD-Q2_K_XL-unsloth-agent-high | unsloth | UD-Q2_K_XL | nur mit kleinem VRAM-Carve-out |
| tbmini-Spark-Q2XL-MTP-spark-agent-high | unsloth | Spark-Q2XL-MTP | nur mit kleinem VRAM-Carve-out |
