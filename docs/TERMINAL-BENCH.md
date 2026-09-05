# Terminal-Bench-Mini-20 – GLM-5.3-Flash auf Strix Halo

Noch keine Läufe (Stand 2026-09-05). Diese Datei und `tbmini-data.js` werden von `bench/quality/report.py`
aus `state/quality/tbench/` erzeugt, sobald Ergebnisse vorliegen.

Geplant (je ein Slot, MTP an, pass@1, reasoning_effort `high`, 131 072 Token Kontext, 60 Minuten je Aufgabe):

| Lauf | Engine | Quant | Zweck |
| --- | --- | --- | --- |
| tbmini-UD-IQ2_XXS-tkmtp-agent-high | tk-mtp (#27917 auf #27773) | UD-IQ2_XXS | Hauptkandidat |
| tbmini-UD-IQ2_XXS-unsloth-agent-high | unsloth (#27754) | UD-IQ2_XXS | Engine-Vergleich bei gleichem Quant |
| tbmini-UD-IQ1_S-tkmtp-agent-high | tk-mtp | UD-IQ1_S | kleinster Quant |
| tbmini-UD-Q2_K_XL-tkmtp-agent-high | tk-mtp | UD-Q2_K_XL | nur mit kleinem VRAM-Carve-out |
| tbmini-Spark-Q2XL-MTP-spark-agent-high | unsloth | Spark-Q2XL-MTP | nur mit kleinem VRAM-Carve-out |
