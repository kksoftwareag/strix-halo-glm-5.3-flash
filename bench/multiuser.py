#!/usr/bin/env python3
"""Sweep 3: Mehrnutzer-Durchsatz. Startet den Server (unter memguard) mit -np N --kv-unified und fährt
1/2/4/8 gleichzeitige Anfragen mit je eigenem Fülltext (kein geteilter Prompt-Cache) und Codewort-Prüfung
(vertauschte Antworten zwischen Slots, vgl. llama.cpp #25992).

  python3 bench/multiuser.py --engine tk --quant UD-IQ2_XXS --levels 1,2,4,8 --ctx-tokens 8000 --max-tokens 512
  python3 bench/multiuser.py --engine tk-mtp --mtp --levels 1,2,4          # MTP mit mehreren Slots (erwartet: schlechter)
Ergebnisse: bench/results/multi/<engine>-<quant>-<stamp>.json
"""
from __future__ import annotations

import argparse, json, os, random, signal, socket, subprocess, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from glm53.config import ServerConfig, build_command   # noqa: E402
from glm53.discovery import discover_all                # noqa: E402
from glm53.hardware import probe                        # noqa: E402

OUT = ROOT / "bench" / "results" / "multi"; OUT.mkdir(parents=True, exist_ok=True)
PORT = 8096; BASE = f"http://127.0.0.1:{PORT}"
WORDS = ("Speicher Kontext Modell Experte Router Token Cache Kernel Puffer Gewicht Schicht Zustand Faltung Matrix "
         "Vektor Abfrage Antwort Fehler Messung Durchsatz Latenz Warteschlange Bandbreite Takt Kern Faden").split()


def http(path, data=None, timeout=3600):
    req = urllib.request.Request(BASE + path, data=json.dumps(data).encode() if data else None, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def filler(seed: int, n_tokens: int) -> str:
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) for _ in range(int(n_tokens * 0.75)))


def kill_llama():
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if (entry / "comm").read_text().strip() in ("llama-server", "llama"):
                    os.kill(int(entry.name), signal.SIGKILL)
            except OSError:
                pass


def one_user(i: int, ctx_tokens: int, max_tokens: int, results: list):
    code = f"KENNWORT-{random.Random(i * 7919).randint(1000, 9999)}"
    prompt = (f"Merke dir das Codewort {code}. Hier ist ein langer Fülltext:\n{filler(i, ctx_tokens)}\n\n"
              f"Nenne zuerst das Codewort und erkläre dann ausführlich, wie ein KV-Cache funktioniert.")
    t0 = time.time()
    try:
        r = http("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": False})
        t = r.get("timings", {})
        text = (r.get("choices", [{}])[0].get("message", {}).get("content") or "")
        results.append({"user": i, "wall": time.time() - t0, "ok": code in text, "code": code, **t})
    except Exception as e:  # noqa: BLE001
        results.append({"user": i, "wall": time.time() - t0, "error": repr(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="tk")
    ap.add_argument("--quant", default="UD-IQ2_XXS")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--ctx-tokens", type=int, default=8000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--ctx-per-slot", type=int, default=32768)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--min-avail-gib", type=float, default=8.0)
    ap.add_argument("--budget-gib", type=float, default=None, help="Speicherbudget für die Vorprüfung statt MemAvailable (z.B. 105)")
    ap.add_argument("--batch", type=int, default=None, help="-b (Logical Batch) überschreiben")
    ap.add_argument("--ub", type=int, default=None, help="-ub (Physical Batch) überschreiben")
    a = ap.parse_args()
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            print("Port belegt"); sys.exit(2)
    levels = [int(x) for x in a.levels.split(",")]
    slots = max(levels)
    cfg = ServerConfig(engine=a.engine, quant=a.quant, n_parallel=slots, kv_unified=slots > 1, ctx_size=a.ctx_per_slot * slots,
                       mtp_enabled=a.mtp, reasoning_effort=a.reasoning, host="127.0.0.1", port=PORT, metrics=False)
    if a.batch:
        cfg = cfg.copy(batch=a.batch)
    if a.ub:
        cfg = cfg.copy(ubatch=a.ub)
    cmd = build_command(cfg, discover_all(), probe(), budget=int(a.budget_gib * 2**30) if a.budget_gib else None)
    if not cmd.ok:
        print("FEHLER:", cmd.errors); sys.exit(1)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{a.engine}-{a.quant}-{'mtp' if a.mtp else 'nomtp'}-{stamp}"
    log = (OUT / f"{tag}.log").open("w"); log.write(cmd.shell() + "\n"); log.flush()
    argv = [sys.executable, str(ROOT / "bench/memguard.py"), "--min-avail-gib", str(a.min_avail_gib), "--csv", str(OUT / f"{tag}.csv"), "--"] + cmd.argv
    env = dict(os.environ); env.update(cmd.env)
    proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    out = {"tag": tag, "cmd": cmd.shell(), "levels": []}
    try:
        t0 = time.time()
        while proc.poll() is None and time.time() - t0 < 1500:
            try:
                if http("/health", timeout=3).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            print("Server nicht bereit"); sys.exit(1)
        for n in levels:
            results: list = []
            ths = [threading.Thread(target=one_user, args=(100 * n + i, a.ctx_tokens, a.max_tokens, results)) for i in range(n)]
            t1 = time.time()
            for t in ths: t.start()
            for t in ths: t.join()
            wall = time.time() - t1
            gen = sum(r.get("predicted_n", 0) for r in results)
            per = [r.get("predicted_per_second", 0) for r in results if "error" not in r]
            dn = sum(r.get("draft_n", 0) for r in results); da = sum(r.get("draft_n_accepted", 0) for r in results)
            pps = [r.get("prompt_per_second", 0) for r in results if "error" not in r and r.get("prompt_per_second")]
            lvl = {"users": n, "wall_s": wall, "tokens": gen, "total_tps": gen / wall if wall else 0,
                   "decode_sum_tps": sum(per),   # Summe der Decode-Raten aller Streams (Prefill-Phasen herausgerechnet)
                   "pp_tps": sum(pps) / len(pps) if pps else 0, "prompt_n": sum(r.get("prompt_n", 0) for r in results),
                   "per_user_tps": sum(per) / len(per) if per else 0, "mixups": sum(1 for r in results if r.get("ok") is False),
                   "errors": sum(1 for r in results if "error" in r), "accept": (da / dn) if dn else None,
                   "ttft_s": sum(r.get("prompt_ms", 0) for r in results) / 1000 / max(1, len(results))}
            out["levels"].append(lvl)
            print(f"   {n:2d} Nutzer: Σ Decode {lvl['decode_sum_tps']:6.1f} t/s | je Nutzer {lvl['per_user_tps']:5.1f} t/s | pp je Nutzer {lvl['pp_tps']:6.1f} t/s | TTFT {lvl['ttft_s']:5.1f}s | inkl. Prefill {lvl['total_tps']:4.1f} t/s | Mix-ups {lvl['mixups']} | Fehler {lvl['errors']}"
                  + (f" | Draft {lvl['accept']:.0%}" if lvl["accept"] is not None else ""), flush=True)
    finally:
        kill_llama()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL); proc.wait()
        log.close()
    (OUT / f"{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("Ergebnis:", OUT / f"{tag}.json")


if __name__ == "__main__":
    main()
