#!/usr/bin/env python3
"""Sweep 2: MTP-Feintuning über den Server (unter memguard): Draft-Tiefe, p_min, Temperatur, ngram-mod, je Engine × Quant.

  python3 bench/mtp_sweep.py --engine tk-mtp --quant UD-IQ2_XXS [--ctx 32768] [--quick] [--reasoning high]
Ergebnisse: bench/results/mtp/<engine>-<quant>-<name>.json + summary.jsonl
"""
from __future__ import annotations

import argparse, json, os, signal, socket, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from glm53.config import ServerConfig, build_command   # noqa: E402
from glm53.discovery import discover_all                # noqa: E402
from glm53.hardware import probe                        # noqa: E402

OUT = ROOT / "bench" / "results" / "mtp"; OUT.mkdir(parents=True, exist_ok=True)
PORT = 8097; BASE = f"http://127.0.0.1:{PORT}"
PROMPTS = [
    ("code", "Schreibe eine vollständige Python-Klasse `LRUCache` mit get/put in O(1), Typannotationen, Docstrings und 5 pytest-Tests."),
    ("prosa", "Erkläre einem Erstsemester in etwa 400 Wörtern, wie ein Mixture-of-Experts-Sprachmodell funktioniert und warum es schneller ist als ein dichtes Modell gleicher Größe."),
    ("reasoning", "Ein Zug fährt um 8:00 mit 80 km/h von A nach B (240 km). Ein zweiter Zug fährt um 8:30 mit 120 km/h von B nach A. Wann und wo treffen sie sich? Rechne Schritt für Schritt."),
]


def configs(quick: bool):
    c = [
        ("nomtp-t1.0",       dict(mtp_enabled=False, temp=1.0)),
        ("n1-p0.75-t1.0",    dict(mtp_enabled=True, spec_draft_n_max=1, spec_draft_p_min=0.75, temp=1.0)),
        ("n2-p0.75-t1.0",    dict(mtp_enabled=True, spec_draft_n_max=2, spec_draft_p_min=0.75, temp=1.0)),
        ("n3-p0.75-t1.0",    dict(mtp_enabled=True, spec_draft_n_max=3, spec_draft_p_min=0.75, temp=1.0)),
        ("n4-p0.75-t1.0",    dict(mtp_enabled=True, spec_draft_n_max=4, spec_draft_p_min=0.75, temp=1.0)),
        ("n3-p0.0-t1.0",     dict(mtp_enabled=True, spec_draft_n_max=3, spec_draft_p_min=0.0, temp=1.0)),
        ("n3-p0.75-ng-t1.0", dict(mtp_enabled=True, spec_draft_n_max=3, spec_draft_p_min=0.75, spec_extra_types="ngram-mod", temp=1.0)),
        ("n2-p0.75-t0.6",    dict(mtp_enabled=True, spec_draft_n_max=2, spec_draft_p_min=0.75, temp=0.6)),
        ("n2-p0.75-t0.0",    dict(mtp_enabled=True, spec_draft_n_max=2, spec_draft_p_min=0.75, temp=0.0)),
    ]
    return c[:3] if quick else c


def http(path, data=None, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(data).encode() if data else None, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_ready(proc, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout and proc.poll() is None:
        try:
            if http("/health", timeout=3).get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def kill_llama():
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if (entry / "comm").read_text().strip() in ("llama-server", "llama"):
                    os.kill(int(entry.name), signal.SIGKILL)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="tk-mtp")
    ap.add_argument("--quant", default="UD-IQ2_XXS")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--ub", type=int, default=1024)
    ap.add_argument("--reasoning", default="high")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--min-avail-gib", type=float, default=8.0)
    a = ap.parse_args()
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            print("Port belegt"); sys.exit(2)
    inv, hw = discover_all(), probe()
    summary = OUT / "summary.jsonl"
    for name, over in configs(a.quick):
        tag = f"{a.engine}-{a.quant}-{name}"
        if (OUT / f"{tag}.json").exists() and not a.force:
            print(f"### {tag}: vorhanden"); continue
        cfg = ServerConfig(engine=a.engine, quant=a.quant, ctx_size=a.ctx, ubatch=a.ub, reasoning_effort=a.reasoning,
                           host="127.0.0.1", port=PORT, metrics=False, **over)
        cmd = build_command(cfg, inv, hw)
        if not cmd.ok:
            print(f"### {tag}: FEHLER {cmd.errors}"); continue
        argv = [sys.executable, str(ROOT / "bench/memguard.py"), "--min-avail-gib", str(a.min_avail_gib), "--csv", str(OUT / f"{tag}.csv"), "--"] + cmd.argv
        log = (OUT / f"{tag}.log").open("w"); log.write(cmd.shell() + "\n"); log.flush()
        print(f"### {time.strftime('%H:%M:%S')} {tag}", flush=True)
        env = dict(os.environ); env.update(cmd.env)
        t0 = time.time()
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        res = {"tag": tag, "engine": a.engine, "quant": a.quant, "name": name, "over": over, "runs": [], "error": ""}
        try:
            if not wait_ready(proc, 1500):
                res["error"] = "nicht bereit"; print("   FEHLER: nicht bereit")
            else:
                res["load_s"] = time.time() - t0
                http("/v1/chat/completions", {"messages": [{"role": "user", "content": "Sag nur: bereit."}], "max_tokens": 16})
                for key, p in PROMPTS:
                    tq = time.time()
                    r = http("/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "max_tokens": a.max_tokens, "stream": False})
                    t = r.get("timings", {})
                    res["runs"].append({"prompt": key, "wall": time.time() - tq, **t})
                    dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
                    print(f"   {key:10} pp {t.get('prompt_per_second', 0):6.1f} | tg {t.get('predicted_per_second', 0):6.2f} t/s | n={t.get('predicted_n')}"
                          + (f" | draft {da}/{dn} = {da / dn:.0%}" if dn else ""), flush=True)
        except Exception as e:  # noqa: BLE001
            res["error"] = f"{e.__class__.__name__}: {e}"; print("   FEHLER:", res["error"])
        finally:
            kill_llama()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL); proc.wait()
            log.close()
        if res["runs"]:
            tg = [x.get("predicted_per_second", 0) for x in res["runs"]]
            res["tg_mean"] = sum(tg) / len(tg)
            res["tg_code"] = next((x.get("predicted_per_second") for x in res["runs"] if x["prompt"] == "code"), None)
            dn = sum(x.get("draft_n", 0) for x in res["runs"]); da = sum(x.get("draft_n_accepted", 0) for x in res["runs"])
            res["accept"] = da / dn if dn else None
            print(f"   => tg Ø {res['tg_mean']:.2f} t/s" + (f", Akzeptanz {res['accept']:.0%}" if dn else "") + f", Load {res.get('load_s', 0):.0f}s", flush=True)
        (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        with summary.open("a") as f:
            f.write(json.dumps({k: res.get(k) for k in ("tag", "engine", "quant", "name", "tg_mean", "tg_code", "accept", "load_s", "error")}) + "\n")
        time.sleep(3)
    print("### fertig", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
