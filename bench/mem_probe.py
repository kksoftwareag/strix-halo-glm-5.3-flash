#!/usr/bin/env python3
"""Footprint-Messung: Server (aus Preset/Overrides) unter memguard starten, eine Anfrage, hart beenden.
Meldet Ladezeit, pp/tg, Draft-Akzeptanz und Peak-Verbrauch (MemAvailable-Delta, GTT, RSS).

  python3 bench/mem_probe.py NAME --preset tkmtp-agent [--ctx 32768 --no-mtp …]
Ergebnisse: bench/results/mem/NAME.{json,csv,log}
"""
from __future__ import annotations

import argparse, json, os, signal, socket, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from glm53.__main__ import add_overrides, config_from      # noqa: E402
from glm53.config import build_command                      # noqa: E402
from glm53.discovery import discover_all                    # noqa: E402
from glm53.hardware import probe                            # noqa: E402

OUT = ROOT / "bench" / "results" / "mem"
OUT.mkdir(parents=True, exist_ok=True)
PORT = int(os.environ.get("PROBE_PORT", "8098"))
BASE = f"http://127.0.0.1:{PORT}"
PROMPT = "Zähle von 1 bis 30 und erkläre danach in 5 Sätzen, was ein KV-Cache ist."


def http(path, data=None, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def kill_servers(sig=signal.SIGKILL):
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if (entry / "comm").read_text().strip() in ("llama-server", "llama"):
                    os.kill(int(entry.name), sig)
            except OSError:
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    add_overrides(ap)
    ap.add_argument("--min-avail-gib", type=float, default=8.0)
    ap.add_argument("--max-tokens", type=int, default=300)
    a = ap.parse_args()
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            print(json.dumps({"name": a.name, "error": f"Port {PORT} belegt"})); return 2
    a.host, a.port = "127.0.0.1", PORT
    cfg = config_from(a)
    cmd = build_command(cfg, discover_all(), probe(), budget=int(a.budget_gib * 2**30) if a.budget_gib else None)
    if not cmd.ok:
        print(json.dumps({"name": a.name, "error": cmd.errors})); return 1
    argv = [sys.executable, str(ROOT / "bench/memguard.py"), "--min-avail-gib", str(a.min_avail_gib), "--csv", str(OUT / f"{a.name}.csv"), "--"] + cmd.argv
    log = (OUT / f"{a.name}.log").open("w"); log.write(cmd.shell() + "\n"); log.flush()
    env = dict(os.environ); env.update(cmd.env)
    t0 = time.time()
    proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    res = {"name": a.name, "preset": a.preset, "cmd": cmd.shell(), "estimate_gib": round(cmd.estimate.total / 2**30, 2) if cmd.estimate else None}
    ready = False
    while proc.poll() is None and time.time() - t0 < float(os.environ.get("READY_TIMEOUT", "1500")):
        try:
            if http("/health", timeout=2).get("status") == "ok":
                ready = True; break
        except Exception:
            pass
        time.sleep(1)
    if ready:
        res["load_s"] = round(time.time() - t0, 1)
        try:
            r = http("/v1/chat/completions", {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": a.max_tokens, "stream": False})
            t = r.get("timings", {})
            res.update(tg=t.get("predicted_per_second"), pp=t.get("prompt_per_second"), n_gen=t.get("predicted_n"),
                       draft_n=t.get("draft_n"), draft_acc=t.get("draft_n_accepted"))
            res["text_head"] = (r.get("choices", [{}])[0].get("message", {}).get("content") or "")[:200]
        except Exception as e:  # noqa: BLE001
            res["req_error"] = repr(e)
        time.sleep(2)
    else:
        res["error"] = "Server nicht bereit (Timeout/Abbruch)"
    kill_servers()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL); proc.wait()
    res["guard_exit"] = proc.returncode
    log.close()
    tail = (OUT / f"{a.name}.log").read_text(errors="replace").splitlines()
    res["guard_summary"] = next((ln for ln in reversed(tail) if "[memguard] fertig" in ln), "")
    for ln in tail:
        if "model buffer size" in ln or "KV buffer size" in ln or "compute buffer size" in ln or "RS buffer size" in ln:
            res.setdefault("buffers", []).append(ln.strip()[-70:])
    (OUT / f"{a.name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
