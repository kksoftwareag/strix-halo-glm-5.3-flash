#!/usr/bin/env python3
"""Terminal-Bench-Mini-20 gegen den lokalen GLM-5.3-Flash-Server fahren.

Startet den Server aus einem Preset (plus Overrides), wartet auf /health, ruft den Runner aus
bench/quality/terminal-bench-mini auf und räumt danach auf. Der Server läuft unter bench/memguard.py.

  bench/quality/tbench.py --tier smoke                          # 1 Aufgabe, Rauchtest
  bench/quality/tbench.py --tier full --preset unsloth-agent    # alle 20 Aufgaben, andere Engine
  bench/quality/tbench.py --task fix-git --attempts 1           # eine Aufgabe
  bench/quality/tbench.py --dry-run                             # nur Speicherbilanz und Kommandos
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
TBM = HERE / "terminal-bench-mini"
MEMGUARD = PROJECT / "bench" / "memguard.py"
sys.path.insert(0, str(PROJECT))
from glm53.__main__ import add_overrides, config_from   # noqa: E402
from glm53.config import build_command                   # noqa: E402
from glm53.discovery import discover_all                 # noqa: E402
from glm53.hardware import probe                         # noqa: E402

GIB = 2**30
UBUNTU_HOSTS = ("archive.ubuntu.com", "security.ubuntu.com")
DEFAULT_MIRROR = "ftp.fau.de"
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def http_json(url: str, timeout: float = 5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def wait_ready(base: str, deadline: float) -> bool:
    last = ""
    while time.time() < deadline:
        try:
            d = http_json(base + "/health", timeout=5)
            if d.get("status") == "ok":
                return True
            last = str(d)
        except (urllib.error.URLError, OSError, ValueError) as e:
            last = e.__class__.__name__
        time.sleep(3)
    log(f"Server wurde nicht bereit (zuletzt: {last})")
    return False


def probe_seconds(url: str, timeout: float = 8.0) -> float:
    t0 = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout):
            return time.time() - t0
    except Exception:
        return float("inf")


def apt_mirror_hosts(mode: str, slow_after: float = 3.0) -> list[str]:
    if mode in ("off", "aus", ""):
        return []
    host = DEFAULT_MIRROR if mode == "auto" else mode
    if mode == "auto":
        dt = probe_seconds("http://archive.ubuntu.com/ubuntu/dists/noble/Release")
        if dt <= slow_after:
            log(f"   apt-Spiegel     nicht nötig (archive.ubuntu.com antwortet in {dt:.1f}s)")
            return []
        log(f"   apt-Spiegel     archive.ubuntu.com {'antwortet nicht' if dt == float('inf') else f'braucht {dt:.1f}s'} -> {host}")
    try:
        ip = socket.getaddrinfo(host, 80, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    except OSError as e:
        log(f"   WARNUNG: Spiegel {host} nicht auflösbar ({e}); apt bleibt beim Original.")
        return []
    if mode != "auto":
        log(f"   apt-Spiegel     {host} ({ip})")
    return [f"{h}:{ip}" for h in UBUNTU_HOSTS]


def run_filtered(argv: list[str], cwd: str, raw: bool = False, every: float = 60.0, env: dict | None = None) -> int:
    if raw:
        return subprocess.call(argv, cwd=cwd, env=env)
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
    out, last, buf = sys.stdout.buffer, 0.0, b""
    assert proc.stdout
    while True:
        try:
            chunk = proc.stdout.read(4096)
        except KeyboardInterrupt:
            log("   breche den Benchmark ab …")
            stop_tree(proc.pid, grace=20.0)
            raise
        if not chunk:
            break
        buf += chunk
        parts = buf.replace(b"\r", b"\n").split(b"\n")
        buf = parts.pop()
        for part in parts:
            text = part.decode("utf-8", "replace")
            if any(c in text for c in SPINNER):
                now = time.time()
                if now - last < every:
                    continue
                last = now
            if text.strip():
                out.write(text.encode("utf-8", "replace") + b"\n")
                out.flush()
    if buf.strip():
        out.write(buf + b"\n")
        out.flush()
    return proc.wait()


def mem_available() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def leftover_servers() -> list[int]:
    out = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if (entry / "comm").read_text().strip() in ("llama", "llama-server"):
                    out.append(int(entry.name))
            except OSError:
                continue
    return out


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def descendants(pid: int) -> list[int]:
    kids: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            ppid = int(stat[stat.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(entry.name))
    out, queue = [], [pid]
    while queue:
        for child in kids.get(queue.pop(0), []):
            out.append(child)
            queue.append(child)
    return out


def kill_group(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except OSError:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def wait_gone(pids: list[int], seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if not any(alive(p) for p in pids):
            return True
        time.sleep(0.5)
    return not any(alive(p) for p in pids)


def stop_tree(pid: int, grace: float = 30.0) -> None:
    kids = descendants(pid)
    for k in kids:
        kill_group(k, signal.SIGINT)
    if not wait_gone(kids, grace):
        for k in kids:
            kill_group(k, signal.SIGKILL)
        wait_gone(kids, 15.0)
    kill_group(pid, signal.SIGTERM)
    if not wait_gone([pid], 10.0):
        kill_group(pid, signal.SIGKILL)


def profile_label(cfg) -> str:
    parts = []
    if cfg.mtp_enabled:
        parts.append(f"mtp{cfg.spec_draft_n_max}")
        if "ngram" in cfg.spec_extra_types:
            parts.append("ngram")
    else:
        parts.append("no-mtp")
    parts.append(f"thinking-{cfg.reasoning_effort}" if cfg.thinking else "no-thinking")
    if cfg.n_parallel > 1:
        parts.append(f"np{cfg.n_parallel}")
    return "-".join(parts)


def _term(signum, frame):  # pragma: no cover
    raise KeyboardInterrupt


def main() -> int:
    signal.signal(signal.SIGTERM, _term)
    ap = argparse.ArgumentParser(description="Terminal-Bench-Mini-20 gegen den lokalen Server")
    g = ap.add_argument_group("Server")
    add_overrides(g)
    g.add_argument("--reserve-gib", type=float, default=8.0, help="Spielraum für die Docker-Container; darunter Warnung (Default 8)")
    g.add_argument("--use-running", action="store_true", help="keinen Server starten, laufenden benutzen")
    g.add_argument("--endpoint", default="", help="Endpunkt überschreiben")
    b = ap.add_argument_group("Benchmark")
    b.add_argument("--tier", default="full", choices=["smoke", "full"])
    b.add_argument("--task", default="")
    b.add_argument("--tasks", default="", help="Liste (a,b,c) oder @datei")
    b.add_argument("--attempts", type=int, default=0, help="Versuche je Aufgabe (Default 2 = pass@2, 1 = pass@1)")
    b.add_argument("--concurrency", type=int, default=0)
    b.add_argument("--agent-timeout", type=int, default=0, help="Sekunden je Versuch (Default 10800)")
    b.add_argument("--results-dir", default=str(PROJECT / "state" / "quality" / "tbench"))
    b.add_argument("--job-name", default="")
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--raw-output", action="store_true")
    b.add_argument("--apt-mirror", default="auto")
    b.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    if not (TBM / "terminal_bench.py").is_file():
        log(f"{TBM} fehlt – erst 'bench/quality/fetch.sh' laufen lassen.")
        return 2
    cfg = config_from(a)
    if a.concurrency and cfg.n_parallel < a.concurrency:
        cfg = cfg.copy(n_parallel=a.concurrency, kv_unified=True)
    inv, hw = discover_all(), probe()
    cmd = build_command(cfg, inv, hw, budget=int(a.budget_gib * GIB) if a.budget_gib else None)
    r = cmd.resolved
    ctx_per_slot = cfg.ctx_size // max(1, cfg.n_parallel)
    concurrency = a.concurrency or 1
    log("== Server")
    log(f"   Engine        {r.engine.label if r.engine else '-'}")
    log(f"   Modell        {cfg.quant}  ({r.model.size_gib:.1f} GiB, Schema {r.model.arch})" if r.model else "   Modell        (fehlt)")
    log(f"   MTP           {'draft-mtp n' + str(cfg.spec_draft_n_max) if cfg.mtp_enabled else 'aus'}")
    log(f"   Kontext       {cfg.ctx_size} gesamt, {ctx_per_slot} je Slot ({cfg.n_parallel} Slots)")
    if cmd.estimate:
        for k, v in cmd.estimate.rows():
            log(f"   {k:34} {v}")
        log(f"   Bewertung     {cmd.estimate.verdict}")
    for w in cmd.warnings:
        log("   WARNUNG: " + w)
    for e in cmd.errors:
        log("   FEHLER: " + e)
    if not cmd.ok:
        return 1
    head = cmd.estimate.headroom / GIB if cmd.estimate else 0
    need = a.reserve_gib * max(1, concurrency)
    if head < need:
        log(f"   WARNUNG: nur {head:.1f} GiB Spielraum für {concurrency} Container (empfohlen {need:.0f} GiB)")

    base = a.endpoint.rstrip("/") if a.endpoint else f"http://{cfg.host}:{cfg.port}"
    if base.endswith("/v1"):
        base = base[:-3]
    endpoint = base + "/v1"
    if any(h in base for h in ("127.0.0.1", "localhost")):
        log("   WARNUNG: Der Agent läuft im Docker-Container und erreicht 127.0.0.1 des Hosts nicht – LAN-Adresse (--host) verwenden.")

    runner = [sys.executable, "terminal_bench.py", "run",
              "--endpoint", endpoint, "--model", cfg.alias, "--context-length", str(ctx_per_slot),
              "--platform", "strix-halo", "--platform-name", "AMD Ryzen AI MAX+ 395 (Strix Halo)",
              "--model-name", "GLM-5.3-Flash", "--engine", f"llama.cpp/{cfg.engine}",
              "--engine-version", (r.engine.version() if r.engine else "?"),
              "--backend", "rocm" if cfg.backend == "hip" else cfg.backend, "--backend-version", hw.rocm_version or "?",
              "--quant", cfg.quant, "--inference-profile", profile_label(cfg), "--results-dir", a.results_dir]
    tasks: list[str] = []
    if a.tasks:
        tasks = ([ln.strip() for ln in Path(a.tasks[1:]).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
                 if a.tasks.startswith("@") else [t.strip() for t in a.tasks.split(",") if t.strip()])
    elif a.task:
        tasks = [a.task]
    if not tasks:
        runner += ["--tier", a.tier]
    if a.attempts:
        runner += ["--attempts", str(a.attempts)]
    if a.concurrency:
        runner += ["--concurrency", str(a.concurrency)]
    if a.agent_timeout:
        runner += ["--agent-timeout", str(a.agent_timeout)]
    if a.job_name and not tasks:
        runner += ["--job-name", a.job_name]
    runner += [x for x in a.rest if x != "--"]
    if cfg.api_key:
        runner += ["--api-key", cfg.api_key]

    stamp = time.strftime("%Y%m%d-%H%M%S")
    logdir = PROJECT / "state" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    srvlog = logdir / f"tbench-server-{stamp}.log"
    server_cmd = [sys.executable, str(MEMGUARD), "--min-avail-gib", str(cfg.mem_guard_gib),
                  "--csv", str(logdir / f"tbench-mem-{stamp}.csv"), "--"] + cmd.argv
    log("")
    log("== Benchmark")
    log("   " + " ".join(runner))
    if not a.use_running:
        log(f"   Server-Log    {srvlog}")
        log(f"   Wächter       SIGKILL bei MemAvailable < {cfg.mem_guard_gib:.1f} GiB")
    renv = dict(os.environ)
    hosts = apt_mirror_hosts(a.apt_mirror)
    if hosts:
        renv["GLM53_EXTRA_HOSTS"] = ",".join(hosts)
        renv["GLM53_REAL_DOCKER"] = shutil.which("docker") or "/usr/bin/docker"
        renv["GLM53_SHIM_DIR"] = str(PROJECT / "state" / "quality")
        renv["PATH"] = f"{HERE / 'dockershim'}{os.pathsep}{renv.get('PATH', '')}"
    if a.dry_run:
        log("")
        log("$ " + cmd.shell())
        return 0

    proc = None
    try:
        if not a.use_running:
            env = dict(os.environ)
            env.update(cmd.env)
            fh = srvlog.open("w", buffering=1)
            fh.write("# " + cmd.shell() + "\n")
            t_start = time.time()
            proc = subprocess.Popen(server_cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            log(f"   Server gestartet (pid {proc.pid}), warte auf /health …")
            if not wait_ready(base, time.time() + 1800):
                return 1
            log(f"   bereit nach {time.time() - t_start:.0f}s")
        elif not wait_ready(base, time.time() + 30):
            log("FEHLER: kein laufender Server unter " + base)
            return 1
        t0 = time.time()
        if tasks:
            rc = 0
            for i, task in enumerate(tasks, 1):
                argv = list(runner) + ["--task", task] + (["--job-name", f"{a.job_name}-{task}"] if a.job_name else [])
                log(f"-- [{i}/{len(tasks)}] {task}")
                t1 = time.time()
                one = run_filtered(argv, cwd=str(TBM), raw=a.raw_output, env=renv)
                log(f"-- {task}: exit {one} nach {(time.time() - t1) / 60:.0f} min")
                rc = rc or one
        else:
            rc = run_filtered(runner, cwd=str(TBM), raw=a.raw_output, env=renv)
        log(f"== fertig nach {(time.time() - t0) / 3600:.2f} h, exit {rc}")
        return rc
    except KeyboardInterrupt:
        log("== abgebrochen")
        for q in descendants(os.getpid()):
            kill_group(q, signal.SIGKILL)
        return 130
    finally:
        if proc and proc.poll() is None:
            log("   stoppe Server …")
            t_stop = time.time()
            stop_tree(proc.pid, grace=30.0)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log("   WARNUNG: Wächter reagiert nicht mehr.")
            rest = leftover_servers()
            if rest:
                log(f"   WARNUNG: Server-Prozesse leben noch: {rest} – werden hart beendet.")
                for q in rest:
                    kill_group(q, signal.SIGKILL)
                wait_gone(rest, 15.0)
            log(f"   Server beendet nach {time.time() - t_stop:.0f}s, MemAvailable {mem_available() / GIB:.1f} GiB")


if __name__ == "__main__":
    sys.exit(main())
