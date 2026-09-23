"""看护 jev-mario 直播：观战页 + ngrok/cloudflared 隧道，谁挂了就重启谁。

    .venv-mario/Scripts/python.exe supervise.py --level 1-1 --port 8123 --fps 45

为什么需要它：
  - 免费隧道没有 uptime 保证，断了不会自己回来；
  - 观战页脚本如果抛异常退出，直播就静默黑掉了；
  - 隧道每次重连都会换一个新地址，必须有人把新地址捞出来告诉用户。

行为：
  - 两个子进程各自独立看护，退出后 3 秒重启；
  - 从隧道输出里解析公网 URL，写到 public-url.txt 并打印；
  - 隧道换地址时打印 "地址已变" 提示（旧地址立即失效）；
  - Ctrl+C 时一起收掉子进程。

注意：它自己也得活着，所以双击 .cmd 之后**不要关那个窗口**。
"""
import argparse
import atexit
import contextlib
import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL_FILE = HERE / "public-url.txt"
URL_RE = re.compile(
    r"https://[a-z0-9][a-z0-9-]*\."
    r"(?:trycloudflare\.com|ngrok-free\.(?:app|dev)|ngrok\.io)"
)
RESTART_DELAY = 3.0
NGROK = os.environ.get("NGROK_BIN") or shutil.which("ngrok") or "ngrok"
NGROK_DOMAIN = os.environ.get("NGROK_DOMAIN", "")
# ngrok 免费版不允许 agent 走代理（ERR_NGROK_9009）。只要清掉这几个环境变量，
# 它就直接连云端、不走代理。proxy_url 别写进 ngrok.yml 就行。
PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")

stop = threading.Event()
current_url = None
lock = threading.Lock()


def log(*parts):
    print(time.strftime("[%H:%M:%S]"), *parts, flush=True)


def pump(stream, prefix, on_line=None):
    for line in stream:
        line = line.rstrip()
        if not line:
            continue
        if on_line:
            on_line(line)
        if prefix:
            print(f"{prefix} {line}", flush=True)


def on_tunnel_line(line):
    global current_url
    match = URL_RE.search(line)
    if not match:
        return
    url = match.group(0)
    with lock:
        if url == current_url:
            return
        previous, current_url = current_url, url
    URL_FILE.write_text(url + "\n", encoding="utf-8")
    log("=" * 68)
    if previous:
        log("公网地址已变（旧地址立即失效）：")
        log(f"  旧  {previous}")
    else:
        log("公网直播地址：")
    log(f"  {url}")
    log(f"（同时写入 {URL_FILE.name}）")
    log("=" * 68)


def port_busy(port: int) -> bool:
    import socket
    with socket.socket() as probe:
        probe.settimeout(0.4)
        return probe.connect_ex(("127.0.0.1", port)) == 0


LOCK_FILE = HERE / "supervise.lock"
_lock_identity = None


def take_singleton_lock(port: int) -> bool:
    """同一台机器上只允许一个 supervisor 存活。

    光查"端口是否被占"不够：两个 supervisor 同时启动时都还没绑端口，会各起一个观战页，
    Windows 上两个 http.server 能同时 bind 同一端口，连接被分摊 → 页面时好时坏，极难排查。
    用文件锁把窗口期堵掉。
    """
    global _lock_identity
    for attempt in range(2):
        try:
            handle = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            info = os.fstat(handle)
            _lock_identity = (info.st_dev, info.st_ino)
            try:
                with os.fdopen(handle, "w", encoding="ascii") as output:
                    output.write(str(os.getpid()))
            except BaseException:
                release_lock()
                raise
            return True
        except FileExistsError:
            try:
                info = LOCK_FILE.stat()
                stale = int(LOCK_FILE.read_text().strip() or "0")
            except FileNotFoundError:
                continue
            except ValueError:
                stale = 0
            # A creator may still be writing its PID. Only reclaim malformed
            # files after a grace period, and never over an active viewer.
            expired = (stale > 0 and not _alive(stale)) or (
                stale <= 0 and time.time() - info.st_mtime > 5 and not port_busy(port)
            )
            if expired:
                log(f"发现残留锁（PID {stale} 已不在），清掉重试")
                try:
                    current = LOCK_FILE.stat()
                    if (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino):
                        LOCK_FILE.unlink()
                except FileNotFoundError:
                    pass
                continue
            log(f"已经有 supervisor 在跑（PID {stale}）。要重启就先把它关掉。")
            return False
    return False


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, errors="replace", timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        # An unavailable process query is not evidence that the owner died.
        return True
    if result.returncode:
        return True
    return any(len(row) > 1 and row[1] == str(pid)
               for row in csv.reader((result.stdout or "").splitlines()))


def release_lock():
    global _lock_identity
    if _lock_identity is None:
        return
    with contextlib.suppress(OSError):
        info = LOCK_FILE.stat()
        if (info.st_dev, info.st_ino) == _lock_identity:
            LOCK_FILE.unlink()
    _lock_identity = None


def terminate_child(child):
    """Reap the child before releasing the supervisor's singleton lock."""
    if child.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            child.kill()
        child.wait(timeout=5)


def supervise(name, argv, on_line=None, child_env=None):
    fast_exits = 0
    while not stop.is_set():
        log(f"启动 {name}: {' '.join(str(a) for a in argv[:3])} ...")
        launched_at = time.perf_counter()
        try:
            child = subprocess.Popen(
                argv, cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1, env=child_env,
            )
        except OSError as exc:
            log(f"{name} 起不来：{exc}")
            return
        output = threading.Thread(target=pump, args=(child.stdout, f"[{name}]", on_line), daemon=True)
        output.start()
        try:
            while not stop.is_set() and child.poll() is None:
                stop.wait(0.5)
            if stop.is_set():
                return
        finally:
            terminate_child(child)
            output.join(timeout=1)
            if not output.is_alive():
                child.stdout.close()
        lived = time.perf_counter() - launched_at
        fast_exits = fast_exits + 1 if lived < 2.5 else 0
        if fast_exits >= 3:
            log(f"{name} 连续 {fast_exits} 次秒退，八成是端口被占或依赖缺失 —— 不再重启，先修问题")
            return
        log(f"{name} 退出了（code={child.returncode}，活了 {lived:.1f}s），{RESTART_DELAY:.0f} 秒后重启")
        stop.wait(RESTART_DELAY)


def main():
    stop.clear()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", default="1-1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--fps", type=float, default=45.0)
    ap.add_argument("--mode", default="chat")
    ap.add_argument("--model", default="dual", choices=["local", "jev", "dual"],
                    help="模型来源：dual(双画面，默认) / local(本地) / jev(官方TypeSafe)")
    ap.add_argument("--tunnel", default=None, choices=["ngrok", "cloudflared", "none"],
                    help="公网隧道方式：ngrok(固定域名，默认) / cloudflared(临时隧道) / none(不起隧道)")
    ap.add_argument(
        "--domain", default=NGROK_DOMAIN,
        help="ngrok 固定域名（可选；默认读取 NGROK_DOMAIN）",
    )
    ap.add_argument("--web-search", action="store_true",
                    help="重复场景触发教练时启用联网攻略搜索（需要 BAIDU_AI_SEARCH_API_KEY）")
    ap.add_argument("--verified-route", action="store_true",
                     help="直接启用已验证恢复路线（用于验收；页面会明确标记）")
    ap.add_argument("--replay-only", action="store_true",
                    help="离线双画面验收，不调用模型（自动禁用公网隧道）")
    ap.add_argument("--model-only", action="store_true",
                    help="仅同步模型决策；禁用路线、守卫、脱困、教练和规则兜底")
    ap.add_argument("--stay-on-level", action="store_true",
                    help="通关 --level 后停留在成功画面，不自动进入下一关（验收/观战用）")
    a = ap.parse_args()
    if a.model_only and (a.replay_only or a.verified_route or a.web_search):
        ap.error("--model-only cannot be combined with --replay-only, --verified-route or --web-search")
    if a.replay_only and a.tunnel not in (None, "none"):
        ap.error("--replay-only 只支持 --tunnel none")
    a.tunnel = a.tunnel or ("none" if a.replay_only else "ngrok")

    if a.web_search:
        os.environ["MENTOR_WEB_SEARCH_ENABLED"] = "true"

    python = sys.executable
    viewer = [python, str(HERE / "watch_local.py"), "--level", a.level,
              "--port", str(a.port), "--fps", str(a.fps), "--mode", a.mode, "--model", a.model]
    if a.verified_route:
        viewer.append("--verified-route")
    if a.replay_only:
        viewer.append("--replay-only")
    if a.model_only:
        viewer.append("--model-only")
    if a.stay_on_level:
        viewer.append("--stay-on-level")

    def shutdown(*_):
        if stop.is_set():
            return
        log("收到中断，正在收尾 ...")
        stop.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log(f"观战页本机地址: http://127.0.0.1:{a.port}/")
    if not take_singleton_lock(a.port):
        raise SystemExit(1)
    atexit.register(release_lock)
    threads = []
    if port_busy(a.port):
        # 端口已被占用就别再起一个：Windows 上两个 http.server 能同时 bind 同一端口，
        # 连接会被分摊，现象是"页面时好时坏"，非常难查。宁可少起一个。
        log(f"警告：{a.port} 端口上已经有观战页在跑，本次只接管隧道、不再起第二个观战页。")
        log("      （要换新代码，先把旧的那个关掉再运行本脚本。）")
    else:
        threads.append(threading.Thread(target=supervise, args=("viewer", viewer), daemon=True))
    if a.tunnel == "none":
        log("已禁用隧道，只跑本机观战页")
    elif a.tunnel == "ngrok":
        tunnel = [NGROK, "http", str(a.port)]
        if a.domain:
            url = f"https://{a.domain}/"
            URL_FILE.write_text(url + "\n", encoding="utf-8")
            log("=" * 68)
            log(f"公网直播地址（固定，重启不变）：{url}")
            log(f"（同时写入 {URL_FILE.name}）")
            log("=" * 68)
            tunnel.append(f"--domain={a.domain}")
        else:
            log("等待 ngrok 分配公网地址；也可用 --domain 或 NGROK_DOMAIN 指定固定域名")
        # Only ngrok bypasses the host proxy. The viewer must retain its
        # environment so TypeSafe Jev and the optional mentor search can reach
        # their APIs instead of silently degrading to fallback actions.
        tunnel_env = os.environ.copy()
        for key in PROXY_VARS:
            tunnel_env.pop(key, None)
        threads.append(threading.Thread(
            target=supervise,
            args=("ngrok", tunnel, on_tunnel_line, tunnel_env),
            daemon=True,
        ))
    else:
        tunnel = ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{a.port}", "--no-autoupdate"]
        threads.append(threading.Thread(target=supervise, args=("tunnel", tunnel, on_tunnel_line), daemon=True))
    failed = False
    try:
        for thread in threads:
            thread.start()
        while not stop.is_set():
            if not any(thread.is_alive() for thread in threads):
                log("没有运行中的看护任务，退出")
                failed = bool(threads)
                break
            stop.wait(0.5)
    except KeyboardInterrupt:
        shutdown()
    finally:
        stop.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join()
        release_lock()
        atexit.unregister(release_lock)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
