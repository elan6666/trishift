from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import getpass
import os
import tarfile
import time

import paramiko


@dataclass(frozen=True)
class HostCfg:
    hostname: str
    port: int
    user: str


def _load_ssh_config(alias: str) -> HostCfg:
    cfg_path = Path.home() / ".ssh" / "config"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing SSH config: {cfg_path}")
    cfg = paramiko.SSHConfig()
    cfg.parse(cfg_path.open("r", encoding="utf-8", errors="replace"))
    host = cfg.lookup(alias)
    hostname = host.get("hostname", alias)
    port = int(host.get("port", "22"))
    user = host.get("user", os.environ.get("USER", "root"))
    return HostCfg(hostname=hostname, port=port, user=user)


def _ssh_connect(alias: str, password: str | None) -> paramiko.SSHClient:
    host = _load_ssh_config(alias)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host.hostname, port=host.port, username=host.user, password=password, timeout=30)
    return ssh


def _ssh_exec(ssh: paramiko.SSHClient, cmd: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = ssh.exec_command("bash -lc " + repr(cmd))
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    return int(rc), out, err


def _sftp_get(ssh: paramiko.SSHClient, remote_path: str, local_path: Path) -> None:
    sftp = ssh.open_sftp()
    try:
        sftp.get(remote_path, str(local_path))
    finally:
        sftp.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Watch remote sweep progress, fetch results to local, and generate Chinese analysis markdown."
    )
    ap.add_argument("--ssh", default="autodl-gpu", help="SSH host alias in ~/.ssh/config")
    ap.add_argument("--remote_root", required=True, help="remote sweep root path")
    ap.add_argument("--expected_runs", type=int, required=True, help="expected number of runs")
    ap.add_argument("--poll_sec", type=int, default=60)
    ap.add_argument(
        "--local_out",
        default="",
        help="local output directory (default: artifacts/from_server/<ts>_fetched)",
    )
    ap.add_argument(
        "--password_env",
        default="TRISHIFT_SSH_PW",
        help="env var name containing SSH password (optional; will prompt if interactive)",
    )
    ap.add_argument("--no_analyze", action="store_true", help="skip local analysis generation")
    args = ap.parse_args()

    pw = os.environ.get(str(args.password_env), "")
    password: str | None = pw if pw else None
    if password is None and os.isatty(0):
        password = getpass.getpass(f"{args.ssh} password: ")
    if password is None:
        raise SystemExit(
            f"Missing password. Set env {args.password_env} or run from an interactive terminal."
        )

    remote_root = str(args.remote_root).rstrip("/")
    expected = int(args.expected_runs)

    if args.local_out.strip():
        local_out = Path(args.local_out)
    else:
        local_out = Path("artifacts") / "from_server" / f"{time.strftime('%Y%m%d_%H%M%S')}_fetched"
    local_out.mkdir(parents=True, exist_ok=True)

    ssh = _ssh_connect(str(args.ssh), password=password)
    try:
        # Poll until all metrics.csv exist.
        while True:
            cmd = (
                f"set -e; ROOT={remote_root!r}; "
                "done=$(find \"$ROOT\" -maxdepth 2 -name metrics.csv 2>/dev/null | wc -l); "
                "echo $done"
            )
            rc, out, err = _ssh_exec(ssh, cmd)
            if rc != 0:
                raise RuntimeError(err.strip() or out.strip())
            try:
                done = int(out.strip().splitlines()[-1])
            except Exception:
                done = 0
            pct = 100.0 * done / max(expected, 1)
            print(f"[remote] metrics.csv: {done}/{expected} ({pct:.1f}%)")
            if done >= expected:
                break
            time.sleep(int(args.poll_sec))

        # Tar the sweep root on remote to fetch as a single file.
        base = remote_root.rstrip("/").split("/")[-1]
        remote_tar = f"/tmp/{base}.tgz"
        rc, out, err = _ssh_exec(
            ssh,
            f"set -e; tar -czf {remote_tar!r} -C {str(Path(remote_root).parent)!r} {base!r}; echo {remote_tar!r}",
        )
        if rc != 0:
            raise RuntimeError(err.strip() or out.strip())

        local_tar = local_out / f"{base}.tgz"
        print(f"[fetch] downloading {remote_tar} -> {local_tar}")
        _sftp_get(ssh, remote_tar, local_tar)

    finally:
        ssh.close()

    # Extract locally.
    extract_root = local_out / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local_tar, "r:gz") as tf:
        tf.extractall(path=extract_root)

    local_sweep = extract_root / base
    print(f"[local] sweep_root={local_sweep}")

    if not args.no_analyze:
        import subprocess, sys

        cmd = [sys.executable, "scripts/analyze_sweep_cn.py", "--sweep_root", str(local_sweep), "--baseline_idx", "1"]
        subprocess.run(cmd, check=False)
        print(f"[local] wrote: {local_sweep / 'analysis_results_cn.md'}")
        print(f"[local] wrote: {local_sweep / 'analysis_full_metrics_comparison_cn.md'}")

    print(str(local_out))


if __name__ == "__main__":
    main()

