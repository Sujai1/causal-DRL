"""EC2 experiment runner — setup, deploy, run, status, and fetch.

Manages remote experiment execution on EC2 instances via SSH/rsync.
Seeds are reserved locally before dispatch to prevent collisions.

Usage:
    python scripts/ec2.py setup  --host 3.145.216.26
    python scripts/ec2.py deploy --host 3.145.216.26
    python scripts/ec2.py run    --host 3.145.216.26 --seeds 143-172 --parallel 8
    python scripts/ec2.py run    --host 3.145.216.26 --seeds 143-172 --parallel 8 \
        -- --k_targets 16 32 48 --baselines custom_dqn_noreg_ln heuristic_noop
    python scripts/ec2.py status --host 3.145.216.26
    python scripts/ec2.py fetch  --host 3.145.216.26
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PEM_KEY = Path.home() / "Desktop" / "Sujai.pem"
REMOTE_USER = "ec2-user"
REMOTE_DIR = "/home/ec2-user/causal-DRL"
PYTHON = "python3.10"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SSH_OPTS = [
    "-i", str(PEM_KEY),
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=10",
]

RSYNC_EXCLUDES = [
    ".venv", "outputs", "__pycache__", ".git", "*.egg-info",
    ".mypy_cache", ".pytest_cache", "*.pyc",
]

PIP_PACKAGES = [
    "pyRDDLGym", "rddlrepository", "pyRDDLGym-symbolic",
    "stable-baselines3", "gymnasium",
]

OLD_FILES_TO_CLEAN = [
    "NPEET/", "dodiscover/", "Python-3.10.12.tgz", "aws_test_hello.py",
]

# Default flags passed to run_all_baselines.py (excluding --seed, --ba_m, --num_machines)
DEFAULT_BASELINE_FLAGS = [
    "--topology", "barabasi_albert",
    "--timesteps", "200000",
    "--horizon", "100",
    "--gamma", "0.99",
    "--baselines",
    "custom_dqn_noreg_ln",
    "custom_dqn_gradient_balanced",
    "heuristic_noop",
    "heuristic_random_reboot",
    "heuristic_random_down",
    "heuristic_highest_degree",
    "heuristic_most_down_neighbors",
    "heuristic_myopic_greedy",
    "--k_targets", "16", "32", "48", "64", "80", "96", "112",
    "--eps_decay_frac", "0.2",
    "--reboot_prob", "0.005",
    "--reboot_penalty", "1.75",
    "--hidden_dim", "128",
]


# ---------------------------------------------------------------------------
# SSH / rsync helpers
# ---------------------------------------------------------------------------

def _remote(host: str) -> str:
    return f"{REMOTE_USER}@{host}"


def _ssh(host: str, cmd: str, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command on the remote host via SSH."""
    full_cmd = ["ssh", *SSH_OPTS, _remote(host), cmd]
    if capture:
        return subprocess.run(full_cmd, capture_output=True, text=True)
    return subprocess.run(full_cmd)


def _rsync(src: str, dst: str, excludes: list[str] | None = None) -> subprocess.CompletedProcess:
    """Rsync wrapper with PEM key and standard options."""
    cmd = [
        "rsync", "-avz",
        "-e", f"ssh {' '.join(SSH_OPTS)}",
    ]
    for exc in (excludes or []):
        cmd.extend(["--exclude", exc])
    cmd.extend([src, dst])
    return subprocess.run(cmd)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_setup(args: argparse.Namespace) -> None:
    """One-time instance setup: install tmux + pip packages, clean old files."""
    host = args.host
    print("=== Installing tmux ===")
    _ssh(host, "sudo yum install -y tmux")

    print("\n=== Installing system dependencies ===")
    _ssh(host, "sudo yum install -y graphviz-devel")

    print("\n=== Installing pip packages ===")
    pkgs = " ".join(PIP_PACKAGES)
    _ssh(host, f"{PYTHON} -m pip install --user {pkgs} pygraphviz")

    print("\n=== Cleaning old files ===")
    targets = " ".join(f"~/{f}" for f in OLD_FILES_TO_CLEAN)
    _ssh(host, f"rm -rf {targets}")

    print("\n=== Installing project package ===")
    _ssh(host, f"cd {REMOTE_DIR} && {PYTHON} -m pip install --user -e .")

    print("\n=== Disk usage ===")
    _ssh(host, "df -h /")

    print("\nSetup complete.")


def cmd_deploy(args: argparse.Namespace) -> None:
    """Sync project code to the remote instance."""
    host = args.host
    src = str(PROJECT_ROOT) + "/"
    dst = f"{_remote(host)}:{REMOTE_DIR}/"

    print(f"Deploying {src} -> {dst}")
    result = _rsync(src, dst, excludes=RSYNC_EXCLUDES)
    if result.returncode != 0:
        print("ERROR: rsync failed")
        sys.exit(1)

    print("\n=== Reinstalling project package ===")
    _ssh(host, f"cd {REMOTE_DIR} && {PYTHON} -m pip install --user -e .")

    print("\n=== Remote file listing ===")
    _ssh(host, f"ls -la {REMOTE_DIR}/")

    print("\nDeploy complete.")


def cmd_run(args: argparse.Namespace) -> None:
    """Launch experiments in a tmux session on the remote instance."""
    host = args.host
    parallel = args.parallel

    if args.seeds:
        seeds = _parse_seed_range(args.seeds)
    else:
        print("ERROR: --seeds is required")
        sys.exit(1)

    if not seeds:
        print("No seeds to run.")
        return

    # Reserve seeds locally before dispatching
    from select_seeds import register_seeds
    register_seeds(seeds)
    print(f"Reserved {len(seeds)} seeds in local tracker: {seeds}")

    # Use passthrough args if provided, otherwise defaults
    ba_m = args.ba_m
    num_machines = args.num_machines
    if args.extra_args:
        baseline_flags = " ".join(args.extra_args)
    else:
        baseline_flags = " ".join(DEFAULT_BASELINE_FLAGS)

    script_body = _build_parallel_script(seeds, parallel, ba_m, num_machines, baseline_flags)

    # Upload the script and launch in tmux
    script_name = "run_batch.sh"
    remote_script = f"{REMOTE_DIR}/{script_name}"

    # Write script via heredoc over SSH
    escaped_body = script_body.replace("'", "'\\''")
    _ssh(host, f"cat > {remote_script} << 'SCRIPT_EOF'\n{script_body}\nSCRIPT_EOF")
    _ssh(host, f"chmod +x {remote_script}")

    # Kill existing tmux session if any, then launch new one
    _ssh(host, "tmux kill-session -t exp 2>/dev/null; true")
    _ssh(host, f"tmux new-session -d -s exp 'cd {REMOTE_DIR} && bash {script_name} 2>&1 | tee run_batch.log'")

    print(f"\nLaunched {len(seeds)} seeds ({parallel} parallel) in tmux session 'exp'")
    print(f"Monitor with: python scripts/ec2.py status --host {host}")


def cmd_status(args: argparse.Namespace) -> None:
    """Check experiment progress on the remote instance."""
    host = args.host

    # Check tmux session
    result = _ssh(host, "tmux has-session -t exp 2>/dev/null && echo RUNNING || echo STOPPED", capture=True)
    session_status = result.stdout.strip()
    print(f"tmux session 'exp': {session_status}")

    # Count completed experiments
    result = _ssh(
        host,
        f"find {REMOTE_DIR}/outputs -name 'summary.json' -path '*comparison_m10*' 2>/dev/null | wc -l",
        capture=True,
    )
    completed = result.stdout.strip()
    print(f"Completed experiments (summary.json): {completed}")

    # Show tail of log
    print("\n=== Last 30 lines of run_batch.log ===")
    _ssh(host, f"tail -30 {REMOTE_DIR}/run_batch.log 2>/dev/null || echo '(no log file)'")


def cmd_fetch(args: argparse.Namespace) -> None:
    """Download experiment results from the remote instance."""
    host = args.host
    local_outputs = PROJECT_ROOT / "outputs"
    local_outputs.mkdir(exist_ok=True)

    src = f"{_remote(host)}:{REMOTE_DIR}/outputs/"
    dst = str(local_outputs) + "/"

    # Only fetch comparison result directories
    print(f"Fetching results: {src} -> {dst}")
    result = subprocess.run([
        "rsync", "-avz",
        "-e", f"ssh {' '.join(SSH_OPTS)}",
        "--include", "*/",
        "--include", "*comparison_m10*/**",
        "--exclude", "*",
        src, dst,
    ])
    if result.returncode != 0:
        print("ERROR: rsync fetch failed")
        sys.exit(1)

    print("\nFetch complete.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_seed_range(seed_str: str) -> list[int]:
    """Parse seed specification: '3-24' or '3,5,7' or '3-5,10-12'."""
    seeds = []
    for part in seed_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))



def _build_parallel_script(
    seeds: list[int],
    parallel: int,
    ba_m: int,
    num_machines: int,
    baseline_flags: str,
) -> str:
    """Generate a bash script that runs experiments in parallel batches."""
    return f"""#!/bin/bash
set -e
cd {REMOTE_DIR}

# Limit each process to 1 thread to avoid thrashing with many parallel jobs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_NUM_THREADS=1

mkdir -p {REMOTE_DIR}/outputs

seeds=({" ".join(str(s) for s in seeds)})
parallel={parallel}
total=${{#seeds[@]}}
completed=0

echo "Starting $total experiments ({parallel} parallel)"
echo "Seeds: ${{seeds[@]}}"
echo "Start time: $(date)"
echo ""

for ((i=0; i<${{#seeds[@]}}; i+=parallel)); do
    batch_pids=()
    for ((j=i; j<i+parallel && j<${{#seeds[@]}}; j++)); do
        seed=${{seeds[j]}}
        echo "[$(date +%H:%M:%S)] Starting seed $seed ($(( j+1 ))/$total)"
        {PYTHON} scripts/run_all_baselines.py \\
            --seed $seed \\
            --ba_m {ba_m} \\
            --num_machines {num_machines} \\
            {baseline_flags} \\
            > {REMOTE_DIR}/outputs/seed_${{seed}}.log 2>&1 &
        batch_pids+=($!)
    done

    # Wait for batch to finish
    for pid in "${{batch_pids[@]}}"; do
        wait $pid || echo "WARNING: process $pid exited with error"
    done

    completed=$((completed + ${{#batch_pids[@]}}))
    echo "[$(date +%H:%M:%S)] Batch done. Progress: $completed/$total"
    echo ""
done

echo "All $total experiments complete at $(date)"
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EC2 experiment runner for causal-DRL"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared host argument
    host_kwargs = dict(type=str, help="EC2 instance public IP")

    # setup
    p = sub.add_parser("setup", help="One-time instance setup")
    p.add_argument("--host", required=True, **host_kwargs)

    # deploy
    p = sub.add_parser("deploy", help="Sync project code to instance")
    p.add_argument("--host", required=True, **host_kwargs)

    # run
    p = sub.add_parser("run", help="Launch experiments in tmux")
    p.add_argument("--host", required=True, **host_kwargs)
    p.add_argument("--seeds", type=str, required=True,
                   help="Seed range, e.g. '143-172' or '3,5,7,10-12'")
    p.add_argument("--parallel", type=int, default=4,
                   help="Number of concurrent experiments (default: 4)")
    p.add_argument("--ba_m", type=int, default=2,
                   help="BA attachment parameter (default: 2)")
    p.add_argument("--num_machines", type=int, default=10,
                   help="Number of machines in SysAdmin (default: 10)")
    p.add_argument("extra_args", nargs="*",
                   help="Extra args for run_all_baselines.py (after '--')")

    # status
    p = sub.add_parser("status", help="Check experiment progress")
    p.add_argument("--host", required=True, **host_kwargs)

    # fetch
    p = sub.add_parser("fetch", help="Download results from instance")
    p.add_argument("--host", required=True, **host_kwargs)

    return parser


def main() -> None:
    # Ensure select_seeds is importable from scripts/
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "setup": cmd_setup,
        "deploy": cmd_deploy,
        "run": cmd_run,
        "status": cmd_status,
        "fetch": cmd_fetch,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
