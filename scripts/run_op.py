#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def normalize_param_key(key: str) -> str:
    return "".join(ch.upper() if ch.isalnum() else "_" for ch in key)


def parse_kv(items: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        values[key] = value
    return values


def discover_binaries(bin_dir: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for candidate in sorted(bin_dir.iterdir()):
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        proc = subprocess.run([str(candidate), "--describe"], capture_output=True, text=True)
        if proc.returncode != 0 or not proc.stdout.strip():
            continue
        try:
            payload = json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            continue
        op_name = payload.get("op_name")
        if op_name:
            discovered[op_name] = candidate
    return discovered


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one CUDA op directly or under ncu/debug tools.")
    parser.add_argument("--build-dir", required=True, help="CMake build directory")
    parser.add_argument("--op", required=True, help="Operator name")
    parser.add_argument("--mode", default="both", choices=["correctness", "performance", "both"], help="Runner mode")
    parser.add_argument("--case-name", default="manual", help="Case name for terminal/report output")
    parser.add_argument("--param", action="append", default=[], help="Shape/tuning parameter, format KEY=VALUE")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=None, help="Measurement iterations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--tool", default="direct", choices=["direct", "ncu", "compute-sanitizer", "cuda-gdb"], help="Launch tool")
    parser.add_argument("--tool-args", default="", help="Extra args passed to the selected tool")
    parser.add_argument("--ncu-report", default="", help="When --tool ncu, write report to this path stem (without .ncu-rep)")
    parser.add_argument("--json", action="store_true", help="Forward --json to the runner")
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    bin_dir = build_dir / "bin"
    if not bin_dir.exists():
        raise FileNotFoundError(f"Binary directory does not exist: {bin_dir}")

    discovered = discover_binaries(bin_dir)
    if args.op not in discovered:
        available = ", ".join(sorted(discovered.keys()))
        raise KeyError(f"Operator '{args.op}' not found. Available operators: {available}")

    binary = discovered[args.op]
    env = os.environ.copy()
    env["CUDA_OP_CASE_NAME"] = args.case_name
    if args.warmup is not None:
        env["CUDA_OP_WARMUP"] = str(args.warmup)
    if args.iters is not None:
        env["CUDA_OP_ITERS"] = str(args.iters)
    if args.seed is not None:
        env["CUDA_OP_SEED"] = str(args.seed)

    for key, value in parse_kv(args.param).items():
        env[f"CUDA_OP_PARAM_{normalize_param_key(key)}"] = value

    runner_cmd = [str(binary), "--mode", args.mode]
    if args.json:
        runner_cmd.append("--json")

    tool_args = shlex.split(args.tool_args)
    if args.tool == "direct":
        cmd = runner_cmd
    elif args.tool == "ncu":
        cmd = ["ncu"]
        if args.ncu_report:
            report_path = Path(args.ncu_report).expanduser()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            cmd.extend(["-f", "-o", str(report_path)])
        cmd.extend(tool_args)
        cmd.extend(runner_cmd)
    elif args.tool == "compute-sanitizer":
        cmd = ["compute-sanitizer", *tool_args, *runner_cmd]
    else:
        cmd = ["cuda-gdb", *tool_args, "--args", *runner_cmd]

    print(" ".join(shlex.quote(part) for part in cmd), flush=True)
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
