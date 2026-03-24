#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REGISTER_PATTERNS = [
    re.compile(r'LC_REGISTER_[A-Z0-9_]+(?:_EX)?\(\s*"([^"]+)"'),
    re.compile(r'REGISTER_OP(?:_FUNCS)?\(\s*"([^"]+)"'),
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(fh)
        elif suffix == ".json":
            data = json.load(fh)
        else:
            raise ValueError(f"Unsupported config format: {path}")
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping/object, got {type(data).__name__}")
    return data


def normalize_op_cfg(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"name": item}
    if isinstance(item, dict) and "name" in item:
        return dict(item)
    raise TypeError(f"Unsupported op config: {item!r}")


def cmake_target_name(src_root: Path, source_path: Path) -> str:
    rel_path = source_path.relative_to(src_root).as_posix()
    suffix = rel_path.replace("/", "__").replace(".", "_").replace("-", "_")
    return f"op__{suffix}"


def discover_targets(src_root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for path in sorted(src_root.rglob("*")):
        if path.suffix not in {".cu", ".cpp"} or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")

        names: set[str] = set()
        for pattern in REGISTER_PATTERNS:
            names.update(match.group(1) for match in pattern.finditer(text))

        if not names:
            continue

        target_name = cmake_target_name(src_root, path)
        for name in names:
            if name in mapping and mapping[name] != target_name:
                duplicates.setdefault(name, [mapping[name]]).append(target_name)
            mapping[name] = target_name

    if duplicates:
        parts = []
        for name, targets in sorted(duplicates.items()):
            unique_targets = sorted(set(targets))
            parts.append(f"{name}: {', '.join(unique_targets)}")
        raise RuntimeError("Duplicate operator registrations detected: " + "; ".join(parts))
    return mapping


def requested_op_names(config: dict[str, Any]) -> list[str]:
    requested: list[str] = []
    for raw_op_cfg in config.get("ops", []):
        op_cfg = normalize_op_cfg(raw_op_cfg)
        requested.append(str(op_cfg["name"]))
        for baseline in op_cfg.get("baselines", []) or []:
            requested.append(str(baseline))
    return requested


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve YAML-configured operator names to CMake targets.")
    parser.add_argument("--root-dir", required=True, help="Project root directory")
    parser.add_argument("--config", required=True, help="Benchmark config path")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    config_path = Path(args.config).resolve()
    src_root = root_dir / "src"

    config = load_config(config_path)
    name_to_target = discover_targets(src_root)
    requested_names = requested_op_names(config)

    missing = sorted({name for name in requested_names if name not in name_to_target})
    if missing:
        print("Missing build targets for operators: " + ", ".join(missing), file=sys.stderr)
        return 1

    targets = sorted({name_to_target[name] for name in requested_names})
    for target in targets:
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
