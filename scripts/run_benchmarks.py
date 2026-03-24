#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import yaml

ANSI_RESET = "\033[0m"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_CYAN = "\033[36m"
ANSI_YELLOW = "\033[33m"


def normalize_param_key(key: str) -> str:
    return "".join(ch.upper() if ch.isalnum() else "_" for ch in key)


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(fh)
        elif suffix == ".json":
            data = json.load(fh)
        else:
            raise ValueError(f"Unsupported config format: {path}. Use .yaml/.yml or .json.")

    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping/object, got {type(data).__name__}")
    return data


def discover_binaries(bin_dir: Path) -> dict[str, dict[str, Any]]:
    discovered: dict[str, dict[str, Any]] = {}
    for candidate in sorted(bin_dir.iterdir()):
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        proc = subprocess.run([str(candidate), "--describe"], capture_output=True, text=True)
        if proc.returncode != 0:
            continue
        stdout = proc.stdout.strip()
        if not stdout:
            continue
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            continue
        op_name = payload.get("op_name")
        if not op_name:
            continue
        discovered[op_name] = {"path": candidate, "describe": payload}
    return discovered


def default_param_map(meta: dict[str, Any]) -> dict[str, int]:
    defaults: dict[str, int] = {}
    for item in meta.get("params", []):
        defaults[str(item["name"])] = int(item["default_value"])
    return defaults


def edge_values(edge_cfg: dict[str, Any]) -> list[int]:
    start = int(edge_cfg["from"])
    stop = int(edge_cfg["to"])
    stride = int(edge_cfg.get("stride", 1))
    if stride <= 0:
        raise ValueError("edge.stride must be > 0")
    if stop < start:
        raise ValueError("edge.to must be >= edge.from")
    return list(range(start, stop + 1, stride))


def case_name_from_params(params: dict[str, Any]) -> str:
    return "_".join(f"{str(key).lower()}{value}" for key, value in params.items())


def normalize_op_cfg(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        return {"name": item}
    if isinstance(item, dict) and "name" in item:
        return dict(item)
    raise TypeError(f"Unsupported op config: {item!r}")


def dedupe_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, Any], ...]] = set()
    deduped: list[dict[str, Any]] = []
    for case in cases:
        key = tuple(sorted(case["params"].items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def build_cases(meta: dict[str, Any], global_cfg: dict[str, Any], op_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    base = default_param_map(meta)
    base.update({str(k): int(v) for k, v in global_cfg.get("params", {}).items()})
    base.update({str(k): int(v) for k, v in op_cfg.get("params", {}).items()})

    cases: list[dict[str, Any]] = []
    edge_cfg = op_cfg.get("edge")
    if edge_cfg:
        axes = list(meta.get("edge_axes", []))
        if not axes:
            raise ValueError(f"Operator {meta.get('op_name', meta.get('name', 'unknown'))} does not expose edge_axes")
        for edge in edge_values(edge_cfg):
            params = dict(base)
            for axis in axes:
                params[str(axis)] = edge
            cases.append({"name": f"edge_{edge}", "params": params, "plot_x": edge, "plot_label": str(edge)})

    for index, shape in enumerate(op_cfg.get("shapes", [])):
        params = dict(base)
        if not isinstance(shape, dict):
            raise TypeError(f"shape entry must be an object, got {shape!r}")
        params.update({str(k): int(v) for k, v in shape.items() if k != "name"})
        cases.append(
            {
                "name": shape.get("name", case_name_from_params(params)),
                "params": params,
                "plot_x": float(index + 1),
                "plot_label": shape.get("name", f"shape{index + 1}"),
            }
        )

    if not cases:
        cases.append({"name": "default", "params": dict(base), "plot_x": 1.0, "plot_label": "default"})

    return dedupe_cases(cases)


def run_single_stage(binary: Path, stage: str, case_cfg: dict[str, Any], warmup: int | None, iters: int | None, seed: int | None,
                     cache: dict[tuple[Any, ...], dict[str, Any]]) -> dict[str, Any]:
    params = case_cfg["params"]
    cache_key = (str(binary), stage, tuple(sorted(params.items())), warmup, iters, seed)
    if cache_key in cache:
        return dict(cache[cache_key])

    env = os.environ.copy()
    env["CUDA_OP_STAGE"] = stage
    env["CUDA_OP_CASE_NAME"] = str(case_cfg["name"])
    if warmup is not None:
        env["CUDA_OP_WARMUP"] = str(warmup)
    if iters is not None:
        env["CUDA_OP_ITERS"] = str(iters)
    if seed is not None:
        env["CUDA_OP_SEED"] = str(seed)
    for key, value in params.items():
        env[f"CUDA_OP_PARAM_{normalize_param_key(str(key))}"] = str(value)

    proc = subprocess.run([str(binary), "--mode", stage, "--json"], capture_output=True, text=True, env=env)
    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError(f"{binary.name} returned no JSON payload.\nstderr:\n{proc.stderr}")

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON from {binary.name}.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}") from exc

    payload["_returncode"] = proc.returncode
    payload["_stderr"] = proc.stderr
    payload["_stdout"] = proc.stdout
    cache[cache_key] = payload
    return dict(payload)


def metric_lookup(perf: dict[str, Any] | None, metric_name: str) -> tuple[Any, Any]:
    if not perf:
        return "", ""
    for item in perf.get("extra_metrics", []):
        if item.get("name") == metric_name:
            return item.get("value", ""), item.get("unit", "")
    return "", ""


def format_float(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.{digits}f}"


def ac_tag(correctness_ok: bool | None) -> str:
    if correctness_ok is None:
        return f"{ANSI_YELLOW}[SKIP]{ANSI_RESET}"
    return f"{ANSI_GREEN}[AC]{ANSI_RESET}" if correctness_ok else f"{ANSI_RED}[WA]{ANSI_RESET}"


def print_live_result(group_name: str, impl_name: str, case_cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    correctness = payload.get("correctness") or None
    performance = payload.get("performance") or {}
    correctness_ok = None if correctness is None else bool(correctness.get("ok"))
    bw_value, bw_unit = metric_lookup(performance, "effective_bandwidth")

    parts = [
        ac_tag(correctness_ok),
        group_name,
        f"impl={impl_name}",
        f"case={case_cfg['name']}",
    ]
    if correctness is not None:
        parts.append(f"{correctness.get('metric_name')}={format_float(correctness.get('metric'), 6)}")
    if performance:
        parts.append(f"{format_float(performance.get('ms'))} ms")
        parts.append(f"{performance.get('unit_name')}={format_float(performance.get('unit_value'))}")
    if bw_value != "":
        parts.append(f"BW={format_float(bw_value)} {bw_unit}")
    print("  ".join(parts), flush=True)


def to_measurement_row(group_name: str, implementation_name: str, case_cfg: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    correctness = payload.get("correctness") or {}
    performance = payload.get("performance") or {}
    effective_bandwidth, effective_bandwidth_unit = metric_lookup(performance, "effective_bandwidth")
    elements_per_second, elements_per_second_unit = metric_lookup(performance, "elements_per_second")

    return {
        "group_name": group_name,
        "implementation_name": implementation_name,
        "case_name": case_cfg["name"],
        "plot_x": case_cfg["plot_x"],
        "plot_label": case_cfg["plot_label"],
        "params_json": json.dumps(case_cfg["params"], ensure_ascii=False, sort_keys=True),
        "latency_ms": performance.get("ms", ""),
        "cpu_ms": performance.get("cpu_ms", ""),
        "primary_metric_name": performance.get("unit_name", ""),
        "primary_metric_value": performance.get("unit_value", ""),
        "effective_bandwidth_value": effective_bandwidth,
        "effective_bandwidth_unit": effective_bandwidth_unit,
        "elements_per_second_value": elements_per_second,
        "elements_per_second_unit": elements_per_second_unit,
        "input_size_bytes": performance.get("input_size", 0),
        "output_size_bytes": performance.get("output_size", 0),
        "input_format_json": json.dumps(performance.get("input_format", []), ensure_ascii=False),
        "output_format_json": json.dumps(performance.get("output_format", []), ensure_ascii=False),
        "correctness_ok": correctness.get("ok", ""),
        "correctness_metric_name": correctness.get("metric_name", ""),
        "correctness_metric_value": correctness.get("metric", ""),
        "correctness_threshold": correctness.get("threshold", ""),
        "correctness_note": correctness.get("note", ""),
        "performance_note": performance.get("note", ""),
        "return_code": payload.get("_returncode", 0),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_number(value: float) -> str:
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:.0f}"
    if abs_value >= 100:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def render_svg_line_chart(title: str, x_label: str, y_label: str, series: dict[str, list[tuple[float, float, str]]], output_path: Path) -> None:
    points = [point for values in series.values() for point in values]
    if not points:
        return

    width = 980
    height = 620
    margin_left = 90
    margin_right = 220
    margin_top = 70
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_x = [x for x, _, _ in points]
    all_y = [y for _, y, _ in points]
    min_x = min(all_x)
    max_x = max(all_x)
    min_y = 0.0 if min(all_y) >= 0 else min(all_y)
    max_y = max(all_y)

    if min_x == max_x:
        min_x -= 1.0
        max_x += 1.0
    if min_y == max_y:
        max_y = min_y + 1.0

    def map_x(value: float) -> float:
        return margin_left + (value - min_x) / (max_x - min_x) * plot_width

    def map_y(value: float) -> float:
        return margin_top + plot_height - (value - min_y) / (max_y - min_y) * plot_height

    palette = ["#15803d", "#1d4ed8", "#b45309", "#b91c1c", "#0f766e", "#7c3aed", "#475569", "#a16207"]
    x_ticks = 5
    y_ticks = 5
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcfd"/>',
        f'<text x="{margin_left}" y="36" font-size="24" font-family="monospace" fill="#111827">{escape(title)}</text>',
    ]

    for i in range(x_ticks + 1):
        tick_value = min_x + (max_x - min_x) * i / x_ticks
        x = map_x(tick_value)
        svg.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 28}" text-anchor="middle" font-size="12" font-family="monospace" fill="#374151">{escape(format_number(tick_value))}</text>'
        )

    for i in range(y_ticks + 1):
        tick_value = min_y + (max_y - min_y) * i / y_ticks
        y = map_y(tick_value)
        svg.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" font-family="monospace" fill="#374151">{escape(format_number(tick_value))}</text>'
        )

    svg.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="2"/>'
    )
    svg.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="2"/>')

    legend_y = margin_top + 10
    for index, (series_name, values) in enumerate(sorted(series.items())):
        color = palette[index % len(palette)]
        ordered = sorted(values, key=lambda item: item[0])
        polyline = " ".join(f"{map_x(x):.2f},{map_y(y):.2f}" for x, y, _ in ordered)
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{polyline}"/>')
        for x, y, case_label in ordered:
            svg.append(f'<circle cx="{map_x(x):.2f}" cy="{map_y(y):.2f}" r="4.5" fill="{color}" />')
            svg.append(
                f'<text x="{map_x(x):.2f}" y="{map_y(y) - 10:.2f}" text-anchor="middle" font-size="10" font-family="monospace" fill="#4b5563">{escape(case_label)}</text>'
            )
        legend_item_y = legend_y + index * 24
        legend_x = margin_left + plot_width + 24
        svg.append(f'<line x1="{legend_x}" y1="{legend_item_y}" x2="{legend_x + 28}" y2="{legend_item_y}" stroke="{color}" stroke-width="3"/>')
        svg.append(
            f'<text x="{legend_x + 38}" y="{legend_item_y + 4}" font-size="12" font-family="monospace" fill="#111827">{escape(series_name)}</text>'
        )

    svg.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 22}" text-anchor="middle" font-size="14" font-family="monospace" fill="#111827">{escape(x_label)}</text>'
    )
    svg.append(
        f'<text x="24" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 24 {margin_top + plot_height / 2:.2f})" text-anchor="middle" font-size="14" font-family="monospace" fill="#111827">{escape(y_label)}</text>'
    )
    svg.append("</svg>")

    output_path.write_text("\n".join(svg), encoding="utf-8")


def write_summary(report_path: Path, measurement_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> None:
    grouped_measurements: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in measurement_rows:
        grouped_measurements[(row["group_name"], row["case_name"])].append(row)

    grouped_comparisons: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        grouped_comparisons[(row["group_name"], row["case_name"])].append(row)

    lines: list[str] = []
    for (group_name, case_name), rows in sorted(grouped_measurements.items()):
        lines.append(f"[{group_name}] case={case_name}")
        for row in sorted(rows, key=lambda item: item["implementation_name"]):
            correctness = row["correctness_ok"]
            correctness_text = "n/a" if correctness == "" else ("AC" if correctness else "WA")
            lines.append(
                f"  - {row['implementation_name']}: latency={row['latency_ms']} ms, {row['primary_metric_name']}={row['primary_metric_value']}, correctness={correctness_text}"
            )
        for row in grouped_comparisons.get((group_name, case_name), []):
            lines.append(f"    vs {row['baseline_name']}: speedup={row['speedup_vs_baseline']:.4f}x")
        lines.append("")

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CUDA operator benchmarks from a simple YAML config.")
    parser.add_argument("--build-dir", required=True, help="CMake build directory")
    parser.add_argument("--config", required=True, help="Benchmark config file (.yaml/.yml preferred)")
    parser.add_argument("--report-dir", default="", help="Optional report directory override")
    args = parser.parse_args()

    build_dir = Path(args.build_dir).resolve()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    global_cfg = config.get("global", {})
    report_dir_value = args.report_dir or global_cfg.get("report_dir") or config.get("report_dir") or "reports/latest"
    report_dir = Path(report_dir_value)
    if not report_dir.is_absolute():
        report_dir = (config_path.parent.parent / report_dir).resolve()

    if report_dir.exists():
        shutil.rmtree(report_dir)
    ensure_dir(report_dir)
    raw_dir = ensure_dir(report_dir / "raw")
    plot_dir = ensure_dir(report_dir / "plots")

    bin_dir = build_dir / "bin"
    if not bin_dir.exists():
        raise FileNotFoundError(f"Binary directory does not exist: {bin_dir}")

    discovered = discover_binaries(bin_dir)
    if not discovered:
        raise RuntimeError(f"No operator binaries were discovered under {bin_dir}")

    warmup = global_cfg.get("warmup")
    iters = global_cfg.get("iters")
    seed = global_cfg.get("seed")
    check_correctness = bool(global_cfg.get("check_correctness", True))
    check_baselines_correctness = bool(global_cfg.get("check_baselines_correctness", False))

    measurement_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    measurement_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for raw_op_cfg in config.get("ops", []):
        op_cfg = normalize_op_cfg(raw_op_cfg)
        group_name = op_cfg["name"]
        if group_name not in discovered:
            available = ", ".join(sorted(discovered.keys()))
            raise KeyError(f"Operator '{group_name}' not found. Available operators: {available}")

        meta = discovered[group_name]["describe"]
        cases = build_cases(meta, global_cfg, op_cfg)
        baseline_names = list(op_cfg.get("baselines", []))
        baseline_names = [name for name in baseline_names if name and name != group_name]

        print(f"\n{group_name}  kind={meta.get('kind')}  cases={len(cases)}", flush=True)

        for case_cfg in cases:
            rows_by_impl: dict[str, dict[str, Any]] = {}
            impl_order = [group_name] + baseline_names
            for impl_name in impl_order:
                if impl_name not in discovered:
                    raise KeyError(f"Implementation '{impl_name}' not found in build output.")

                binary = discovered[impl_name]["path"]
                correctness_payload = None
                should_check = check_correctness and (impl_name == group_name or check_baselines_correctness)
                if should_check:
                    correctness_payload = run_single_stage(binary, "correctness", case_cfg, warmup, iters, seed, measurement_cache)
                    if correctness_payload.get("_returncode", 0) != 0:
                        failures.append(f"{impl_name} correctness failed for case {case_cfg['name']}")

                performance_payload = run_single_stage(binary, "performance", case_cfg, warmup, iters, seed, measurement_cache)
                if performance_payload.get("_returncode", 0) != 0:
                    failures.append(f"{impl_name} performance failed for case {case_cfg['name']}")

                merged_payload = {
                    "op_name": performance_payload.get("op_name", impl_name),
                    "case_name": performance_payload.get("case_name", case_cfg["name"]),
                    "mode": "both" if correctness_payload is not None else "performance",
                    "correctness": None if correctness_payload is None else correctness_payload.get("correctness"),
                    "performance": performance_payload.get("performance"),
                    "_returncode": max(
                        0 if correctness_payload is None else correctness_payload.get("_returncode", 0),
                        performance_payload.get("_returncode", 0),
                    ),
                }

                print_live_result(group_name, impl_name, case_cfg, merged_payload)
                rows_by_impl[impl_name] = to_measurement_row(group_name, impl_name, case_cfg, merged_payload)
                measurement_rows.append(rows_by_impl[impl_name])

                raw_file = raw_dir / f"{sanitize_name(group_name)}__{sanitize_name(case_cfg['name'])}__{sanitize_name(impl_name)}.json"
                raw_file.write_text(json.dumps(merged_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            target_row = rows_by_impl[group_name]
            for baseline_name in baseline_names:
                baseline_row = rows_by_impl[baseline_name]
                target_ms = float(target_row["latency_ms"])
                baseline_ms = float(baseline_row["latency_ms"])
                speedup = baseline_ms / target_ms if target_ms else 0.0
                comparison_rows.append(
                    {
                        "group_name": group_name,
                        "case_name": case_cfg["name"],
                        "target_name": group_name,
                        "baseline_name": baseline_name,
                        "target_latency_ms": target_ms,
                        "baseline_latency_ms": baseline_ms,
                        "speedup_vs_baseline": speedup,
                    }
                )
                print(f"    {ANSI_CYAN}[CMP]{ANSI_RESET} {group_name} vs {baseline_name}  case={case_cfg['name']}  speedup={speedup:.4f}x", flush=True)

    write_csv(report_dir / "benchmark_measurements.csv", measurement_rows)
    write_csv(report_dir / "benchmark_comparisons.csv", comparison_rows)
    write_summary(report_dir / "benchmark_summary.txt", measurement_rows, comparison_rows)

    grouped_for_plots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in measurement_rows:
        grouped_for_plots[row["group_name"]].append(row)

    for group_name, rows in grouped_for_plots.items():
        if len(rows) < 2:
            continue
        latency_series: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        throughput_series: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        throughput_name = ""
        for row in rows:
            x_value = float(row["plot_x"])
            label = row["plot_label"]
            latency_series[row["implementation_name"]].append((x_value, float(row["latency_ms"]), label))
            if row["primary_metric_name"] and row["primary_metric_value"] != "":
                throughput_name = row["primary_metric_name"]
                throughput_series[row["implementation_name"]].append((x_value, float(row["primary_metric_value"]), label))

        render_svg_line_chart(
            title=f"{group_name} latency",
            x_label="edge / case index",
            y_label="Latency (ms)",
            series=latency_series,
            output_path=plot_dir / f"{sanitize_name(group_name)}__latency.svg",
        )
        if throughput_series and throughput_name:
            render_svg_line_chart(
                title=f"{group_name} throughput",
                x_label="edge / case index",
                y_label=throughput_name,
                series=throughput_series,
                output_path=plot_dir / f"{sanitize_name(group_name)}__throughput.svg",
            )

    print(f"\nReports written to: {report_dir}", flush=True)
    if failures:
        print("\nFailures detected:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
