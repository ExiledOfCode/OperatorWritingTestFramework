#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_ncu}"
REPORT_DIR="${REPORT_DIR:-${ROOT_DIR}/reports/ncu}"
OP="${OP:-frame_gemm_demo8_128x128x8_tile_thread4val}"
M="${M:-2048}"
N="${N:-2048}"
K="${K:-2048}"
WARMUP="${WARMUP:-0}"
ITERS="${ITERS:-1}"
NCU_SET="${NCU_SET:-full}"
CASE_NAME="${CASE_NAME:-ncu_m${M}_n${N}_k${K}}"
REPORT_STEM="${REPORT_DIR}/${OP}_${CASE_NAME}"

mkdir -p "${REPORT_DIR}"

python3 "${ROOT_DIR}/scripts/run_op.py" \
  --build-dir "${BUILD_DIR}" \
  --op "${OP}" \
  --mode performance \
  --case-name "${CASE_NAME}" \
  --param "M=${M}" \
  --param "N=${N}" \
  --param "K=${K}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --tool ncu \
  --ncu-report "${REPORT_STEM}" \
  --tool-args "--set ${NCU_SET}"
