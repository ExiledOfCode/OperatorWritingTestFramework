#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build_ncu}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
CONFIG_PATH="${1:-${ROOT_DIR}/configs/benchmark_plan.yaml}"
BUILD_TARGET_MODE="${BUILD_TARGET_MODE:-selected}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Ncu

if [[ "${BUILD_TARGET_MODE}" == "all" ]]; then
  cmake --build "${BUILD_DIR}" -j"${BUILD_JOBS}"
else
  mapfile -t BUILD_TARGETS < <(
    python3 "${ROOT_DIR}/scripts/select_build_targets.py" \
      --root-dir "${ROOT_DIR}" \
      --config "${CONFIG_PATH}"
  )
  if [[ "${#BUILD_TARGETS[@]}" -eq 0 ]]; then
    cmake --build "${BUILD_DIR}" -j"${BUILD_JOBS}"
  else
    cmake --build "${BUILD_DIR}" --target "${BUILD_TARGETS[@]}" -j"${BUILD_JOBS}"
  fi
fi
