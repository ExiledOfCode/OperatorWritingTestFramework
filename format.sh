#!/bin/bash
set -euo pipefail

# 需要递归格式化的目录（可按需改）
FORMAT_DIRS=("src" "tests" "tool" "include")

# 允许用环境变量覆盖（例如：FORMAT_DIRS="src include" ./format.sh）
if [ "${FORMAT_DIRS_OVERRIDE:-}" != "" ]; then
  # shellcheck disable=SC2206
  FORMAT_DIRS=($FORMAT_DIRS_OVERRIDE)
fi

# 支持的后缀
EXTS=( -name "*.cpp" -o -name "*.hpp" -o -name "*.cc" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" )

# 如果传入了参数 → 只格式化指定文件
if [ "$#" -gt 0 ]; then
  for file in "$@"; do
    if [ -f "$file" ]; then
      clang-format -i -style=file "$file"
    #   echo "Formatted: $file"
    else
      echo "Warning: $file not found, skipped."
    fi
  done
#   echo "Done."
  exit 0
fi

# 没有参数 → 只递归格式化指定目录
# 只收集存在的目录，避免 find 报错
DIRS_EXIST=()
for d in "${FORMAT_DIRS[@]}"; do
  if [ -d "$d" ]; then
    DIRS_EXIST+=("$d")
  else
    echo "Note: directory '$d' not found, skipped."
  fi
done

if [ "${#DIRS_EXIST[@]}" -eq 0 ]; then
#   echo "No target directories exist. Nothing to format."
  exit 0
fi

find "${DIRS_EXIST[@]}" -type f \( "${EXTS[@]}" \) -print0 \
  | xargs -0 clang-format -i -style=file

# echo "Formatted C++/CUDA files under: ${DIRS_EXIST[*]}"
