#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=1
exec "${TWISTER_PYTHON:-python3}" -u reproduction_supervisor.py run -- --require-server --check-gpu "$@"
