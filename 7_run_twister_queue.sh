#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
    echo "Usage: $0 GPU_ID [QUEUE_FILE]" >&2
    exit 2
fi

GPU_ID="$1"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must be a non-negative NVIDIA GPU index." >&2
    exit 2
fi

# Resolve main.py, queue files, and output paths from the TWISTER directory.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

for dependency in nvidia-smi flock; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
        echo "Required command not found: $dependency" >&2
        exit 1
    fi
done

if ! GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_ID"); then
    echo "Could not query GPU $GPU_ID." >&2
    exit 1
fi
if [[ -z "$GPU_NAME" ]]; then
    echo "No GPU name returned for GPU $GPU_ID." >&2
    exit 1
fi

case "${GPU_NAME,,}" in
    *a6000*) QUEUE_SUFFIX="A6000" ;;
    *3090*) QUEUE_SUFFIX="3090" ;;
    *"titan rtx"*) QUEUE_SUFFIX="titan" ;;
    *blackwell*6000*|*6000*blackwell*) QUEUE_SUFFIX="pro6k" ;;
    *) QUEUE_SUFFIX="default" ;;
esac

QUEUE_FILE="${2:-job_queue_${QUEUE_SUFFIX}.txt}"
LOCK_FILE="${QUEUE_FILE%.txt}.lock"
touch -- "$QUEUE_FILE"

echo "Worker started on GPU $GPU_ID ($GPU_NAME). Waiting for jobs in $QUEUE_FILE..."

while true; do
    job=""

    # Workers sharing a queue must pop its first command under the same lock.
    exec 200>"$LOCK_FILE"
    flock -x 200
    sed -i '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "$QUEUE_FILE"
    if [[ -s "$QUEUE_FILE" ]]; then
        job=$(head -n 1 -- "$QUEUE_FILE")
        sed -i '1d' "$QUEUE_FILE"
    fi
    flock -u 200
    exec 200>&-

    if [[ -n "$job" ]]; then
        echo "[GPU $GPU_ID] Found job: $job"
        # Isolate each command's environment, directory, and exit from the worker.
        if ( eval "$job" ); then
            echo "[GPU $GPU_ID] Job finished."
        else
            job_status=$?
            echo "[GPU $GPU_ID] Job failed (exit $job_status). Continuing with the next job." >&2
        fi
    else
        sleep 10
    fi
done
