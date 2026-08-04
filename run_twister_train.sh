#!/bin/bash
# 5번 GPU가 존재하는지 확인 (에러 없이 실행되면 존재하는 것)
if nvidia-smi -i 5 >/dev/null 2>&1; then
    export CUDA_VISIBLE_DEVICES=5
    echo "✅ GPU 5 is detected. Using GPU 5 (Robone)."
else
    export CUDA_VISIBLE_DEVICES=3
    echo "⚠️ GPU 5 not found. Falling back to GPU 3 (B200)."
fi


env_name=atari100k-seaquest run_name=atari100k python3 main.py
env_name=atari100k-alien run_name=atari100k python3 main.py
env_name=atari100k-alien run_name=atari100k python3 main.py


