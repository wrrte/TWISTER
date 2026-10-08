#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

env_name=atari100k-seaquest run_name=atari100k override_config='{"retrieval":{"batch_size_reduction":"anchors"}}' python3 main.py
