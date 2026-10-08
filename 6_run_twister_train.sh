#!/bin/bash
export CUDA_VISIBLE_DEVICES=6

env_name=atari100k-gopher run_name=atari100k override_config='{"retrieval_enabled":false}' python3 main.py
env_name=atari100k-gopher run_name=atari100k override_config='{"retrieval_enabled":false}' python3 main.py