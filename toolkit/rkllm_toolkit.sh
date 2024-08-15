#!/bin/sh

wget https://raw.githubusercontent.com/airockchip/rknn-llm/release-v1.0.1/rkllm-toolkit/packages/rkllm_toolkit-1.0.1-cp38-cp38-linux_x86_64.whl
pip install --no-cache-dir rkllm_toolkit-1.0.1-cp38-cp38-linux_x86_64.whl
rm rkllm_toolkit-1.0.1-cp38-cp38-linux_x86_64.whl
