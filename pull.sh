#!/bin/bash

GIT_TAG="release-v1.0.1"
VERSION="1.0.1"

BASE_URL="https://raw.githubusercontent.com/airockchip/rknn-llm"

rm -rf toolbox/rkllm_toolkit-* runtime/librkllm-*

wget -P "toolbox" "$BASE_URL/rkllm-toolkit/packages/rkllm_toolkit-1.0.1-cp38-cp38-linux_x86_64.whl"
wget -P "runtime/librkllm-$VERSION-aarch64" "$BASE_URL/$GIT_TAG/rkllm-runtime/runtime/Linux/librkllm_api/aarch64/librkllmrt.so"
wget -P "runtime/librkllm-$VERSION-aarch64" "$BASE_URL/$GIT_TAG/rkllm-runtime/runtime/Linux/librkllm_api/include/rkllm.h"
