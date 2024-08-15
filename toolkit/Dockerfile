FROM --platform=linux/amd64 python:3.8-slim-bookworm

RUN apt update && apt install -y git-lfs

COPY rkllm_toolkit-1.0.1-cp38-cp38-linux_x86_64.whl /rkllm_toolkit.whl

RUN pip install --no-cache-dir /rkllm_toolkit.whl && rm /rkllm_toolkit.whl

COPY toolkit /bin/toolkit

ENTRYPOINT ["/bin/toolkit"]

