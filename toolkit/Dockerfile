FROM --platforms=linux/amd64 python:3.8-bookworm-slim

RUN apt update && apt install -y git && git lfs install

COPY rkllm_toolkit.sh /rkllm_toolkit.sh

RUN bash /rkllm_toolkit.sh && rm /rkllm_toolkit.sh

