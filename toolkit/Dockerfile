FROM python:3.8-slim-bookworm

RUN apt-ge update && apt-get install -y git-lfs

COPY rkllm_toolkit-1.0.1-cp38-cp38-linux_x86_64.whl /tmp/rkllm_toolkit-1.0.1-cp38-cp38-linux_x86_64.whl

RUN pip install --no-cache-dir /tmp/rkllm_toolkit-1.0.1-cp38-cp38-linux_x86_64.whl && \
    rm /tmp/rkllm_toolkit-1.0.1-cp38-cp38-linux_x86_64.whl

COPY toolkit /bin/toolkit

ENTRYPOINT ["/bin/toolkit"]

