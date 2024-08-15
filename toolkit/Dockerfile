FROM --platform=linux/amd64 python:3.8-slim-bookworm

RUN apt update && apt install -y git && git lfs install

COPY rkllm_toolkit.sh /rkllm_toolkit.sh

RUN bash /rkllm_toolkit.sh && rm /rkllm_toolkit.sh

COPY toolkit /bin/toolkit

ENTRYPOINT ["/bin/toolkit"]

