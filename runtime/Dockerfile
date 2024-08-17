FROM gcc:14-bookworm as builder

COPY src/librkllm/librkllm-1.0.1-aarch64/librkllmrt.so /usr/lib/librkllmrt.so
COPY src/librkllm/librkllm-1.0.1-aarch64/rkllm.h /usr/local/include/rkllm.h
COPY runtime.c /tmp/runtime.c

RUN apt-get update && apt-get install -y g++ && \
    gcc -o /usr/bin/rkllm-runtime /tmp/runtime.c -lrkllmrt

FROM debian:bookworm-slim

COPY --from=builder /usr/bin/rkllm-runtime /usr/bin/rkllm-runtime

ENTRYPOINT ["/bin/rkllm-runtime"]

