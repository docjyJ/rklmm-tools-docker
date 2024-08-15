# Tools for running LLM on Rockchip processors


## `rkllm-toolkit`: Convert image to work with rkllm runtime

### Usage

```bash
# RUN in x86-64 computer
docker run --rm -v ./output:/output ghcr.io/docjyj/rkllm-toolkit:develop -o /output microsoft/Phi-3-mini-4k-instruct
docker run --rm ghcr.io/docjyj/rkllm-toolkit:develop --help
```


## `rkllm-runtime`: Run LLM on Rockchip processors

### Usage

```bash
# RUN in Rockchip computer
docker run --rm -v ./output:/output ghcr.io/docjyj/rkllm-runtime:develop /output/Phi-3-mini-4k-instruct.rkllm
docker run --rm ghcr.io/docjyj/rkllm-runtime:develop --help
```


## Thanks

This project is based on the work of [Pelochus/ezrknn-llm](https://github.com/Pelochus/ezrknn-llm)
