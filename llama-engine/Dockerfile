# syntax=docker/dockerfile:1.7
# llama-engine — TurboQuant fork of llama.cpp built for NVIDIA Pascal (compute 6.1)
#
# Two-stage build:
#   1) builder: nvcc + cmake + ninja, clones TurboQuant, compiles the binaries
#   2) runtime: cuda runtime only, copies in the binaries — much smaller image

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG CUDA_ARCH=61

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG CUDA_ARCH
ARG TURBOQUANT_REPO=https://github.com/TheTom/llama-cpp-turboquant.git
ARG TURBOQUANT_BRANCH=feature/turboquant-kv-cache
ARG TURBOQUANT_REF=HEAD

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates \
        libcurl4-openssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 -b "${TURBOQUANT_BRANCH}" "${TURBOQUANT_REPO}" turboquant \
 && cd turboquant && git rev-parse HEAD > /src/turboquant.sha

WORKDIR /src/turboquant
# Pascal-tuned CMake config:
#   - CMAKE_CUDA_ARCHITECTURES=61   so kernels target GTX 1080 specifically
#   - GGML_CUDA_F16=OFF             Pascal FP16 throughput is 1:64 of FP32 — disable
#   - GGML_CUDA_FA_ALL_QUANTS=ON    compile FlashAttention kernels for every KV quant
RUN cmake -G Ninja -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DGGML_CUDA=ON \
      -DGGML_CUDA_FA=ON \
      -DGGML_CUDA_FA_ALL_QUANTS=ON \
      -DGGML_CUDA_F16=OFF \
      -DLLAMA_CURL=ON \
 && cmake --build build --config Release --parallel "$(nproc)"

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libcurl4 libgomp1 tini python3-minimal \
    && rm -rf /var/lib/apt/lists/*

# Pascal runtime tuning baked into the image; can be overridden at `docker run`.
ENV GGML_CUDA_FORCE_MMQ=1 \
    GGML_CUDA_F16=0

# Copy compiled binaries plus every .so the build emitted. This is robust to
# upstream llama.cpp moving library locations between releases.
COPY --from=builder /src/turboquant.sha        /etc/turboquant.sha
COPY --from=builder /src/turboquant/build/bin/ /usr/local/bin/
# Stage the libraries via an intermediate dir to flatten the layout.
RUN --mount=type=bind,from=builder,source=/src/turboquant/build,target=/build \
    set -eux; \
    find /build -name '*.so*' -type f -exec cp -P {} /usr/local/lib/ \; ; \
    ldconfig

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Default knobs — every one of these is tunable from docker-compose / .env.
ENV MODEL_FILE=/models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    ALIAS=qwen3.6-35b-a3b \
    HOST=0.0.0.0 \
    PORT=8080 \
    NGL=999 \
    N_CPU_MOE=30 \
    CTX=8192 \
    KV_TYPE=q8_0 \
    THREADS=auto \
    BATCH=2048 \
    UBATCH=2048 \
    FLASH_ATTN=on \
    EXTRA_ARGS=""

EXPOSE 8080
VOLUME ["/models", "/config"]

# tini reaps zombies so logs flush cleanly on `docker stop`.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1
