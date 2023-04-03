# syntax = docker/dockerfile:1

FROM huggingface/transformers-deepspeed-gpu:latest

RUN apt -y update \
    && apt install -y libaio-dev \
    && python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip uninstall -y torch-tensorrt pydantic

ARG PYTORCH='2.0.0'
ARG CUDA='cu117'

RUN python3 -m pip install --no-cache-dir -U torch==$PYTORCH torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/$CUDA

# recompile apex
RUN python3 -m pip uninstall -y apex
RUN git clone https://github.com/NVIDIA/apex
#  `MAX_JOBS=1` disables parallel building to avoid cpu memory OOM when building image on GitHub Action (standard) runners
RUN cd apex && MAX_JOBS=1 python3 -m pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .

# Pre-build **latest** DeepSpeed
RUN cmake --version
RUN python3 -m pip uninstall -y deepspeed \
    && DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 python3 -m pip install deepspeed --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1

# Install the remaining dependencies
COPY pyproject.toml setup.cfg /workspace/
WORKDIR /workspace
RUN mkdir llm_racing \
    && python3 -m pip install --nocache-dir -U -e . \
    && python3 -m pip uninstall -U -y llm_racing \
    && rmdir llm_racing && rm pyproject.toml
