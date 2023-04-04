# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-08.html#rel_22-08
FROM nvcr.io/nvidia/pytorch:22.08-py3
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

ARG PYTORCH='2.0.0'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu117'

RUN apt -y update
RUN apt install -y libaio-dev
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=v4.27.4
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

# Install latest release PyTorch
# (PyTorch must be installed before pre-compiling any DeepSpeed c++/cuda ops.)
# (https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops)
RUN python3 -m pip install --no-cache-dir -U torch==$PYTORCH torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip install --no-cache-dir ./transformers[deepspeed-testing]

# Uninstall `torch-tensorrt` shipped with the base image
RUN python3 -m pip uninstall -y torch-tensorrt

# recompile apex
RUN python3 -m pip uninstall -y apex
RUN git clone https://github.com/NVIDIA/apex
#  `MAX_JOBS=1` disables parallel building to avoid cpu memory OOM when building image on GitHub Action (standard) runners
RUN cd apex && MAX_JOBS=1 python3 -m pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check .

# Pre-build **latest** DeepSpeed, so it would be ready for testing (otherwise, the 1st deepspeed test will timeout)
RUN python3 -m pip uninstall -y deepspeed
# This has to be run (again) inside the GPU VMs running the tests.
# The installation works here, but some tests fail, if we don't pre-build deepspeed again in the VMs running the tests.
# TODO: Find out why test fail.
RUN DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 python3 -m pip install deepspeed --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1

# Install the remaining dependencies
COPY pyproject.toml setup.cfg /workspace/
WORKDIR /workspace
RUN mkdir llm_racing \
    && python3 -m pip install --no-cache-dir -U -e . \
    && python3 -m pip uninstall -y llm_racing \
    && rmdir llm_racing && rm pyproject.toml
