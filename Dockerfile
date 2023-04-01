# syntax = docker/dockerfile:1

FROM deepspeed/deepspeed:v072_torch112_cu117 as base

COPY pyproject.toml setup.cfg /workspace/
WORKDIR /workspace
RUN mkdir llm_racing \
    && python -m pip install -e . \
    && python -m pip uninstall -y llm_racing \
    && rmdir llm_racing && rm pyproject.toml
