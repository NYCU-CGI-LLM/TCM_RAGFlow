FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.10 python3.10-venv python3-pip build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

WORKDIR /workspace

COPY TCMEmbeddingModel/pyproject.toml TCMEmbeddingModel/uv.lock /workspace/TCMEmbeddingModel/

WORKDIR /workspace/TCMEmbeddingModel
COPY TCMEmbeddingModel /workspace/TCMEmbeddingModel

RUN pip install --no-cache-dir -e .

CMD ["bash"]
