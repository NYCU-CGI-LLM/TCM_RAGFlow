FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

WORKDIR /workspace

COPY ragflow/sdk/python /workspace/ragflow/sdk/python
COPY ragflow_eval/pyproject.toml ragflow_eval/uv.lock /workspace/ragflow_eval/

WORKDIR /workspace/ragflow_eval
COPY ragflow_eval /workspace/ragflow_eval

RUN pip install --no-cache-dir -e /workspace/ragflow/sdk/python \
    && pip install --no-cache-dir -e .

CMD ["bash"]
