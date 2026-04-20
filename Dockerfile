# syntax=docker/dockerfile:1.6
FROM python:3.11-slim-bookworm AS base

LABEL org.opencontainers.image.source="https://github.com/research-AI-studies/atmospheric-intelligence-engine"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.description="Reference implementation of the Atmospheric Intelligence Engine."

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY . .
RUN pip install -e ".[dev]"

CMD ["make", "smoke"]
