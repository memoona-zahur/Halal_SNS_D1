FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ---- api: HTTP intake only, no model runtime ---------------------------------
FROM base AS api

# Dependencies first, from pyproject.toml alone, so a code change does not reinstall them.
COPY pyproject.toml ./
RUN mkdir moderation && touch moderation/__init__.py && pip install . && rm -rf moderation build

COPY moderation ./moderation
COPY migrations ./migrations
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/uploads && chown app:app /app/uploads
USER app

EXPOSE 8000

CMD ["uvicorn", "moderation.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

# ---- worker: loads the models once at startup --------------------------------
FROM base AS worker

# libglib2.0-0: OpenCV (RapidOCR). libgomp1: ONNX Runtime threading.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Torch first and on its own layer, so source changes do not reinstall it.
# GPU build by default; docker-compose.cpu.yml passes the CPU index (a much smaller image).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN pip install torch --index-url ${TORCH_INDEX}

# Dependencies first, from pyproject.toml alone, so a code change does not reinstall them.
COPY pyproject.toml ./
RUN mkdir moderation && touch moderation/__init__.py && pip install ".[models]" && rm -rf moderation build

COPY moderation ./moderation
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/uploads && chown app:app /app/uploads
USER app

# Model weights are mounted read-only at /app/models, never baked into the image.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

CMD ["python", "-m", "moderation.worker"]
