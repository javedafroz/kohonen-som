FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    ARTIFACT_ROOT=/app/artifacts \
    WEB_ROOT=/app/web \
    WEB_CONCURRENCY=2 \
    MAX_CONCURRENT_TRAINS=2 \
    TRAIN_QUEUE_TIMEOUT_SEC=30

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

# Build context is the monorepo root
COPY packages/som_core /tmp/som_core
RUN pip install --no-cache-dir /tmp/som_core \
    && rm -rf /tmp/som_core

COPY apps/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/api/main.py ./
COPY apps/web ./web

RUN mkdir -p /app/artifacts

EXPOSE 8000

# Multiple workers for multi-user request handling; training is capped per worker
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY:-2}"]
