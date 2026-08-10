FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    ARTIFACT_ROOT=/app/artifacts \
    WEB_ROOT=/app/web

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

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
