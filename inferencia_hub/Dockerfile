# syntax=docker/dockerfile:1.7

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

ARG INSTALL_ML=0

COPY inferencia_hub/requirements-base.txt /app/requirements-base.txt
COPY inferencia_hub/requirements-ml.txt /app/requirements-ml.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements-base.txt && \
    if [ "$INSTALL_ML" = "1" ]; then python -m pip install -r /app/requirements-ml.txt; fi

COPY inferencia_hub /app/inferencia_hub
COPY inferencia_hub/docker_server.py /app/server.py

ENV WEB_DIR=/app/inferencia_hub/web

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
