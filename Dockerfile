# Base image pinned by digest (python:3.12-slim, same as Project 02); bump deliberately.
FROM python:3.12-slim@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /app

# Service dependencies only: no dev tools, no evidently/scipy (the offline drift job).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-default-groups --no-install-project

COPY app ./app
COPY models ./models
RUN uv sync --locked --no-dev --no-default-groups

# Non-root, no shell login, no home. Code and model are root-owned and read-only to this user;
# only the observation directory is writable (a named volume copies this ownership on first use).
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /data/observations && chown 10001 /data/observations
USER 10001

ARG SOURCE_REVISION=""
ENV SOURCE_REVISION=${SOURCE_REVISION} PORT=8000 OBSERVATION_DIR=/data/observations
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
