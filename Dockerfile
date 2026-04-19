FROM python:3.12-slim

WORKDIR /app

# Optional extras — pass --build-arg EXTRAS=postgres to install asyncpg
# for GRAPH_BACKEND=postgres support.  Multiple extras are comma-separated
# but today only "postgres" is defined.
#
# Examples:
#   docker build .                              # core only (SQLite backend)
#   docker build --build-arg EXTRAS=postgres .  # + asyncpg (Postgres backend)
ARG EXTRAS=""

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ -n "$EXTRAS" ]; then pip install --no-cache-dir "asyncpg>=0.29.0,<1.0"; fi

COPY src/ ./

ARG UID=1000
ARG GID=1000

RUN addgroup --system --gid $GID archivist \
    && adduser --system --uid $UID --ingroup archivist --no-create-home archivist \
    && mkdir -p /data/archivist /data/memories \
    && chown -R archivist:archivist /data

USER archivist

EXPOSE 3100

# /health returns 200 (healthy) or 503 (degraded but running — e.g. Postgres
# subsystem initialising).  Both are valid "container is alive" states; only
# non-2xx/5xx or a connection failure means the container is broken.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c \
    "import httpx, sys; r=httpx.get('http://localhost:3100/health'); sys.exit(0 if r.status_code in (200, 503) else 1)"

CMD ["python3", "-m", "archivist.app.main"]
