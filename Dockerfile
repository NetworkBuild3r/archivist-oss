FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-cache-dir ".[postgres,qdrant]"

ARG UID=1000
ARG GID=1000

RUN addgroup --system --gid $GID archivist \
    && adduser --system --uid $UID --ingroup archivist --no-create-home archivist \
    && mkdir -p /data/archivist /data/memories \
    && chown -R archivist:archivist /data

USER archivist

# Containers explicitly opt into all-interface binding; bare-metal defaults to 127.0.0.1.
ENV ARCHIVIST_HOST=0.0.0.0

EXPOSE 3100

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:3100/health').raise_for_status()"

CMD ["python3", "-m", "archivist.app.main"]
