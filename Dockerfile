FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn starlette

COPY src/ ./

ARG UID=1000
ARG GID=1000

RUN addgroup --system --gid $GID archivist \
    && adduser --system --uid $UID --ingroup archivist --no-create-home archivist \
    && mkdir -p /data/archivist /data/memories \
    && chown -R archivist:archivist /data

USER archivist

EXPOSE 3100

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:3100/health').raise_for_status()"

CMD ["python", "main.py"]
