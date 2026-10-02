FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY dashboard ./dashboard
COPY sim ./sim

RUN pip install --no-cache-dir .

# demo data lives here
VOLUME /app/data

ENV DIETGATE_CONFIG_DIR=/app/config \
    DIETGATE_DASHBOARD_DIR=/app/dashboard \
    DIETGATE_DB_PATH=/app/data/dietgate.db

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "dietgate.main:app", "--host", "0.0.0.0", "--port", "8080"]
