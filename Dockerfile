FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY config ./config

ENV CONFIG_DIR=/app/config
EXPOSE 8000

CMD ["uvicorn", "--factory", "tokenomics.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
