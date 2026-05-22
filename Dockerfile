FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN mkdir -p logs reports artifacts data mlruns

EXPOSE 8000

CMD ["uvicorn", "src.ai_guard.gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]