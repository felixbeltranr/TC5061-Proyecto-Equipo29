FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt \
    && useradd --create-home --uid 10001 appuser

COPY mibici ./mibici

USER appuser

EXPOSE 8000

CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]