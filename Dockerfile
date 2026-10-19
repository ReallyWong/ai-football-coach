FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; onnxruntime wheels are self-contained.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

CMD uvicorn services.api.app:app --host 0.0.0.0 --port ${PORT:-8000}
