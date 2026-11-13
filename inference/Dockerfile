FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /service
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY data.py models.py losses.py engine.py infer.py server.py ./
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
