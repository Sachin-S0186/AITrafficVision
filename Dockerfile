FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app.py ./app.py
COPY frontend ./frontend
COPY models ./models
ENV MODEL_PATH=/app/models/metr_la_ridge.joblib
ENV PORT=8000
EXPOSE 8000
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
