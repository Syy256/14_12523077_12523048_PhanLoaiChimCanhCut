FROM python:3.12-slim

WORKDIR /app

# Cài dependencies
COPY AI-models/service/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Giữ cấu trúc:
# /app/service/main.py
# /app/models/model.joblib
# /app/models/metadata.json
# /app/models/schema.json

COPY AI-models/service /app/service
COPY AI-models/models /app/models

EXPOSE 8000

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]