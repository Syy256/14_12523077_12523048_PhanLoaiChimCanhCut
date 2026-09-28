FROM python:3.12-slim

WORKDIR /app

COPY App/Backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY App/Backend/main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]