
FROM python:3.9-slim


RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY . .


RUN ls -lR /app/local_sentiment_model


RUN pip install --no-cache-dir --timeout=600 -r requirements.txt


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]