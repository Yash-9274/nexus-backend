# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create NLTK data directory and download required data
RUN mkdir -p /usr/local/share/nltk_data
ENV NLTK_DATA=/usr/local/share/nltk_data

COPY . .

# Download NLTK data during build
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words'); nltk.download('stopwords')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]