# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps (optional but often helpful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port for FastAPI
EXPOSE 8080

# Environment variables
ENV PORT=8080

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]