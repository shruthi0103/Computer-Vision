FROM python:3.9-slim-bookworm

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p uploads static/mod2/temps static/marker static/sift_images

EXPOSE 5000

# CHANGED: Using 'gthread' (threaded) instead of 'eventlet'. Safer for MediaPipe.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]