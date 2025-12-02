FROM python:3.9-slim-bookworm

# Install light system dependencies (No cmake/gcc needed anymore!)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install dependencies (Fast build)
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create all needed folders
RUN mkdir -p uploads static/mod2/temps static/marker static/sift_images

EXPOSE 5000

CMD ["gunicorn", "-k", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]