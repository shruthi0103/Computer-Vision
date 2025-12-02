# 1. Use 'bookworm' (stable) instead of generic 'slim' (which is unstable)
FROM python:3.9-slim-bookworm

# 2. Install system dependencies
# CHANGED: Replaced 'libgl1-mesa-glx' with 'libgl1' (the new name)
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements
COPY requirements.txt .

# 5. Install Python dependencies
# dlib takes a long time (5-10 mins) to compile. This is normal.
RUN pip install --no-cache-dir dlib
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the app code
COPY . .

# 7. Create folders
RUN mkdir -p uploads static/mod2/temps static/marker static/sift_images

# 8. Expose port
EXPOSE 5000

# 9. Run the app
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]