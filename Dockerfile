# 1. Use a slim Python version to keep size down
FROM python:3.9-slim

# 2. Install system dependencies required for OpenCV, Dlib, and MediaPipe
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements first (to cache dependencies)
COPY requirements.txt .

# 5. Install Python dependencies
# We install dlib first because it takes the longest
RUN pip install --no-cache-dir dlib
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code
COPY . .

# 7. Create necessary folders so the app doesn't crash
RUN mkdir -p uploads static/mod2/temps static/marker static/sift_images

# 8. Expose the port Railway uses
EXPOSE 5000

# 9. Run the app using Gunicorn with Eventlet (REQUIRED for WebSockets)
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]