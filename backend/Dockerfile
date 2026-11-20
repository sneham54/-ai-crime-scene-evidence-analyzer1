FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
RUN mkdir -p /app/backend/uploads/images /app/backend/uploads/videos \
    /app/backend/uploads/reports /app/backend/uploads/annotated \
    /app/backend/uploads/annotated_videos /app/backend/uploads/scene_change

EXPOSE 5000
CMD ["python", "run.py"]
