# Образ с CUDA/cuDNN; сервис опционально поднимается на хосте с GPU.
# CUDA 12.2.x: ориентир по строке nvidia-smi «CUDA Version» (макс. toolkit для драйвера).
# Драйвер 535.x обычно показывает 12.2 — образ 12.6+ даёт forward compatibility / ошибки на Pascal (1080 Ti).
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget gnupg ca-certificates ffmpeg libsndfile1 \
        python3 python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# PyTorch (CUDA 12.1 wheels совместимы с драйвером / toolkit 12.2 в образе nvidia/cuda:12.2).
COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir -r /app/requirements.txt

WORKDIR /app
COPY download_models.py /app/download_models.py
COPY app /app/app

EXPOSE 19900
ENV PYTHONUNBUFFERED=1
CMD ["python3", "-m", "app"]