FROM python:3.12-slim

# mediapipe/opencv need these system libs even in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.15.0+cpu torchvision==0.29.2+cpu \
      --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY checkpoint ./checkpoint

ENV MODEL_PATH=/app/checkpoint/checkpoint_aug_epoch70.pth
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
