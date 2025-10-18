FROM python:3.11-slim

# Keep Python lean & logs unbuffered
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for OCR/audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (uses cache better)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) pre-cache the embedding model
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('intfloat/e5-small-v2')
print("✅ cached e5-small-v2")
PY

# Copy app code
COPY . ./

# Spaces sets $PORT
ENV PORT=7860
EXPOSE 7860

# Gunicorn entrypoint
CMD ["bash","-lc","exec gunicorn main:app -b 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 180"]
