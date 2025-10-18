FROM python:3.11-slim

# (Optional) system deps for OCR/audio (uncomment only if you need them)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     tesseract-ocr ffmpeg && \
#     rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better build cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# (Optional) pre-download the embedding model to speed up first run
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('intfloat/e5-small-v2')
print("✅ cached e5-small-v2")
PY

# Copy app code
COPY . ./

# Hugging Face Spaces exposes $PORT
ENV PORT=7860
# CMD ["gunicorn", "-b", "0.0.0.0:7860", "main:app"]
CMD ["bash","-lc","gunicorn -b 0.0.0.0:$PORT main:app"]
