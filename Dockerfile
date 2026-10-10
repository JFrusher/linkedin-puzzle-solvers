FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Bake model weights and corpora into the image so containers start offline.
RUN python -m nltk.downloader -d /usr/local/share/nltk_data wordnet \
    && python -c "import easyocr; easyocr.Reader(['en'], gpu=False)" \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8765 8501
ENTRYPOINT ["python", "main.py"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8765"]
