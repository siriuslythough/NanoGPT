FROM python:3.11-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Project root is importable by Python.
ENV PYTHONPATH=/app


WORKDIR /app


COPY requirements.txt .

# Upgrade pip first.
RUN pip install --no-cache-dir --upgrade pip


# ============================================================
# CUDA-enabled PyTorch
#
# Host NVIDIA driver currently supports CUDA 12.5.
# We deliberately use the official CUDA 12.4 PyTorch build.
# ============================================================

RUN pip install \
    --no-cache-dir \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124


# ============================================================
# Remaining application dependencies
# ============================================================

RUN pip install \
    --no-cache-dir \
    -r requirements.txt


COPY api ./api
COPY data ./data
COPY model ./model
COPY tests ./tests

COPY tokenizer.json ./tokenizer.json
COPY checkpoints/best.pt ./checkpoints/best.pt

COPY generate.py .
COPY benchmark_inference.py .


RUN python -m pytest -q


EXPOSE 8000


HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1


CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]