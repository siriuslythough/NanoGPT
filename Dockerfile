# ============================================================
# Base image
#
# We use Python 3.11 inside the container.
#
# Your host machine uses Python 3.14, but containers do not
# need to use the host's Python installation.
# ============================================================

FROM python:3.11-slim


# ============================================================
# Environment
# ============================================================

# Don't create .pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Print Python logs directly instead of buffering them.
ENV PYTHONUNBUFFERED=1


# ============================================================
# Working directory
#
# Everything below happens relative to /app.
# ============================================================

WORKDIR /app


# ============================================================
# Install Python dependencies
#
# Copy requirements separately first.
#
# Docker caches layers. If requirements.txt does not change,
# dependency installation can be reused on future builds.
# ============================================================

COPY requirements.txt .

RUN pip install \
    --no-cache-dir \
    -r requirements.txt


# ============================================================
# Copy source code
# ============================================================

COPY api ./api
COPY model ./model
COPY data ./data
COPY tests ./tests


# ============================================================
# Copy model artifacts
# ============================================================

COPY tokenizer.json .
COPY checkpoints/best.pt ./checkpoints/best.pt


# ============================================================
# Copy anything required by imports/tests
# ============================================================

COPY *.py ./


# ============================================================
# Document the port used by FastAPI
#
# EXPOSE does not itself publish the port.
# It documents that this application listens on 8000.
# ============================================================

EXPOSE 8000


# ============================================================
# Test during image build
#
# If our API tests fail, Docker refuses to build the image.
# ============================================================

RUN pytest -q


# ============================================================
# Container startup command
#
# IMPORTANT:
#
# 0.0.0.0 means listen on all network interfaces inside
# the container.
#
# If we used 127.0.0.1 here, the API would only be reachable
# from inside the container itself.
# ============================================================

CMD [
    "uvicorn",
    "api.main:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]