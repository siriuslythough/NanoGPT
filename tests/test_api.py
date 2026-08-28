import pytest

from fastapi.testclient import TestClient

from api.main import app


# ============================================================
# Shared client
#
# scope="module" means:
#
#     start FastAPI once
#     load model once
#     run all tests
#     shut down once
#
# Without this, loading the model separately for every test
# would be wasteful.
# ============================================================

@pytest.fixture(scope="module")
def client():

    with TestClient(app) as test_client:

        yield test_client


# ============================================================
# Root endpoint
# ============================================================

def test_root(client):

    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "running"


# ============================================================
# Health endpoint
# ============================================================

def test_health(client):

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"

    assert (
        body["model_loaded"]
        is True
    )

    assert (
        body["context_length"]
        == 256
    )

    assert (
        body["vocab_size"]
        == 512
    )


# ============================================================
# Generation smoke test
# ============================================================

def test_generation(client):

    response = client.post(
        "/generate",
        json={
            "prompt": "ROMEO:",
            "max_new_tokens": 4,
            "temperature": 0.6,
            "top_k": 40
        }
    )

    assert response.status_code == 200

    body = response.json()

    assert body[
        "generated_tokens"
    ] == 4

    assert body[
        "prompt_tokens"
    ] > 0

    assert body[
        "model_ttft_ms"
    ] > 0

    assert body[
        "latency_ms"
    ] > 0

    assert body[
        "tokens_per_second"
    ] > 0

    assert body[
        "device"
    ] in {
        "cpu",
        "cuda"
    }

    assert body[
        "text"
    ].startswith(
        "ROMEO:"
    )


# ============================================================
# Empty prompt
#
# Pydantic should reject this BEFORE inference.
# ============================================================

def test_empty_prompt_rejected(
    client
):

    response = client.post(
        "/generate",
        json={
            "prompt": "",
            "max_new_tokens": 4,
            "temperature": 0.6,
            "top_k": 40
        }
    )

    assert (
        response.status_code
        == 422
    )


# ============================================================
# Invalid temperature
# ============================================================

def test_invalid_temperature_rejected(
    client
):

    response = client.post(
        "/generate",
        json={
            "prompt": "ROMEO:",
            "max_new_tokens": 4,
            "temperature": -1.0,
            "top_k": 40
        }
    )

    assert (
        response.status_code
        == 422
    )


# ============================================================
# Invalid top-k
# ============================================================

def test_invalid_top_k_rejected(
    client
):

    response = client.post(
        "/generate",
        json={
            "prompt": "ROMEO:",
            "max_new_tokens": 4,
            "temperature": 0.6,
            "top_k": 0
        }
    )

    assert (
        response.status_code
        == 422
    )


# ============================================================
# Context overflow
#
# Byte-BPE guarantees arbitrary UTF-8 can be represented.
#
# Using many emoji creates far more than 256 byte-level/BPE
# tokens and therefore reliably exceeds our model context.
# ============================================================

def test_context_overflow_rejected(
    client
):

    response = client.post(
        "/generate",
        json={
            "prompt": "🙂" * 300,
            "max_new_tokens": 64,
            "temperature": 0.6,
            "top_k": 40
        }
    )

    assert (
        response.status_code
        == 400
    )

    body = response.json()

    assert (
        "context length"
        in body["detail"].lower()
    )