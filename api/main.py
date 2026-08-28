import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data.tokenizer import ByteBPETokenizer
from model.gpt import GPT


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = Path(
    "checkpoints/best.pt"
)

TOKENIZER_PATH = Path(
    "tokenizer.json"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(
    "nanogpt-api"
)


# ============================================================
# Global model state
#
# These objects will be loaded ONCE when the server starts.
# We do NOT reload the model for every request.
# ============================================================

model = None
tokenizer = None
model_config = None


# ============================================================
# Request schema
# ============================================================

class GenerationRequest(BaseModel):

    prompt: str = Field(
        ...,
        min_length=1,
        description="Text prompt for generation."
    )

    max_new_tokens: int = Field(
        default=64,
        ge=1,
        le=128
    )

    temperature: float = Field(
        default=0.8,
        gt=0.0,
        le=2.0
    )

    top_k: int = Field(
        default=40,
        ge=1,
        le=512
    )


# ============================================================
# Response schema
# ============================================================

class GenerationResponse(BaseModel):

    text: str

    prompt_tokens: int

    generated_tokens: int

    # Time until first token is COMPUTED by the model.
    model_ttft_ms: float

    # Complete model-generation time.
    latency_ms: float

    tokens_per_second: float

    device: str


class HealthResponse(BaseModel):

    status: str

    device: str

    model_loaded: bool

    context_length: int | None

    vocab_size: int | None


# ============================================================
# Sampling
# ============================================================

def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int
) -> torch.Tensor:
    """
    Convert model logits into one sampled token.

    logits:
        (B, vocab_size)

    Returns:
        (B, 1)
    """

    # --------------------------------------------------------
    # Temperature
    #
    # Lower temperature:
    #     sharper / more deterministic distribution
    #
    # Higher temperature:
    #     flatter / more random distribution
    # --------------------------------------------------------

    logits = (
        logits
        / temperature
    )


    # --------------------------------------------------------
    # Top-k
    #
    # Keep only the k most likely tokens.
    # Everything else becomes impossible.
    # --------------------------------------------------------

    k = min(
        top_k,
        logits.size(-1)
    )

    top_values, top_indices = (
        torch.topk(
            logits,
            k=k,
            dim=-1
        )
    )


    # Turn selected logits into probabilities.
    probabilities = torch.softmax(
        top_values,
        dim=-1
    )


    # Sample index INSIDE the top-k list.
    sampled_position = torch.multinomial(
        probabilities,
        num_samples=1
    )


    # Convert that position back into the actual
    # vocabulary token ID.
    next_token = torch.gather(
        top_indices,
        dim=-1,
        index=sampled_position
    )


    return next_token


# ============================================================
# Generation engine
# ============================================================

@torch.inference_mode()
def generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int
):

    global model
    global tokenizer
    global model_config


    # --------------------------------------------------------
    # Tokenize prompt
    # --------------------------------------------------------

    prompt_ids = tokenizer.encode(
        prompt
    )


    if len(prompt_ids) == 0:

        raise ValueError(
            "Prompt produced zero tokens."
        )


    context_length = (
        model_config["context_length"]
    )


    # --------------------------------------------------------
    # Context-window safety
    #
    # Our current learned positional embeddings only support
    # positions 0 ... context_length - 1.
    # --------------------------------------------------------

    if (
        len(prompt_ids)
        + max_new_tokens
        > context_length
    ):

        raise ValueError(
            f"Prompt uses {len(prompt_ids)} tokens and "
            f"generation requests {max_new_tokens}, "
            f"but model context length is "
            f"{context_length}."
        )


    tokens = torch.tensor(
        prompt_ids,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)


    # CUDA executes asynchronously, so synchronize before
    # taking wall-clock measurements.
    if DEVICE == "cuda":
        torch.cuda.synchronize()


    start_time = (
        time.perf_counter()
    )


    # ========================================================
    # PREFILL
    #
    # Process the full prompt once.
    #
    # This creates K/V entries for all prompt tokens.
    # ========================================================

    # ========================================================
    # PREFILL
    #
    # Process the entire prompt once and populate KV cache.
    # ========================================================

    logits, cache = (
        model.forward_cached(
            tokens,
            cache=None
        )
    )

    next_token = sample_next_token(
        logits[:, -1, :],
        temperature=temperature,
        top_k=top_k
    )

    # CUDA operations are asynchronous.
    # Synchronize so TTFT includes the actual GPU work required
    # to compute and sample the first token.
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    model_ttft = (
        time.perf_counter()
        - start_time
    )

    generated_tokens = [
        next_token
    ]


    # ========================================================
    # INCREMENTAL DECODING
    #
    # Feed ONLY the newly generated token on every step.
    #
    # Previous K/V tensors are reused from cache.
    # ========================================================

    for _ in range(
        max_new_tokens - 1
    ):

        logits, cache = (
            model.forward_cached(
                next_token,
                cache=cache
            )
        )


        next_token = sample_next_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k
        )


        generated_tokens.append(
            next_token
        )


    # --------------------------------------------------------
    # Assemble generated sequence
    # --------------------------------------------------------

    generated_tensor = torch.cat(
        generated_tokens,
        dim=1
    )


    full_sequence = torch.cat(
        [
            tokens,
            generated_tensor
        ],
        dim=1
    )


    if DEVICE == "cuda":
        torch.cuda.synchronize()


    elapsed = (
        time.perf_counter()
        - start_time
    )


    generated_count = (
        generated_tensor.size(1)
    )


    tokens_per_second = (
        generated_count
        / elapsed
    )


    # Move IDs back to CPU before converting to Python list.
    output_ids = (
        full_sequence[0]
        .detach()
        .cpu()
        .tolist()
    )


    output_text = tokenizer.decode(
        output_ids
    )


    return {
        "text": output_text,

        "prompt_tokens": (
            len(prompt_ids)
        ),

        "generated_tokens": (
            generated_count
        ),

        "model_ttft_ms": (
            model_ttft * 1000
        ),

        "latency_ms": (
            elapsed * 1000
        ),

        "tokens_per_second": (
            tokens_per_second
        ),

        "device": DEVICE
    }



# ============================================================
# Model loading
# ============================================================

def load_model():

    global model
    global tokenizer
    global model_config


    logger.info(
        "Loading tokenizer from %s",
        TOKENIZER_PATH
    )


    tokenizer = (
        ByteBPETokenizer.load(
            TOKENIZER_PATH
        )
    )


    logger.info(
        "Loading checkpoint from %s",
        CHECKPOINT_PATH
    )


    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )


    model_config = (
        checkpoint["config"]
    )


    model = GPT(
        **model_config
    ).to(DEVICE)


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.eval()


    logger.info(
        "Model loaded | device=%s | "
        "parameters=%d | context=%d | vocab=%d",
        DEVICE,
        sum(
            p.numel()
            for p in model.parameters()
        ),
        model_config[
            "context_length"
        ],
        model_config[
            "vocab_size"
        ]
    )


# ============================================================
# Model warm-up
# ============================================================

@torch.inference_mode()
def warm_up_model():
    """
    Perform a tiny inference when the server starts.

    Why?

    The first GPU request is often slower because CUDA must
    initialize kernels, memory paths, etc.

    We pay that cost at startup rather than making the first
    real user pay it.
    """

    logger.info(
        "Warming up model..."
    )


    warmup_ids = tokenizer.encode(
        "ROMEO:"
    )


    warmup_tensor = torch.tensor(
        warmup_ids,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)


    # Standard forward is enough to initialize
    # the important model execution path.
    _, _ = model.forward_cached(
        warmup_tensor,
        cache=None
    )


    if DEVICE == "cuda":
        torch.cuda.synchronize()


    logger.info(
        "Model warm-up complete."
    )


# ============================================================
# FastAPI lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Runs ONCE when server starts.
    load_model()

    warm_up_model()

    logger.info(
        "NanoGPT API ready."
    )


    yield


    # Runs once during shutdown.
    logger.info(
        "NanoGPT API shutting down."
    )


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="NanoGPT Inference API",

    description=(
        "Inference service for an "
        "11M-parameter decoder-only GPT."
    ),

    version="1.0.0",

    lifespan=lifespan
)


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "service": (
            "NanoGPT Inference API"
        ),

        "status": "running"
    }


# ============================================================
# Health endpoint
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
def health():

    loaded = (
        model is not None
        and tokenizer is not None
    )


    return {
        "status": (
            "healthy"
            if loaded
            else "unhealthy"
        ),

        "device": DEVICE,

        "model_loaded": loaded,

        "context_length": (
            model_config[
                "context_length"
            ]
            if model_config
            else None
        ),

        "vocab_size": (
            model_config[
                "vocab_size"
            ]
            if model_config
            else None
        )
    }


# ============================================================
# Generation endpoint
# ============================================================

@app.post(
    "/generate",
    response_model=GenerationResponse
)
def generate(
    request: GenerationRequest
):

    request_start = (
        time.perf_counter()
    )


    logger.info(
        "Generation request | "
        "prompt_chars=%d | "
        "max_new_tokens=%d | "
        "temperature=%.2f | "
        "top_k=%d",
        len(request.prompt),
        request.max_new_tokens,
        request.temperature,
        request.top_k
    )


    try:

        result = generate_text(
            prompt=request.prompt,

            max_new_tokens=(
                request.max_new_tokens
            ),

            temperature=(
                request.temperature
            ),

            top_k=request.top_k
        )


    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


    except Exception:

        logger.exception(
            "Generation failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Internal generation error."
            )
        )


    request_latency = (
        time.perf_counter()
        - request_start
    )


    logger.info(
        "Generation complete | "
        "generated_tokens=%d | "
        "model_ttft_ms=%.2f | "
        "model_latency_ms=%.2f | "
        "request_latency_ms=%.2f | "
        "throughput=%.2f tok/s",
        result[
            "generated_tokens"
        ],

        result[
            "model_ttft_ms"
        ],

        result[
            "latency_ms"
        ],

        request_latency * 1000,

        result[
            "tokens_per_second"
        ]
    )


    return result