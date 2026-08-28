import argparse
import statistics
import time

import httpx


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--url",
    type=str,
    default="http://127.0.0.1:8000"
)

parser.add_argument(
    "--requests",
    type=int,
    default=20
)

parser.add_argument(
    "--warmup",
    type=int,
    default=3
)

parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=64
)

parser.add_argument(
    "--prompt",
    type=str,
    default="ROMEO:"
)

parser.add_argument(
    "--temperature",
    type=float,
    default=0.6
)

parser.add_argument(
    "--top-k",
    type=int,
    default=40
)

args = parser.parse_args()


# ============================================================
# Utility
# ============================================================

def percentile(values, p):

    values = sorted(values)

    if not values:
        return 0.0

    position = (
        (len(values) - 1)
        * p / 100
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(values) - 1
    )

    fraction = (
        position - lower
    )

    return (
        values[lower]
        * (1 - fraction)
        +
        values[upper]
        * fraction
    )


# ============================================================
# Request payload
# ============================================================

payload = {
    "prompt": args.prompt,
    "max_new_tokens": args.max_new_tokens,
    "temperature": args.temperature,
    "top_k": args.top_k
}


generate_url = (
    args.url.rstrip("/")
    + "/generate"
)

health_url = (
    args.url.rstrip("/")
    + "/health"
)


# ============================================================
# Check service
# ============================================================

with httpx.Client(
    timeout=30.0
) as client:

    health_response = client.get(
        health_url
    )

    health_response.raise_for_status()

    health = (
        health_response.json()
    )


    print("=" * 68)
    print("API BENCHMARK")
    print("=" * 68)

    print(
        f"URL:                "
        f"{args.url}"
    )

    print(
        f"Device:             "
        f"{health['device']}"
    )

    print(
        f"Context length:     "
        f"{health['context_length']}"
    )

    print(
        f"Prompt:             "
        f"{args.prompt!r}"
    )

    print(
        f"Generated tokens:   "
        f"{args.max_new_tokens}"
    )

    print(
        f"Warm-up requests:   "
        f"{args.warmup}"
    )

    print(
        f"Measured requests:  "
        f"{args.requests}"
    )


    # ========================================================
    # Warm-up
    # ========================================================

    print()
    print("Warming up API...")

    for _ in range(
        args.warmup
    ):

        response = client.post(
            generate_url,
            json=payload
        )

        response.raise_for_status()


    # ========================================================
    # Benchmark
    # ========================================================

    http_latencies = []

    model_latencies = []

    model_ttfts = []

    model_throughputs = []

    errors = 0


    for i in range(
        args.requests
    ):

        start = (
            time.perf_counter()
        )


        try:

            response = client.post(
                generate_url,
                json=payload
            )


            end = (
                time.perf_counter()
            )


            if response.status_code != 200:

                errors += 1

                print(
                    f"Request {i + 1}: "
                    f"HTTP "
                    f"{response.status_code}"
                )

                continue


            body = (
                response.json()
            )


            http_latency_ms = (
                (end - start)
                * 1000
            )


            http_latencies.append(
                http_latency_ms
            )


            model_latencies.append(
                body[
                    "latency_ms"
                ]
            )


            model_ttfts.append(
                body[
                    "model_ttft_ms"
                ]
            )


            model_throughputs.append(
                body[
                    "tokens_per_second"
                ]
            )


        except Exception as error:

            errors += 1

            print(
                f"Request {i + 1} "
                f"failed: {error}"
            )


# ============================================================
# Results
# ============================================================

successful = (
    len(http_latencies)
)

total = (
    args.requests
)

error_rate = (
    errors / total * 100
    if total > 0
    else 0.0
)


print()
print("-" * 68)
print("HTTP END-TO-END LATENCY")
print("-" * 68)

if successful > 0:

    print(
        f"Mean:               "
        f"{statistics.mean(http_latencies):.2f} ms"
    )

    print(
        f"p50:                "
        f"{percentile(http_latencies, 50):.2f} ms"
    )

    print(
        f"p95:                "
        f"{percentile(http_latencies, 95):.2f} ms"
    )

    print(
        f"p99:                "
        f"{percentile(http_latencies, 99):.2f} ms"
    )


print()
print("-" * 68)
print("MODEL METRICS")
print("-" * 68)

if successful > 0:

    print(
        f"Mean model latency: "
        f"{statistics.mean(model_latencies):.2f} ms"
    )

    print(
        f"Mean model TTFT:    "
        f"{statistics.mean(model_ttfts):.2f} ms"
    )

    print(
        f"Mean throughput:    "
        f"{statistics.mean(model_throughputs):.2f} tok/s"
    )


print()
print("-" * 68)
print("RELIABILITY")
print("-" * 68)

print(
    f"Successful:         "
    f"{successful}/{total}"
)

print(
    f"Errors:             "
    f"{errors}"
)

print(
    f"Error rate:         "
    f"{error_rate:.2f}%"
)


print()
print("-" * 68)
print("HTTP OVERHEAD")
print("-" * 68)

if successful > 0:

    mean_http = (
        statistics.mean(
            http_latencies
        )
    )

    mean_model = (
        statistics.mean(
            model_latencies
        )
    )

    overhead = (
        mean_http
        - mean_model
    )

    print(
        f"Mean HTTP latency:  "
        f"{mean_http:.2f} ms"
    )

    print(
        f"Mean model latency: "
        f"{mean_model:.2f} ms"
    )

    print(
        f"Approx. overhead:   "
        f"{overhead:.2f} ms"
    )


print("=" * 68)