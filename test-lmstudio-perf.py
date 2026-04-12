#!/usr/bin/env python3
"""
LM Studio performance benchmarking script.
Tests prefill (time-to-first-token) and generation (decode) throughput
using the OpenAI-compatible /v1/completions streaming endpoint.

Usage:
    python3 test-lmstudio-perf.py
    python3 test-lmstudio-perf.py --url http://127.0.0.1:1234 --tokens 50
"""
import urllib.request
import json
import time
import argparse

BASE_URL = "http://127.0.0.1:1234"
BASE_WORDS = ["The", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog.", "Here", "is", "some", "more", "text", "to", "fill", "up", "the", "context", "window."]


def bench_context(url: str, prompt: str, n_predict: int) -> tuple[int, float, float] | None:
    """
    Send a streaming completion request and return (prompt_tokens, prefill_t/s, decode_t/s).
    - prefill t/s  = prompt_tokens / time_to_first_token
    - decode  t/s  = tokens_generated / decode_duration
    Returns None on error.
    """
    payload = json.dumps({
        "prompt": prompt,
        "max_tokens": n_predict,
        "temperature": 0.0,
        "stream": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        t_send = time.perf_counter()
        t_first = None
        tokens_generated = 0
        prompt_tokens = 0

        with urllib.request.urlopen(req) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Capture prompt token count from the first chunk that has usage
                if "usage" in chunk and chunk["usage"]:
                    prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                text = choices[0].get("text", "")
                finish = choices[0].get("finish_reason")

                # Record time of first actual content token
                if t_first is None and text:
                    t_first = time.perf_counter()

                if text:
                    tokens_generated += 1

                if finish and finish != "null":
                    # Some servers put usage in the final chunk
                    if "usage" in chunk and chunk["usage"]:
                        prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)

        t_end = time.perf_counter()

        if t_first is None:
            print("  Warning: no content tokens received — is a model loaded in LM Studio?")
            return None

        prefill_duration = t_first - t_send          # seconds until first token
        decode_duration  = t_end - t_first           # seconds for remaining tokens

        # Fallback: estimate prompt tokens from word count if server didn't report them
        if prompt_tokens == 0:
            prompt_tokens = max(1, len(prompt.split()))

        prefill_tps = prompt_tokens / prefill_duration if prefill_duration > 0 else 0.0
        decode_tps  = tokens_generated / decode_duration if decode_duration > 0 else 0.0

        return prompt_tokens, prefill_tps, decode_tps

    except urllib.error.URLError as e:
        print(f"  Connection failed: {e}")
        print(f"  Is LM Studio running with a model loaded at {url}?")
        return None
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Benchmark LM Studio inference speed")
    parser.add_argument("--url", default=BASE_URL, help=f"LM Studio base URL (default: {BASE_URL})")
    parser.add_argument("--tokens", type=int, default=50, help="Tokens to generate per test (default: 50)")
    parser.add_argument("--sizes", default="128,1024,4096", help="Comma-separated context sizes to test (default: 128,1024,4096)")
    args = parser.parse_args()

    target_sizes = [int(s.strip()) for s in args.sizes.split(",")]

    print(f"Testing LM Studio performance at {args.url} ...")
    print(f"Generating {args.tokens} tokens per test\n")

    # Quick connectivity check
    try:
        urllib.request.urlopen(f"{args.url}/v1/models", timeout=5)
    except urllib.error.URLError as e:
        print(f"Cannot reach {args.url}: {e}")
        print("Make sure LM Studio is running and the local server is started.")
        return

    print(f"{'Context Size':<15} | {'Prefill (Prompt)':<22} | {'Generation (Decode)':<22} | {'Note'}")
    print("-" * 85)

    for target_words in target_sizes:
        prompt_words = (BASE_WORDS * (target_words // len(BASE_WORDS) + 1))[:target_words]
        prompt = " ".join(prompt_words)

        result = bench_context(args.url, prompt, args.tokens)
        if result is None:
            print(f"{'~' + str(target_words) + ' tokens':<15} | {'ERROR':<22} | {'ERROR':<22} | request failed")
            continue

        prompt_tokens, prefill_tps, decode_tps = result
        note = "(est. prompt tokens)" if prompt_tokens == len(prompt.split()) else ""
        print(
            f"{str(prompt_tokens) + ' tokens':<15} | "
            f"{prefill_tps:.2f} t/s".ljust(22) + "| "
            f"{decode_tps:.2f} t/s".ljust(22) + f"| {note}"
        )

    print()
    print("Note: prefill = prompt_tokens / time-to-first-token (includes model scheduling overhead)")
    print("      decode  = tokens_generated / time_after_first_token")


if __name__ == "__main__":
    main()
