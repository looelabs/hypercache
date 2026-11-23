#!/usr/bin/env python3
"""
Manual end-to-end check of the LMCache connector under a running vLLM server.

Sends one long prompt three times: cold, warm (from LMCache's memory tier, L1), and again after
enough filler prompts to push it out of L1, so the third answer must come from L2 (HyperCache).
Prints the time to the first token of each, and whether all three answers match. Run it against a
vLLM server started with LMCache and the HyperCache connector, at temperature 0. Not run by CI,
because it needs a GPU and a model.

    python3 scripts/e2e/lmcache_vllm_e2e.py --fillers 40
    python3 scripts/e2e/lmcache_vllm_e2e.py --url http://127.0.0.1:8000 --model my/model --fillers 40

Exits 1 when the answers differ.
"""

import argparse
import json
import sys
import time
import urllib.request

PROMPT_ITEMS = 380
FILLER_RECORDS = 800


def ask(url, model, prompt, max_tokens=24):
    """Streams one completion. Returns (seconds to the first token, full text)."""
    body = json.dumps(
        {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True}
    ).encode()
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    start = time.time()
    first_token = None
    text = ""
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            piece = json.loads(line[len("data:") :])["choices"][0]["text"]
            if piece and first_token is None:
                first_token = time.time() - start
            text += piece
    return first_token, text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:18111", help="vLLM server base URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--fillers", type=int, required=True, help="filler prompts sent to push the prompt out of L1")
    parser.add_argument("--settle-s", type=float, default=20, help="pause so L1 eviction gets below its watermark")
    args = parser.parse_args()
    url = args.url.rstrip("/") + "/v1/completions"

    base = " ".join(
        f"Item {i} in the inventory has the serial code {i * 7919 % 10007} and is stored on shelf {i % 37}."
        for i in range(PROMPT_ITEMS)
    )
    prompt = "A: " + base + "\nSummary:"

    results = {"cold": ask(url, args.model, prompt), "warm": ask(url, args.model, prompt)}
    for n in range(args.fillers):
        filler = " ".join(f"Record {n}-{i} has value {i * n % 991}." for i in range(FILLER_RECORDS))
        ask(url, args.model, f"Filler {n}: " + filler)
    time.sleep(args.settle_s)
    results["after_l1_flush"] = ask(url, args.model, prompt)

    for name, (ttft, text) in results.items():
        print(name, f"ttft={ttft:.3f}s", repr(text[:70]))
    identical = len({text for _, text in results.values()}) == 1
    print("identical_output", identical)
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
