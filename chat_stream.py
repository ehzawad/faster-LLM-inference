#!/usr/bin/env python3
"""
Super-fast streaming chat client for the EC/NID Qwen2.5-7B vLLM server.

Talks to the OpenAI-compatible endpoint that serve.sh exposes. Token-by-token
SSE streaming, multi-turn history with token-budget trimming (Bengali tokenizes
to ~1 token/char, so history fills 8192 fast), the trained system prompt injected
automatically, and per-turn TTFT + tok/s.

This is the ONLY piece you run locally (e.g. on your Mac) — it needs no GPU.

    pip install openai huggingface_hub
    python chat_stream.py                          # server on localhost:8000
    BASE_URL=http://gpu-host:8000/v1 python chat_stream.py
    MODEL=ec-lora python chat_stream.py            # required for SERVE_MODE=lora

Commands inside the loop:  /reset  (clear history)   /exit   /tokens N
"""
import json
import os
import time
import urllib.request
from pathlib import Path

from openai import OpenAI

ADAPTER_ID = "ehzawad/ec-SFT-qwen25-7b-lora"
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# Keep prompt within the server's --max-model-len (override if you raised it).
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
SERVER_ROOT = BASE_URL.rsplit("/v1", 1)[0]  # /tokenize lives at the root, not under /v1


def load_system_prompt() -> str:
    """Use the trained system prompt: local merged dir first, else fetch from Hub."""
    local = Path(os.environ.get("MERGED_DIR", "./ec-qwen25-7b-merged")) / "system_prompt.txt"
    if local.is_file():
        return local.read_text(encoding="utf-8").strip()
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(ADAPTER_ID, "system_prompt.txt")).read_text("utf-8").strip()
    except Exception as e:  # noqa: BLE001
        print(f"[warn: could not load system_prompt.txt ({e}); proceeding without it]")
        return ""


def resolve_model(client: OpenAI) -> str:
    """Pick the served model id. In dynamic-LoRA mode /models lists the BASE model
    too, and it can come first — never silently fall through to it."""
    if os.environ.get("MODEL"):
        return os.environ["MODEL"]
    try:
        ids = [m.id for m in client.models.list().data]
    except Exception:  # noqa: BLE001
        return "ec-qwen"
    for preferred in ("ec-qwen", "ec-lora"):  # our merged / dynamic served names
        if preferred in ids:
            return preferred
    non_base = [i for i in ids if i != BASE_MODEL]  # avoid serving the raw base
    if non_base:
        return non_base[0]
    return ids[0] if ids else "ec-qwen"


def count_tokens(messages: list[dict], model: str) -> int | None:
    """Exact prompt token count via vLLM's /tokenize (no local tokenizer needed).
    Returns None if the endpoint is unavailable."""
    body = json.dumps(
        {"model": model, "messages": messages, "add_generation_prompt": True}
    ).encode()
    req = urllib.request.Request(
        f"{SERVER_ROOT}/tokenize", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read()).get("count")
    except Exception:  # noqa: BLE001
        return None


def build_messages(system_prompt: str, history: list[dict], q: str, model: str,
                   reserve: int) -> list[dict]:
    """Newest-first trim: drop oldest user/assistant pairs until the prompt fits
    MAX_MODEL_LEN minus the reservation for the reply."""
    budget = MAX_MODEL_LEN - reserve
    sys_msg = [{"role": "system", "content": system_prompt}] if system_prompt else []
    hist = list(history)
    while True:
        msgs = sys_msg + hist + [{"role": "user", "content": q}]
        n = count_tokens(msgs, model)
        if n is None:                       # /tokenize unavailable -> cap by turn count
            return sys_msg + history[-12:] + [{"role": "user", "content": q}]
        if n <= budget or not hist:
            if n > budget:
                print(f"[warn: prompt {n} tok > budget {budget}; sending anyway]")
            return msgs
        hist = hist[2:]                     # drop the oldest user+assistant pair


def main() -> None:
    client = OpenAI(api_key="EMPTY", base_url=BASE_URL)
    model = resolve_model(client)
    system_prompt = load_system_prompt()
    max_tokens = int(os.environ.get("MAX_TOKENS", "1024"))

    print(f"connected: {BASE_URL}  model={model}  (vLLM greedy, temperature=0)")
    print(f"max_model_len={MAX_MODEL_LEN}   commands: /reset  /exit  /tokens N\n")

    history: list[dict] = []
    while True:
        try:
            q = input("USER> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/exit", "/quit"):
            break
        if q == "/reset":
            history.clear()
            print("[cleared]")
            continue
        if q.startswith("/tokens"):
            parts = q.split()
            if len(parts) == 2 and parts[1].isdigit():
                max_tokens = int(parts[1])
                print(f"[max_tokens={max_tokens}]")
            else:
                print("[usage: /tokens 1024]")
            continue

        messages = build_messages(system_prompt, history, q, model, reserve=max_tokens + 64)

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,          # vLLM greedy, matches the model card's do_sample=False
            stream=True,
            stream_options={"include_usage": True},
        )

        print("BOT > ", end="", flush=True)
        chunks: list[str] = []
        t0 = time.perf_counter()
        ttft = None
        n_out = n_prompt = 0
        for event in stream:
            if event.usage is not None:
                n_out = event.usage.completion_tokens
                n_prompt = event.usage.prompt_tokens
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                print(delta, end="", flush=True)
                chunks.append(delta)
        dt = time.perf_counter() - t0

        reply = "".join(chunks).strip()
        tps = (n_out / dt) if (n_out and dt) else 0.0
        print(f"\n[ttft {1000 * (ttft or 0):.0f} ms | prompt {n_prompt} | "
              f"out {n_out} tok | {tps:.1f} tok/s]\n")

        if not reply:
            print("[WARN: 0 chars generated]")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()
