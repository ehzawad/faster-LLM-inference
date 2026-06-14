# Fast inference + production deploy — `ehzawad/ec-SFT-qwen25-7b-lora`

Making a **Bengali / Banglish Bangladesh Election-Commission / National-ID assistant**
(Qwen2.5-7B + a PEFT LoRA) genuinely *fast*, then shipping it as a real public service.

This README is the full story: what we tried, what the numbers actually said, the decisions we
made, and how to reproduce/deploy it.

---

## TL;DR

- **Model:** `Qwen/Qwen2.5-7B-Instruct` + the public LoRA `ehzawad/ec-SFT-qwen25-7b-lora`, **merged** and run as a single GGUF.
- **Engine:** **llama.cpp** (`llama-server`) — fastest single-stream path we found.
- **Live deploy:** **Modal**, scale-to-zero, A100-40GB, `Q5_K_M` GGUF, **greedy (temp 0)**, with an embedded Bengali chat UI.
- **Single-user speed:** **~95 tok/s** (up from ~25 on the naive `transformers` path).
- **Cost:** **$0 when idle** (scale-to-zero); `modal app stop` = fully off.
- **Public URL:** served at `https://<id>.modal.run` (printed by `modal deploy`).

```
                  Modal (scale-to-zero, A100)
 user ──https──▶ FastAPI chat page  ──localhost──▶ llama-server (Q5_K_M GGUF, --flash-attn on)
                 (injects trained system prompt, temp 0)
```

---

## The journey — what we tried (and what each taught us)

| Attempt | Result | Lesson |
|---|---|---|
| **vLLM on Colab** | ❌ failed | Free T4 (16 GB) can't hold 7B bf16; the server pattern adds a "Connection refused" failure class on Colab. |
| **`transformers` + PEFT (model card code)** | slow, single-request | Runs the LoRA *unmerged* (extra matmuls/token), no paging/batching, Python-thread streaming. |
| **Unsloth (in-process, 4-bit→bf16)** | ✅ ~25 tok/s on A100 | In-process kills the server problem; but HF `generate()` + `TextStreamer` has per-token Python overhead that caps an A100 far below its potential. |
| **llama.cpp GGUF (in-process / `llama-server`)** | ✅ **~100 tok/s solo** | C++ decode loop, no per-token Python, paged KV, real continuous batching. **Winner.** |
| **Modal (production)** | ✅ live, scale-to-zero | Public URL, $0 idle, 30-concurrent capable, one-line quant swaps. |

> Colab note: Colab ships **Python 3.12 + torch 2.8**, but the latest vLLM/llama-cpp wheels pin other
> torch/CUDA versions — naive `pip install` reinstalls torch and breaks CUDA. The fix is matching the
> wheel to the driver (`uv ... --torch-backend=auto`, or llama-cpp-python's `--index-url .../whl/cu124`).

---

## Key findings (the empirical data)

All measured on a single **A100-40GB**, greedy (temp 0), real Bengali EC/NID prompts.

### 1. Quantization: speed vs quality (single-user)

| Quant | Solo tok/s | Size | Quality (vs Q8_0, temp 0) |
|---|---:|---:|---|
| Q8_0 | 82 | 8.1 GB | near-lossless (reference) |
| Q6_K | ~95–100 | 6.3 GB | near-lossless |
| **Q5_K_M (live)** | **~93–110** | 5.4 GB | byte-identical on tested facts |
| Q4_K_M | 101 | 4.7 GB | byte-identical on tested facts |

- **At temp 0, Q4/Q5/Q8 produced *byte-identical* Bengali answers** on the fee / document-list / smart-card prompts — quantization didn't flip a single token on high-confidence facts.
- **~130 tok/s is *not* reachable on an A100** (Q4 caps ~101; the k-quant dequant overhead eats the bandwidth saving). 130+ needs an H100.
- **Chosen: `Q5_K_M`** — the speed sweet spot with negligible quality risk. Q4's extra ~8% (~0.2 s/answer) isn't worth being the lossiest quant for a recall-sensitive bot.

### 2. Concurrency: the regime flips the winner

| Config | Q8_0 | Q5_K_M |
|---|---:|---:|
| Solo (1 user) | 82 tok/s | ~110 tok/s |
| **30 concurrent — aggregate** | **414 tok/s** | 273 tok/s |
| 30 concurrent — per user | ~14 tok/s | ~9 tok/s |
| Peak VRAM | 21.3 GB (52%) | 18.9 GB (46%) |

- **Solo is memory-bandwidth-bound** → smaller quant (Q5) wins.
- **Heavy batching is compute-bound** → Q8's cheaper dequant wins; per-user speed is a *batching* effect, not a quant one.
- **30/30 concurrent succeeded** on one A100 using **52% of VRAM** — huge headroom.

### 3. Things that *don't* make it faster (myths busted)

- **Bigger KV cache ≠ faster.** It's a *capacity* lever (longer context / more concurrency). Decode is bandwidth-bound; a bigger cache only *adds* read traffic per token.
- **Free VRAM ≠ faster.** ~20 GB sat idle at 30-concurrent; it can't be converted into tok/s.
- **Speculative decoding** (Qwen2.5-0.5B draft): measured **0.65× — 35% *slower***. A generic draft has low acceptance on the fine-tuned Bengali style, so verification overhead exceeds the savings. *Lossless* at temp 0, but useless here. Discarded.

### 4. The real levers

- **Engine** (`transformers` → llama.cpp): the biggest win (~4×).
- **Quant** (Q8 → Q5): ~+15–34% solo.
- **`--parallel`** (not KV/quant) for "per-user feels slow under load" — trades max concurrency for active-stream speed.
- **temperature 0** (greedy): most deterministic / most factual — correct for a recall-sensitive bot.

---

## Deploy it yourself (Modal — the live path)

Everything is in [`deploy/modal_app.py`](deploy/modal_app.py). One image, scale-to-zero, public URL.

```bash
python3 -m venv .venv && .venv/bin/pip install modal
.venv/bin/modal token new                                   # one-time Modal auth

# 1) build the GGUF once (merges the public LoRA on an A100, quantizes onto a Volume)
.venv/bin/modal run   deploy/modal_app.py::build_gguf

# 2) deploy the public chat service  ->  prints https://<id>.modal.run
.venv/bin/modal deploy deploy/modal_app.py
```

Control & cost:
```bash
.venv/bin/modal app stop ec-nid-chat     # COMPLETELY off -> $0 (redeploy to turn back on)
```
- **Idle = $0** (scale-to-zero after 120 s); you pay per active GPU-second only.
- A100-40GB ≈ $2.10/hr *while serving*; first request after idle cold-starts ~30–60 s.

Built-in test/benchmark functions:
```bash
.venv/bin/modal run deploy/modal_app.py::loadtest        # 30-concurrent throughput + VRAM
.venv/bin/modal run deploy/modal_app.py::compare_quants  # Q4/Q5/Q8 side-by-side speed + Bengali answers
.venv/bin/modal run deploy/modal_app.py::spectest        # speculative-decoding A/B (it loses; kept for evidence)
```

### How it works
- **`build_gguf`** (A100): downloads the public base + LoRA, merges on GPU, converts to GGUF, k-quantizes, saves to a Modal **Volume** (persists across runs). The trained tokenizer/chat-template + `system_prompt.txt` are carried so prompts render exactly as in training.
- **`Server`** (`@app.cls`, scale-to-zero): on cold start launches `llama-server` on localhost (`--flash-attn on`, full GPU offload), then a FastAPI app serves the chat page at `/` and proxies `/chat` (injecting the system prompt) with SSE streaming. Hardened: input validation, per-IP rate limit, in-flight lock, upstream error handling, llama-server liveness check, prominent NID disclaimer.

> CUDA gotcha solved here: Modal builds images **without a GPU**, so compiling llama.cpp with CUDA needs the
> toolkit's **stub `libcuda`** at link time (the real driver is present at runtime). `torch` is pinned to
> `cu124` to match the CUDA 12.4 image. Also note `--flash-attn` now takes a value (`on`), not a bare flag.

---

## Other deploy paths

- **Persistent GPU VM** — [`deploy/docker-compose.yml`](deploy/docker-compose.yml): `llama-server` (private) + **Open WebUI** (public) + **Caddy** (TLS). See [`deploy/README.md`](deploy/README.md).
- **RunPod Pod** — [`deploy/RUNPOD.md`](deploy/RUNPOD.md) + [`deploy/runpod-entrypoint.sh`](deploy/runpod-entrypoint.sh): build-on-pod from the public adapter, network-volume cached.
- **Standalone GPU host** (no Docker) — `merge_adapter.py` → `serve.sh` (vLLM) → `chat_stream.py` (streaming client). The original `transformers`/PEFT path this project replaced.

---

## Production caveats (honest notes)

- **The real fidelity lever is retrieval, not the quant.** For exact, time-sensitive facts (fees, deadlines), pair the model with retrieval over official sources (`services.nidw.gov.bd`) rather than trusting static weights — the model card says the same. Q-level only affects whether the *recalled* answer drifts; it doesn't make stale facts current.
- **Before trusting a lighter quant (e.g. Q4) in production**, run a ~50–120 prompt Bengali/Banglish A/B (rare fees, corrections, Banglish spellings, numerals, adversarial "are you sure?") graded against source truth — not just byte-match on common prompts.
- **Security for a public gov bot:** keep inference private (only the frontend public), add per-IP rate limiting (Cloudflare/WAF), don't log raw prompts (users paste NID numbers), and keep the disclaimer prominent.
- Methodology and quant/deploy decisions were cross-checked with a multi-agent review at each step.

---

## Repo layout

```
deploy/
  modal_app.py          # LIVE: Modal scale-to-zero llama.cpp service + chat UI + test fns
  docker-compose.yml    # persistent GPU-VM stack (llama-server + Open WebUI + Caddy)
  Caddyfile  .env.example  README.md  RUNPOD.md  runpod-entrypoint.sh
merge_adapter.py        # merge LoRA -> vLLM-ready checkpoint (standalone host path)
serve.sh                # vLLM OpenAI server (standalone host path)
chat_stream.py          # local streaming client
requirements-*.txt
```
