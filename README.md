# Super-fast streaming inference for `ehzawad/ec-SFT-qwen25-7b-lora`

Drop-in replacement for the model card's `transformers` + PEFT + `TextIteratorStreamer`
code, which is single-request, runs the LoRA **unmerged** (extra matmuls every
token), and has no KV-cache paging or continuous batching.

This uses **vLLM** (PagedAttention + continuous batching) with the adapter
**merged** into the base, **prefix caching**, and **SSE token streaming** over
the OpenAI-compatible API.

> On **Colab**, use `Stage1_v5_vLLM_Colab.ipynb` instead — it serves *dynamic
> LoRA* (no 15 GB merge to the throwaway disk) and pins the install to Colab's
> CUDA driver. The files below are for a **persistent GPU host**.

## Topology (persistent host)

- **GPU host** (L4 / L40S / H100 / A100): runs `merge_adapter.py` once, then `serve.sh`.
- **Local (your Mac)**: runs `chat_stream.py` only — no GPU, no torch.

## Quickstart

On the GPU host:
```bash
pip install -r requirements-gpu.txt
python merge_adapter.py                 # downloads adapter + base, merges -> ./ec-qwen25-7b-merged
bash serve.sh                           # vLLM OpenAI server on :8000  (bf16)
# L4 (Ada) — faster + frees VRAM for KV:  QUANT=fp8 bash serve.sh
```

Locally:
```bash
pip install -r requirements-client.txt
BASE_URL=http://<gpu-host>:8000/v1 python chat_stream.py
```

## Your two GPUs (reconciled with a Codex council review)

| GPU  | Arch   | Weights        | "Larger KV" lever                  | Command                                              |
|------|--------|----------------|------------------------------------|------------------------------------------------------|
| L4   | Ada    | bf16 *or* FP8  | FP8 weights free ~7 GB **+** FP8 KV | `QUANT=fp8 bash serve.sh`, add `KV_CACHE_DTYPE=fp8`   |
| A100 | Ampere | bf16 only      | raw VRAM (40/80 GB) **+** FP8 KV    | `bash serve.sh`, raise `MAX_LEN`, add `KV_CACHE_DTYPE=fp8` |

**Two FP8 knobs that are NOT the same flag** (this was a real bug in the first draft):
- `QUANT=fp8` → quantizes **weights/activations**. Ada/Hopper only for a real
  speedup (A100 runs FP8 weight-only via Marlin = no compute win). Prefer
  `QUANT=fp8_per_block` if your vLLM supports it (more robust to outlier tensors
  for this Bengali adapter); plain `fp8` is the safe fallback.
- `KV_CACHE_DTYPE=fp8` → quantizes the **KV cache** (`--kv-cache-dtype fp8`).
  This is the actual "larger KV / longer context" lever: ~2× KV capacity, and it
  works on **both** L4 and A100. `QUANT=fp8` does **not** turn this on.

**Fidelity caveat (this is a recall-sensitive EC/NID model):** FP8 is not free.
Treat both FP8 weights and especially **uncalibrated FP8 KV** as something to A/B
against bf16 on a small Bengali/Banglish eval before trusting in production — KV
quantization error shows up most in long-context exact recall (dates, NID digits,
form numbers, fees). Keep the bf16 merged checkpoint as your reference.

## KV-cache math (Qwen2.5-7B, so you can size batches)

`28 layers × (K+V) × 4 kv-heads × 128 head-dim × 2 B = 56 KiB / token` (bf16).
- One full 8192-token sequence ≈ **0.44 GiB** bf16, **0.22 GiB** FP8 KV.
- 16384 ctx ≈ 0.875 GiB; 32768 ctx ≈ 1.75 GiB per sequence.
- After ~15 GB of bf16 weights, L4 (24 GB) has only a few GiB for KV → FP8
  weights and/or FP8 KV are what buy you real batch size there. A100 has plenty.

## Context length

`MAX_LEN` is a *cap*, not the source of KV size (KV is governed by free VRAM ×
`--gpu-memory-utilization` × `--kv-cache-dtype`). Qwen2.5-7B is native 32768, so
raising `MAX_LEN` up to 32768 on the A100 needs **no** RoPE/YaRN change. Beyond
32768 you'd add YaRN via `--hf-overrides`. The adapter trained at 8192, so treat
longer context as something to eval, not a free win.

## Speed knobs (in `serve.sh`)

- `--enable-prefix-caching` — on by default in current vLLM; kept for clarity.
  The repo's `system_prompt.txt` is ~1300 tokens, so caching that shared prefix
  across requests is a real win. (If your deployment overrides it with a
  one-liner, the benefit shifts to repeated multi-turn history instead.)
- `QUANT` / `KV_CACHE_DTYPE` — see the FP8 table above.
- speculative decoding (commented example) — single-stream latency.

## Files

- `Stage1_v5_vLLM_Colab.ipynb` — **Colab** path: dynamic LoRA, driver-matched install, streaming.
- `merge_adapter.py` — fold LoRA into base, write vLLM-ready checkpoint + tokenizer + system prompt.
- `serve.sh` — vLLM launch; Option A = merged (default, fastest), Option B = dynamic LoRA (`SERVE_MODE=lora`).
- `chat_stream.py` — local streaming client; injects system prompt, trims history, prints TTFT + tok/s.
