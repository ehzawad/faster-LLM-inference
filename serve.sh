#!/usr/bin/env bash
# Launch a super-fast vLLM OpenAI-compatible server for the EC/NID Qwen2.5-7B model.
#
# Run this on a CUDA GPU host (L4 / L40S / H100 ideal; A100 fine in bf16).
# Then stream from anywhere with:  python chat_stream.py
#
# Usage:
#   bash serve.sh                         # Option A, merged model at ./ec-qwen25-7b-merged
#   bash serve.sh /path/to/merged         # Option A with explicit merged dir
#   SERVE_MODE=lora bash serve.sh         # Option B: dynamic LoRA, no merge step
set -euo pipefail

MODEL_DIR="${1:-./ec-qwen25-7b-merged}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-ec-qwen}"

# ---- Speed knobs (tune to your GPU) -----------------------------------------
# max-model-len: training capped sequences at 8192. Keeping this tight (vs the
#   32k Qwen default) frees KV-cache memory for larger batches = more throughput.
MAX_LEN="${MAX_LEN:-8192}"
# Fraction of VRAM vLLM may claim for weights + KV cache.
GPU_UTIL="${GPU_UTIL:-0.90}"
# QUANT = WEIGHT/activation FP8. Real speedup ONLY on Ada/Hopper (L4, L40S, H100);
#   A100 (Ampere) runs FP8 weight-only via Marlin = no compute win, so leave empty.
#   Values: fp8 (safe legacy dynamic) | fp8_per_block (preferred if supported —
#   more robust to outlier tensors for this Bengali adapter) | fp8_per_tensor.
QUANT="${QUANT:-}"
# KV_CACHE_DTYPE = SEPARATE knob that quantizes the KV cache (NOT implied by QUANT).
#   This is the real "larger KV / longer context" lever (~2x capacity), works on
#   BOTH L4 and A100. fp8 KV is uncalibrated by default — A/B it on Bengali first.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"

COMMON_FLAGS=(
  --port "$PORT"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --enable-prefix-caching          # default-on in recent vLLM; reuses shared prefix (system prompt + repeated history)
)
[ -n "$QUANT" ]          && COMMON_FLAGS+=(--quantization "$QUANT")
[ -n "$KV_CACHE_DTYPE" ] && COMMON_FLAGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")

# Advanced single-stream latency: uncomment to add n-gram speculative decoding.
# Great when answers echo the (retrieved/system) prompt; harmless otherwise.
#   COMMON_FLAGS+=(--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4}')

echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo '?')"

if [ "${SERVE_MODE:-merged}" = "lora" ]; then
  # ---- Option B: serve base + adapter dynamically (skip merge_adapter.py) ----
  # Slightly slower per token than merged, but no merge/disk step and lets you
  # add more adapters later. The client must pass model="ec-lora".
  ADAPTER_ID="${ADAPTER_ID:-ehzawad/ec-SFT-qwen25-7b-lora}"
  BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
  echo ">> Option B: dynamic LoRA  base=$BASE_MODEL  adapter=$ADAPTER_ID"
  exec vllm serve "$BASE_MODEL" \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules "ec-lora=$ADAPTER_ID" \
    --tokenizer "$ADAPTER_ID" \
    "${COMMON_FLAGS[@]}"
else
  # ---- Option A (recommended, fastest): serve the merged checkpoint ----------
  echo ">> Option A: merged model  dir=$MODEL_DIR  quant=${QUANT:-none}"
  exec vllm serve "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    "${COMMON_FLAGS[@]}"
fi
