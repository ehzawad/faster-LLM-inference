#!/usr/bin/env bash
set -Eeuo pipefail

# RunPod A100 entrypoint for EC/NID Qwen2.5-7B llama.cpp serving.
# First boot: build Q6_K GGUF from public HF base + public LoRA onto the mounted volume.
# Later boots: reuse the persisted GGUF and start llama-server immediately.

MODEL_DIR="${MODEL_DIR:-/workspace/ec-nid/models}"
WORK_DIR="${WORK_DIR:-/workspace/ec-nid/build}"
GGUF_FILE="${GGUF_FILE:-${MODEL_DIR}/ec-qwen25-7b.Q6_K.gguf}"
BASE_REPO="${BASE_REPO:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER_REPO="${ADAPTER_REPO:-ehzawad/ec-SFT-qwen25-7b-lora}"
BASE_REVISION="${BASE_REVISION:-}"
ADAPTER_REVISION="${ADAPTER_REVISION:-}"
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"

MODEL_ALIAS="${MODEL_ALIAS:-ec-nid-qwen25-7b}"
START_OPEN_WEBUI="${START_OPEN_WEBUI:-false}"
if [[ -z "${LLAMA_HOST:-}" ]]; then
  if [[ "${START_OPEN_WEBUI}" == "true" ]]; then
    LLAMA_HOST="127.0.0.1"
  else
    LLAMA_HOST="${HOST:-0.0.0.0}"
  fi
fi
if [[ -z "${LLAMA_PORT:-}" ]]; then
  if [[ "${START_OPEN_WEBUI}" == "true" ]]; then
    LLAMA_PORT="8000"
  else
    LLAMA_PORT="${PORT:-8000}"
  fi
fi
WEBUI_HOST="${WEBUI_HOST:-${HOST:-0.0.0.0}}"
WEBUI_PORT="${WEBUI_PORT:-${PORT:-8080}}"
WEBUI_DATA_DIR="${WEBUI_DATA_DIR:-/workspace/ec-nid/open-webui}"
OPEN_WEBUI_VERSION="${OPEN_WEBUI_VERSION:-latest}"
# 32 slots x 8192 tokens. With explicit --parallel, this must be the total KV budget.
CTX_SIZE="${CTX_SIZE:-262144}"
PARALLEL="${PARALLEL:-32}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
CLEAN_INTERMEDIATES="${CLEAN_INTERMEDIATES:-true}"
API_KEY="${INFERENCE_API_KEY:-${API_KEY:-}}"
REQUIRE_API_KEY="${REQUIRE_API_KEY:-true}"

LLAMA_DIR="${WORK_DIR}/llama.cpp"
MERGED_DIR="${WORK_DIR}/merged-qwen25-7b-ec"
BF16_GGUF="${WORK_DIR}/ec-qwen25-7b.bf16.gguf"
LOCK_DIR="${MODEL_DIR}/.build.lock"

log() {
  printf '[runpod-entrypoint] %s\n' "$*" >&2
}

require_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi not found. Use a CUDA RunPod image with GPU attached."
    exit 1
  fi

  nvidia-smi -L >&2
  local gpu_name
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
  if [[ -z "${gpu_name}" ]]; then
    log "ERROR: no NVIDIA GPU reported by nvidia-smi."
    exit 1
  fi
  if [[ "${gpu_name}" != *A100* ]]; then
    log "WARNING: GPU is '${gpu_name}', not A100. For ~30 users at ctx ${CTX_SIZE}, reduce PARALLEL or CTX_SIZE unless VRAM is comparable."
  fi
}

install_system_deps() {
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs build-essential cmake ninja-build \
      python3 python3-pip python3-venv
    git lfs install --skip-repo || true
  else
    log "apt-get not found; assuming system build dependencies already exist."
  fi
}

install_python_deps() {
  export PIP_BREAK_SYSTEM_PACKAGES=1
  python3 -m pip install --upgrade pip
  python3 -m pip install --upgrade \
    "transformers>=4.45" \
    "peft>=0.13" \
    "accelerate>=0.34" \
    "huggingface_hub>=0.25" \
    safetensors sentencepiece protobuf numpy

  python3 - <<'PY'
import importlib.util
missing = [name for name in ["torch", "transformers", "peft"] if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing required Python packages: {missing}. Use a RunPod PyTorch CUDA image, or bake torch into a custom image.")
PY
}

build_llama_cpp() {
  if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
    git clone "${LLAMA_CPP_REPO}" "${LLAMA_DIR}"
  fi

  git -C "${LLAMA_DIR}" fetch --tags --prune
  git -C "${LLAMA_DIR}" checkout "${LLAMA_CPP_REF}"

  if [[ ! -x "${LLAMA_DIR}/build/bin/llama-server" || ! -x "${LLAMA_DIR}/build/bin/llama-quantize" ]]; then
    cmake -S "${LLAMA_DIR}" -B "${LLAMA_DIR}/build" \
      -G Ninja \
      -DGGML_CUDA=ON \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build "${LLAMA_DIR}/build" --config Release -j "$(nproc)"
  fi
}

merge_lora() {
  log "Merging ${ADAPTER_REPO} into ${BASE_REPO}"
  export BASE_REPO ADAPTER_REPO BASE_REVISION ADAPTER_REVISION MERGED_DIR
  python3 - <<'PY'
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_repo = os.environ["BASE_REPO"]
adapter_repo = os.environ["ADAPTER_REPO"]
merged_dir = os.environ["MERGED_DIR"]
base_revision = os.environ.get("BASE_REVISION") or None
adapter_revision = os.environ.get("ADAPTER_REVISION") or None

model = AutoModelForCausalLM.from_pretrained(
    base_repo,
    revision=base_revision,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(model, adapter_repo, revision=adapter_revision)
model = model.merge_and_unload()
model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="4GB")

tokenizer = AutoTokenizer.from_pretrained(base_repo, revision=base_revision, use_fast=True)
tokenizer.save_pretrained(merged_dir)
PY
}

write_manifest() {
  local manifest="${GGUF_FILE}.manifest"
  {
    printf 'gguf_file=%s\n' "${GGUF_FILE}"
    printf 'base_repo=%s\n' "${BASE_REPO}"
    printf 'base_revision=%s\n' "${BASE_REVISION:-unpinned}"
    printf 'adapter_repo=%s\n' "${ADAPTER_REPO}"
    printf 'adapter_revision=%s\n' "${ADAPTER_REVISION:-unpinned}"
    printf 'llama_cpp_ref=%s\n' "$(git -C "${LLAMA_DIR}" rev-parse HEAD)"
    printf 'quantization=Q6_K\n'
    printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sha256sum "${GGUF_FILE}" | awk '{print "sha256=" $1}'
  } > "${manifest}"
  sha256sum "${GGUF_FILE}" > "${GGUF_FILE}.sha256"
}

build_gguf_if_missing() {
  mkdir -p "${MODEL_DIR}" "${WORK_DIR}"

  if [[ -s "${GGUF_FILE}" ]]; then
    log "Found existing GGUF: ${GGUF_FILE}"
    return
  fi

  if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    log "Another process is building the GGUF. Waiting for ${GGUF_FILE}..."
    while [[ ! -s "${GGUF_FILE}" ]]; do
      sleep 10
    done
    return
  fi
  trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

  if [[ -s "${GGUF_FILE}" ]]; then
    rmdir "${LOCK_DIR}" 2>/dev/null || true
    trap - EXIT
    return
  fi

  log "GGUF missing. Building once onto the mounted volume."
  install_system_deps
  install_python_deps
  build_llama_cpp
  merge_lora

  log "Converting merged model to bf16 GGUF"
  python3 "${LLAMA_DIR}/convert_hf_to_gguf.py" "${MERGED_DIR}" \
    --outfile "${BF16_GGUF}" \
    --outtype bf16

  log "Quantizing bf16 GGUF to Q6_K"
  "${LLAMA_DIR}/build/bin/llama-quantize" "${BF16_GGUF}" "${GGUF_FILE}.tmp" Q6_K
  mv "${GGUF_FILE}.tmp" "${GGUF_FILE}"
  write_manifest

  if [[ "${CLEAN_INTERMEDIATES}" == "true" ]]; then
    rm -rf "${MERGED_DIR}" "${BF16_GGUF}"
  fi

  rmdir "${LOCK_DIR}" 2>/dev/null || true
  trap - EXIT
}

build_llama_args() {
  local server="${LLAMA_DIR}/build/bin/llama-server"
  if [[ ! -x "${server}" ]]; then
    log "llama-server binary missing; building llama.cpp."
    install_system_deps
    build_llama_cpp
  fi

  LLAMA_SERVER="${server}"
  LLAMA_ARGS=(
    -m "${GGUF_FILE}"
    --alias "${MODEL_ALIAS}"
    --host "${LLAMA_HOST}"
    --port "${LLAMA_PORT}"
    --ctx-size "${CTX_SIZE}"
    --n-gpu-layers "${N_GPU_LAYERS}"
    --flash-attn
    --parallel "${PARALLEL}"
    --cont-batching
    --metrics
  )

  if [[ -n "${API_KEY}" ]]; then
    LLAMA_ARGS+=(--api-key "${API_KEY}")
  elif [[ "${REQUIRE_API_KEY}" == "true" ]]; then
    log "ERROR: INFERENCE_API_KEY/API_KEY is empty. Refusing to start an unauthenticated public inference server."
    exit 1
  else
    log "WARNING: INFERENCE_API_KEY/API_KEY is empty; llama-server API will be unauthenticated."
  fi
}

start_llama_server() {
  build_llama_args
  log "Starting llama-server on ${LLAMA_HOST}:${LLAMA_PORT} with --parallel ${PARALLEL} --cont-batching"
  exec "${LLAMA_SERVER}" "${LLAMA_ARGS[@]}" "$@"
}

start_llama_server_background() {
  build_llama_args
  log "Starting llama-server on ${LLAMA_HOST}:${LLAMA_PORT} with --parallel ${PARALLEL} --cont-batching"
  "${LLAMA_SERVER}" "${LLAMA_ARGS[@]}" "$@" &
  LLAMA_PID=$!
  trap 'kill "${LLAMA_PID}" 2>/dev/null || true' EXIT
}

wait_for_llama() {
  local auth_args=()
  if [[ -n "${API_KEY}" ]]; then
    auth_args=(-H "Authorization: Bearer ${API_KEY}")
  fi

  for _ in $(seq 1 180); do
    if curl -fsS "${auth_args[@]}" "http://${LLAMA_HOST}:${LLAMA_PORT}/v1/models" >/dev/null 2>&1; then
      return
    fi
    if ! kill -0 "${LLAMA_PID}" 2>/dev/null; then
      log "ERROR: llama-server exited before Open WebUI could connect."
      wait "${LLAMA_PID}" || true
      exit 1
    fi
    sleep 2
  done

  log "ERROR: llama-server did not become ready on ${LLAMA_HOST}:${LLAMA_PORT}."
  exit 1
}

install_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
}

start_open_webui() {
  install_uv
  mkdir -p "${WEBUI_DATA_DIR}"

  export DATA_DIR="${WEBUI_DATA_DIR}"
  export OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-http://${LLAMA_HOST}:${LLAMA_PORT}/v1}"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY}}"
  export DEFAULT_MODELS="${DEFAULT_MODELS:-${MODEL_ALIAS}}"
  export WEBUI_NAME="${WEBUI_NAME:-Bangladesh NID/EC Assistant}"
  if [[ -n "${WEBUI_SECRET_KEY:-}" ]]; then
    export WEBUI_SECRET_KEY
  fi
  export WEBUI_AUTH="${WEBUI_AUTH:-true}"
  export ENABLE_SIGNUP="${ENABLE_SIGNUP:-false}"
  export DEFAULT_USER_ROLE="${DEFAULT_USER_ROLE:-pending}"
  export ENABLE_OPENAI_API="${ENABLE_OPENAI_API:-true}"
  export DEFAULT_MODEL_PARAMS="${DEFAULT_MODEL_PARAMS:-{\"max_tokens\":512,\"temperature\":0.2,\"top_p\":0.9}}"
  export USER_PERMISSIONS_CHAT_FILE_UPLOAD="${USER_PERMISSIONS_CHAT_FILE_UPLOAD:-false}"
  export USER_PERMISSIONS_CHAT_WEB_UPLOAD="${USER_PERMISSIONS_CHAT_WEB_UPLOAD:-false}"
  export USER_PERMISSIONS_CHAT_SYSTEM_PROMPT="${USER_PERMISSIONS_CHAT_SYSTEM_PROMPT:-false}"
  export USER_PERMISSIONS_CHAT_PARAMS="${USER_PERMISSIONS_CHAT_PARAMS:-false}"

  log "Starting Open WebUI on ${WEBUI_HOST}:${WEBUI_PORT}; llama-server stays on ${LLAMA_HOST}:${LLAMA_PORT}"
  exec uvx --python 3.11 "open-webui@${OPEN_WEBUI_VERSION}" serve --host "${WEBUI_HOST}" --port "${WEBUI_PORT}"
}

start_open_webui_stack() {
  start_llama_server_background "$@"
  wait_for_llama
  start_open_webui
}

require_gpu
build_gguf_if_missing
if [[ "${START_OPEN_WEBUI}" == "true" ]]; then
  start_open_webui_stack "$@"
else
  start_llama_server "$@"
fi
