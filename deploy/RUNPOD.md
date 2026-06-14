# RunPod deployment reconciliation - EC/NID Qwen2.5-7B llama.cpp

## Recommendation

Use a **RunPod GPU Pod on A100 with a network volume**. Do the **first-boot rebuild on the pod** from the public Hugging Face sources, persist the finished Q6_K GGUF on the volume, then reuse it on every later boot.

That is the best current path because the Q6_K GGUF exists only inside Colab, not on the Mac. Rebuilding avoids a fragile Colab-to-Mac-to-host transfer and keeps the user's Apple Silicon Mac out of CUDA/model-build work entirely.

After the GGUF exists on the network volume, there is nothing material between "rebuilt on pod" and "downloaded prebuilt artifact" for steady-state serving. Both become: mount the volume, find `/workspace/ec-nid/models/ec-qwen25-7b.Q6_K.gguf`, start `llama-server`.

## Pod vs Serverless

Use **Pods**, not Serverless, for this pilot:

- The user wants a public browser chat that can stay warm for ~30 concurrent users.
- `llama-server` is a long-running HTTP process; a Pod maps cleanly to that.
- Serverless is better for request/handler APIs and bursty workloads. It can work with active workers, but then it is no longer materially simpler or cheaper for an always-warm public chat, and it adds handler/container complexity.

Use an A100 Pod only while the demo is actually being used. A100 pricing is roughly $1-1.5/hr depending on SKU/market, so an always-on public link becomes real spend quickly. Stop the Pod when idle, and terminate it only after the GGUF is safely on a persistent/network volume or backed up.

## Compose vs RunPod Pod

`docker-compose.yml` is a good shape for a VM-style GPU host:

```text
Internet -> Caddy -> Open WebUI -> private llama-server
```

A normal RunPod Pod is a **single container**, not a VM running Docker Compose. Do not assume the pod can run the whole compose stack unless you deliberately build a Docker-in-Docker/supervisor image. For RunPod:

- Private smoke test only: expose `llama-server` on one HTTP port and use the RunPod proxy URL with `--api-key`.
- Public pilot: build a single custom image that runs `llama-server` on localhost plus Open WebUI as the exposed service.
- The existing compose remains the right artifact if the target is a GPU VM with Docker + NVIDIA Container Toolkit.

For "anyone can check it quickly", expose **Open WebUI**, not the built-in `llama-server` UI. The built-in UI is fine for a private smoke test, but it does not give you accounts, a polished citizen-facing chat surface, or a good place for the required safety banner. If you temporarily expose `llama-server` alone, treat it as an API-only smoke test and require `--api-key`.

Image choice:

- If the GGUF already exists on the volume, `ghcr.io/ggml-org/llama.cpp:server-cuda` is the cleanest serving image.
- If the pod must rebuild from LoRA, use a RunPod PyTorch/CUDA image or a custom image that includes PyTorch. The official llama.cpp serving image is not the right base for the PEFT merge step.
- Long-term cleanest: custom image with Python deps and llama.cpp pinned/baked in, plus the network volume for only model artifacts.

RunPod's HTTP proxy gives the exposed port a public HTTPS URL. It is not an authentication layer. If a port is exposed, anyone who has the URL can reach that service, so auth must be enforced by Open WebUI accounts, a frontend/session layer, or `llama-server --api-key`.

## Model delivery comparison

| Path | First boot | Reproducibility | Operational risk | When to use |
|---|---:|---|---|---|
| A. Rebuild on pod | About 10 minutes plus base-model download/build time | Good if base revision, adapter revision, llama.cpp commit, and pip deps are pinned | More moving parts on first boot: Python deps, HF availability, conversion, quantization | Best now because source repos are public and the GGUF is not on the Mac |
| B. Prebuild + host GGUF | Usually just a ~6 GB download before serving | Best if artifact has a SHA256 and immutable HF/volume path | Need to get the Colab artifact into private HF, S3, or a RunPod volume first | Best after the first successful pod build, or for production cold starts |

With a network volume, the build/download should happen only once. Subsequent cold starts are dominated by pod scheduling, container startup, and model load, not conversion.

## Entrypoint

Use `runpod-entrypoint.sh` as the start command or bake it into a custom image. For the recommended public
Open WebUI launch, set `START_OPEN_WEBUI=true`; the script starts `llama-server` on localhost and Open WebUI
on the exposed port. It does this:

1. Fail fast if no NVIDIA GPU is visible.
2. Warn if the GPU is not A100-class while `PARALLEL=32`.
3. Check the mounted volume for the final Q6_K GGUF.
4. If missing, install deps, build CUDA llama.cpp, merge public LoRA into public base, convert to bf16 GGUF, quantize Q6_K, and write a manifest/SHA256.
5. Start `llama-server` with `--parallel` and `--cont-batching`.
6. If `START_OPEN_WEBUI=true`, wait for `llama-server`, then start Open WebUI on `WEBUI_PORT`.

Expected image: a CUDA/PyTorch RunPod image, not the user's Mac. Torch should already be present. The script installs `transformers`, `peft`, `accelerate`, `huggingface_hub`, `safetensors`, `sentencepiece`, `protobuf`, `numpy`, plus system build deps if `apt-get` is available.

The Colab-specific `torchao`/`peft` probe workaround is not part of this path. On a clean RunPod PyTorch/CUDA image, use normal `peft` + `transformers`; pin versions if a later package release breaks compatibility.

Important env vars:

```bash
MODEL_DIR=/workspace/ec-nid/models
WORK_DIR=/workspace/ec-nid/build
GGUF_FILE=/workspace/ec-nid/models/ec-qwen25-7b.Q6_K.gguf
BASE_REPO=Qwen/Qwen2.5-7B-Instruct
ADAPTER_REPO=ehzawad/ec-SFT-qwen25-7b-lora
BASE_REVISION=                 # strongly recommended: pin before production
ADAPTER_REVISION=              # strongly recommended: pin before production
LLAMA_CPP_REF=                 # strongly recommended: pin before production
CTX_SIZE=262144              # 32 slots x 8192 tokens; do not leave this at 8192 for 32-way 8k chat
PARALLEL=32
INFERENCE_API_KEY=             # required for public exposure
WEBUI_SECRET_KEY=              # generate a separate openssl rand -hex 32 value
START_OPEN_WEBUI=true          # recommended public demo mode
WEBUI_PORT=8080                # expose this HTTP port in RunPod, not LLAMA_PORT
WEBUI_BANNERS=                 # bilingual disclaimer JSON from .env.example
```

## Mac vs user division of labor

Codex on the user's Mac can:

- Write and maintain deployment files.
- Install local CLI helpers such as `runpodctl` or call the RunPod REST API once an API key is provided.
- Create/update a pod template, environment variables, and start command.
- Poll logs and verify `https://<pod-id>-<port>.proxy.runpod.net`.
- Run HTTP smoke tests against `/v1/models` and a chat completion.

Codex on the Mac cannot:

- Run CUDA.
- Run this Q6_K model at the target speed.
- Build a CUDA llama.cpp binary for the RunPod host.
- Access/fund the user's RunPod account without credentials and spend approval.

Only the user can:

- Create/fund the RunPod account.
- Provide a RunPod API key or do console steps.
- Approve hourly A100 spend and network-volume storage.
- Decide whether the first public link uses Open WebUI accounts or a thinner custom frontend. Avoid anonymous raw `llama-server` for a public GPU link.

The Mac does **not** need to build llama.cpp. The pod builds CUDA llama.cpp, or a custom Linux/amd64 image can be built in CI.

## Public 30-user readiness

For ~30 concurrent users:

- Use A100 40 GB minimum; A100 80 GB is safer.
- Keep `--parallel 32 --ctx-size 262144 --cont-batching --flash-attn --n-gpu-layers 999`.
- Expect shared throughput under 30 active generations, not 30 users each getting single-user speed.
- Plan around roughly 10-20 tok/s per actively generating user if all ~30 are generating at once; idle chat sessions do not consume decode throughput. Long prompts will mainly show up as slower time-to-first-token.
- RunPod proxy URLs are public if the port is exposed. Treat the URL as internet-facing.

Move from llama.cpp to vLLM when sustained active generations, not just open browser sessions, push p95 user-visible speed below roughly 8 tok/s, p95 short-prompt TTFT is unacceptable, or `llama-server` deferred/queued requests keep growing. Add a second GPU/replica when vLLM still misses the target or you need redundancy.

Public safety checklist:

- Prefer Open WebUI accounts or a thin frontend over exposing raw `llama-server`.
- If raw `llama-server` is exposed, set `INFERENCE_API_KEY`; otherwise it is an open inference API.
- Add rate limiting before broad sharing. RunPod proxy alone is not an app-level per-IP limiter; use Cloudflare/WAF in front of a domain, a Cloudflare Worker/Tunnel, or app-level quotas.
- Disable or minimize prompt logging; citizen users may paste NID numbers.
- Keep the bilingual disclaimer and a time-bound-fact guardrail: fees, deadlines, eligibility, and official procedures must be verified against official EC/NID channels.
- Cap output tokens for cost and latency, e.g. 512-768 max output tokens and a modest per-user daily request cap for the pilot.

## What to ask the user for

To proceed end to end from the Mac, ask for:

1. RunPod API key.
2. Confirmation of A100 spend limit and whether interruptible/community is acceptable.
3. Permission to create a network volume, recommended 80-100 GB for build intermediates.
4. Public exposure choice:
   - Recommended: Open WebUI/custom frontend in front.
   - Private smoke only: raw `llama-server` proxy URL with API key.
5. Optional domain if they want Caddy/real DNS instead of the RunPod proxy URL.
