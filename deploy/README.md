# Product-grade deployment — EC/NID Qwen2.5-7B (llama.cpp)

Always-on public chat for the Bengali/Banglish EC/NID assistant, using **`llama-server`** (your choice)
to serve the **Q6_K GGUF** built from the public base + public LoRA and persisted on the target host.
Reconciled with a Codex-council review.

```
Internet ──▶ Caddy (TLS, 80/443) ──▶ Open WebUI (chat panel, login)
                                          └──▶ llama-server (inference, PRIVATE — no public port)
```

**Rule #1:** only the frontend is public. `llama-server` stays on the internal Docker network behind an
API key. Never expose the raw inference port or a model UI to the internet.

For RunPod Pods specifically, see [`RUNPOD.md`](./RUNPOD.md). A normal RunPod Pod is a single-container
runtime, so this compose stack is directly runnable on a GPU VM, but only a reference shape for RunPod
unless you build a custom multi-process image.

## 1. Get a GPU host (tuned for ~30 concurrent users)

**Recommended: RunPod GPU Pod, A100.** Why it fits "easily up + anyone can check it + 30 concurrent":
- One-click GPU pod; expose Open WebUI's HTTP port and RunPod hands you a **public HTTPS proxy URL**
  (`https://<pod-id>-8080.proxy.runpod.net`) — no domain/DNS needed just to let invited testers try it.
- A100 (40/80 GB) has the VRAM for **32 concurrent 8k slots** of KV cache (see sizing below).
- ~**$1.4-1.5/hr** for current A100 80GB RunPod SKUs; stop the pod when idle to save.

| Provider | GPU | Spin-up ease | Public URL | ~ cost | Fit for 30 concurrent |
|---|---|---|---|---:|---|
| **RunPod Pod** | **A100 80GB** | ★★★ one-click | built-in proxy URL | ~$1.4-1.5/hr | **best easy pick** |
| Lambda / Vast | A100 | ★★ manual net | bring your own | ~$1–2/hr | cheaper, more setup |
| AWS `g5.12xlarge` | 4×A10G | ★ full infra | Elastic IP + domain | ~$5.6/hr | overkill unless scaling |
| HF Inference Endpoints | A100 | ★★★ managed | managed URL (API) | ~$1,825/mo | great API, needs separate UI |

**GPU sizing for 30 concurrent (Q6_K + `--parallel 32`, ctx 8192):**
`~6.25 GB weights + 32 × 0.44 GB KV ≈ 20 GB` before runtime overhead → **A100 (40/80 GB) required.**
An L4 (24 GB) is too tight for 32 full 8k slots; use it only with `--parallel 8–12` or a smaller total
context budget.

Important llama.cpp detail: `--ctx-size 8192 --parallel 32` is not the right 32-user/8k launch shape.
Reserve the full KV budget:

```bash
--parallel 32 --ctx-size 262144 --cont-batching
```

**Throughput reality (one A100, llama.cpp):** single user ≈ 100 tok/s; with ~30 generating at once,
continuous batching shares the GPU → aggregate ~300–600 tok/s, so each active user sees ~10–20 tok/s.
That's fine for chat. If you need *every* user at full speed under constant 30-way load, that's the point
to move to **vLLM** (commented in the compose) or a second GPU.

## 2. Host setup
```bash
# Docker + NVIDIA Container Toolkit, then verify the GPU is visible inside containers:
docker run --rm --gpus all ubuntu nvidia-smi      # must print your GPU
```

## 3. Put the model on the host
For a VM-style deployment using this compose file, put the final GGUF at:
```
/srv/ec-nid/models/ec-qwen25-7b.Q6_K.gguf
```

For the current RunPod-first path, do not transfer the Colab-only GGUF through the Mac. Use
`runpod-entrypoint.sh` on an A100 Pod with a network volume and `START_OPEN_WEBUI=true`; it rebuilds the
GGUF once from:

- Base: `Qwen/Qwen2.5-7B-Instruct`
- LoRA: `ehzawad/ec-SFT-qwen25-7b-lora`

After that, the persisted volume makes later boots start from the existing Q6_K file.

Expose only `WEBUI_PORT=8080` in RunPod. Leave `LLAMA_PORT=8000` unexposed; in Open WebUI mode the script
binds llama-server to localhost and uses the API key internally.

## 4. Configure + launch
```bash
cd deploy
cp .env.example .env          # then edit: DOMAIN, INFERENCE_API_KEY (openssl rand -hex 32)
# point your domain's DNS A-record at this host FIRST (Caddy needs it for TLS)
docker compose up -d
docker compose logs -f caddy  # watch TLS provision
```
Open `https://<your-domain>` → create the **admin account** (first signup becomes admin; `ENABLE_SIGNUP=false`
keeps it closed after). You now have a public, TLS-secured chat panel.

## 5. Inject the trained system prompt (important)
`llama-server` does **not** auto-apply your `system_prompt.txt`. In Open WebUI:
**Workspace → Models → `ec-nid-qwen25-7b` → System Prompt** → paste the full ~1302-token `system_prompt.txt`,
and disable user system-prompt override. Without this, answers lose the trained EC/NID behavior.

## Scaling llama.cpp (your chosen server)
- `--parallel N` + `--cont-batching` (already set to 32) = N concurrent generation slots. Size `--ctx-size`
  for the total slot budget; 32 users at 8k means `--ctx-size 262144`.
- Good for the requested **~30 concurrent chat users** if shared throughput is acceptable. If you need every
  user to retain near single-stream speed under constant 30-way load, move to **vLLM** or add GPUs. The
  commented `vllm` block in `docker-compose.yml` shows the alternate shape, though llama.cpp remains the
  chosen server here.
- For more headroom on one box: raise `--parallel`, or run 2 GPUs / 2 replicas behind the proxy.

## Must-have before you go public (council checklist, condensed)
- [x] Inference private (no public port), API key held by the gateway — **done by this compose**
- [x] Real domain + TLS + security headers — **Caddy**
- [x] Login required, no open signup — **WEBUI_AUTH / ENABLE_SIGNUP=false**
- [x] Safety/PII disclaimer banner — **WEBUI_BANNERS in .env**
- [ ] **Per-IP rate limiting** — put Cloudflare/WAF in front (Caddy has no built-in limiter); cap bursts
- [ ] **Server-side token cap** — set `max_tokens ≤ 512–768` in the Open WebUI model params
- [ ] **No raw PII in logs** — don't log prompts/outputs; users may paste NID numbers (10/13/17-digit)
- [ ] **Time-bound-fact guardrail** — the system prompt must make it defer fees/deadlines to 105 / the portal
- [ ] **Health + auto-restart** — `restart: unless-stopped` is set; add an external uptime check + a synthetic canary chat
- [ ] **Metrics** — `llama-server` exposes `/metrics`; scrape with Prometheus, dashboard latency/TTFT/tok-s/errors
- [ ] **Disable risky features** in Open WebUI — no file upload, no web browsing, no user-set system prompt

## ⚠️ One caveat the council flagged (Open WebUI + PII)
Open WebUI **persists chat history** to its database (`/srv/ec-nid/open-webui`). For an authenticated pilot
that's fine. For a *fully public, anonymous* citizen service where people may paste NID numbers, that storage
is a privacy liability — either disable history retention, set short retention, or front it with a thin custom
chat UI that doesn't store transcripts. Decide this before opening it to the general public.

## Files
- `docker-compose.yml` — llama-server (private) + Open WebUI (public via Caddy) + Caddy TLS
- `Caddyfile` — TLS reverse proxy, security headers, body-size cap; inference deliberately not routed
- `.env.example` — domain, API key, branding, disclaimer banner
- `RUNPOD.md` — RunPod Pod/serverless reconciliation, model-delivery recommendation, user handoff
- `runpod-entrypoint.sh` — first-boot rebuild/reuse script for a RunPod A100 network-volume setup
