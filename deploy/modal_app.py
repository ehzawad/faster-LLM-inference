"""
Modal deployment — EC/NID Qwen2.5-7B assistant (llama.cpp), scale-to-zero, ~30 concurrent.

Engine: llama-server (llama.cpp) for real slot-based concurrent batching, behind a tiny FastAPI
page that injects the trained system prompt and serves a public chat UI. A100-40GB.

USAGE (from the repo root, using the venv that has modal):
    .venv/bin/modal run   deploy/modal_app.py::build_gguf    # ONE-TIME: merge adapter -> Q8_0 GGUF on a Volume (~15 min)
    .venv/bin/modal deploy deploy/modal_app.py               # deploy -> prints your public https://...modal.run URL

CONTROL:
    .venv/bin/modal app stop ec-nid-chat                     # COMPLETELY OFF ($0). Redeploy to turn back on.
Idle costs $0 automatically (scale-to-zero). You only pay per active second while it's generating.
"""
import subprocess
import time
import modal

APP_NAME = "ec-nid-chat"
ADAPTER = "ehzawad/ec-SFT-qwen25-7b-lora"   # public LoRA
BASE = "Qwen/Qwen2.5-7B-Instruct"           # public base
VOL_DIR = "/models"
# Decision: SPEED is the top priority (small drift accepted). Q4_K_M ≈ 101 tok/s solo — the FASTEST quant
# (vs Q5 94, Q8 82). It's byte-identical to Q8 on common facts and drifts only on rare ones (acceptable).
# Q8_0 (verbatim) and Q5_K_M also live on the Volume — a 1-line swap if priorities change.
QUANT = "Q4_K_M"
GGUF = f"{VOL_DIR}/ec-qwen25-7b.{QUANT}.gguf"
LLAMA_PORT = 8080

app = modal.App(APP_NAME)
models = modal.Volume.from_name("ec-nid-models", create_if_missing=True)

# One image for both steps: CUDA toolkit (to compile llama-server) + the merge/convert python deps.
# No torchao is installed, so peft's probe returns False cleanly (the Colab crash can't happen here).
image = (
    # devel image = CUDA 12.4 toolkit (nvcc) needed to COMPILE llama.cpp with CUDA.
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "cmake", "curl", "libgomp1")
    .run_commands(
        "nvcc --version",   # prove CUDA toolkit 12.4 is present in the build log
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /llama.cpp",
        # CMAKE_CUDA_ARCHITECTURES=80 (A100 = sm_80) is REQUIRED: Modal builds images without a GPU,
        # so 'native' arch auto-detection would fail. Pinning sm_80 also speeds the compile.
        # The CUDA backend links the driver API (libcuda), but the build host has NO driver.
        # Point the linker at the toolkit's STUB libcuda; Modal provides the real driver at runtime.
        "ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1",
        "cmake -S /llama.cpp -B /llama.cpp/build "
        "-DGGML_CUDA=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=ON -DCMAKE_CUDA_ARCHITECTURES=80 "
        "-DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs",
        "LIBRARY_PATH=/usr/local/cuda/lib64/stubs LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs "
        "cmake --build /llama.cpp/build --config Release -j --target llama-server",
        "test -x /llama.cpp/build/bin/llama-server && echo 'llama-server built OK'",
        # prove it's actually CUDA-linked (informational; static builds may not show in ldd)
        "ldd /llama.cpp/build/bin/llama-server | grep -i cuda || echo 'WARN: no dynamic CUDA libs in ldd'",
    )
    # torch pinned to the cu124 build so it matches the image's CUDA 12.4 toolkit and CANNOT hit a
    # 'driver too old' error (default torch wheels are cu128 and need a newer driver than 12.4).
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers>=4.45", "peft>=0.13", "tokenizers>=0.20",
        "accelerate", "safetensors", "huggingface_hub", "sentencepiece", "protobuf",
        "numpy", "gguf", "fastapi[standard]", "httpx",
    )
    # Also build llama-quantize (needed for k-quants like Q5_K_M; convert alone only emits q8_0).
    # Reuses the already-configured CUDA build dir, so ONLY this target compiles (llama-server is cached).
    .run_commands(
        "LIBRARY_PATH=/usr/local/cuda/lib64/stubs LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs "
        "cmake --build /llama.cpp/build --config Release -j --target llama-quantize",
        "test -x /llama.cpp/build/bin/llama-quantize && echo 'llama-quantize built OK'",
    )
)
LLAMA_BIN = "/llama.cpp/build/bin/llama-server"
QUANTIZE_BIN = "/llama.cpp/build/bin/llama-quantize"


def llama_cmd(gguf=GGUF):
    # NOTE: current llama.cpp wants `--flash-attn on` (it's no longer a bare boolean flag).
    # cont-batching is default-on now, so it's not passed. 30 slots x 8192 = 245760 total ctx.
    return [LLAMA_BIN, "-m", gguf, "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
            "--n-gpu-layers", "999", "--flash-attn", "on",
            "--ctx-size", "245760", "--parallel", "30"]


def wait_for_health(proc, tries=240):
    import time as _t
    import urllib.request
    for _ in range(tries):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=2).status == 200:
                return
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited during startup (check logs for bad flags)")
        _t.sleep(1)
    raise RuntimeError("llama-server did not become healthy in time")


# ---------------- ONE-TIME: merge adapter -> Q8_0 GGUF on the Volume ----------------
@app.function(image=image, gpu="A100-40GB", volumes={VOL_DIR: models}, timeout=60 * 30)
def build_gguf(force: bool = False):
    import os, shutil, torch
    from pathlib import Path
    from huggingface_hub import hf_hub_download
    # Fail loud if CUDA isn't actually usable (driver/torch mismatch) instead of silently using CPU.
    assert torch.cuda.is_available(), "CUDA not available in build_gguf — torch/driver mismatch"
    print("torch", torch.__version__, "| cuda", torch.version.cuda, "| gpu", torch.cuda.get_device_name(0))

    # Stage the system prompt onto the Volume so serving has NO runtime Hugging Face dependency.
    sp_dst = Path(f"{VOL_DIR}/system_prompt.txt")
    if not sp_dst.exists():
        shutil.copyfile(hf_hub_download(ADAPTER, "system_prompt.txt"), sp_dst)
        models.commit()

    if Path(GGUF).exists() and not force:
        print("GGUF already present:", GGUF)
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    merged = "/tmp/merged"
    print("[1/2] merging LoRA into base on GPU ...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True)
    m = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
    m.save_pretrained(merged, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(ADAPTER).save_pretrained(merged)  # trained chat_template -> GGUF
    del base, m
    print("[2/3] converting -> bf16 GGUF (lossless intermediate) ...")
    bf16 = "/tmp/model.bf16.gguf"
    subprocess.run(["python", "/llama.cpp/convert_hf_to_gguf.py", merged,
                    "--outtype", "bf16", "--outfile", bf16], check=True)
    print(f"[3/3] quantizing -> {QUANT} ...")
    tmp = GGUF + ".tmp"
    subprocess.run([QUANTIZE_BIN, bf16, tmp, QUANT, str(os.cpu_count() or 4)], check=True)
    os.replace(tmp, GGUF)   # atomic: a crashed quantize never leaves a partial GGUF at the final path
    models.commit()
    print("BUILT:", GGUF)


# ---------------- LOAD TEST: 30 concurrent Bengali questions + GPU memory footprint ----------------
@app.function(image=image, gpu="A100-40GB", volumes={VOL_DIR: models}, timeout=60 * 15)
def loadtest(n: int = 30, max_tokens: int = 256, gguf: str = GGUF):
    """Fire n concurrent EC/NID questions at a local llama-server; report latency/throughput + VRAM.
    Run: modal run deploy/modal_app.py::loadtest [--n 1] [--gguf /models/...gguf]"""
    import asyncio
    import threading
    from pathlib import Path
    import httpx

    print("MODEL:", gguf.split("/")[-1])
    proc = subprocess.Popen(llama_cmd(gguf))
    wait_for_health(proc)

    sp = Path(f"{VOL_DIR}/system_prompt.txt")
    system = sp.read_text("utf-8").strip() if sp.exists() else ""

    QUESTIONS = [
        "নতুন ভোটার হতে কী কী কাগজপত্র লাগে?",
        "আমার বয়স ৩৮, আমি এখনো ভোটার হইনি, কীভাবে ভোটার হতে পারি?",
        "আমার নামের বানানে ভুল হয়েছে, কীভাবে সংশোধন করবো?",
        "জন্ম তারিখ ও বাবা-মায়ের নাম সংশোধন করতে কী কী কাগজপত্র লাগবে?",
        "স্মার্ট কার্ড রেডি হয়েছে কিনা কোথা থেকে জানব?",
        "স্মার্ট কার্ড হারিয়ে গেলে কী করনীয়?",
        "এনআইডি নষ্ট হয়ে গেছে, পুনঃইস্যু করা যাবে?",
        "পুনঃইস্যুর জন্য কত টাকা লাগে?",
        "অনলাইনে এনআইডি অ্যাকাউন্ট খুলতে পারছি না কেন?",
        "আবেদন করছি কিন্তু ওটিপি আসছে না, এখন কী করব?",
        "এনআইডি অনলাইন কপি কীভাবে ডাউনলোড করব?",
        "ভোটার এলাকা স্থানান্তর কীভাবে করব?",
    ]

    def gpu_used_total():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        u, t = [int(x) for x in out.split(",")]
        return u, t

    base_used, total = gpu_used_total()   # after model load, before traffic
    peak = {"used": base_used}
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                peak["used"] = max(peak["used"], gpu_used_total()[0])
            except Exception:
                pass
            time.sleep(0.25)
    th = threading.Thread(target=poll, daemon=True); th.start()

    async def one(i):
        q = QUESTIONS[i % len(QUESTIONS)]
        payload = {"model": "ec", "stream": False, "temperature": 0.2, "top_p": 0.9, "max_tokens": max_tokens,
                   "messages": ([{"role": "system", "content": system}] if system else [])
                   + [{"role": "user", "content": q}]}
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=600) as c:
                r = await c.post(f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions", json=payload)
            dt = time.perf_counter() - t0
            if r.status_code != 200:
                return {"ok": False, "dt": dt, "toks": 0}
            toks = r.json().get("usage", {}).get("completion_tokens", 0)
            return {"ok": True, "dt": dt, "toks": toks}
        except Exception:
            return {"ok": False, "dt": time.perf_counter() - t0, "toks": 0}

    async def run():
        t0 = time.perf_counter()
        res = await asyncio.gather(*[one(i) for i in range(n)])
        return res, time.perf_counter() - t0

    results, wall = asyncio.run(run())
    stop.set(); th.join(timeout=2)

    ok = [r for r in results if r["ok"]]
    lat = sorted(r["dt"] for r in ok)
    toks = sum(r["toks"] for r in ok)
    pct = lambda p: lat[min(len(lat) - 1, int(len(lat) * p))] if lat else 0.0

    print("\n================ {}-CONCURRENT LOAD TEST ================".format(n))
    print(f"succeeded           : {len(ok)}/{n}")
    print(f"wall clock          : {wall:.1f}s  (all {n} fired simultaneously)")
    print(f"latency p50/p95/max : {pct(.5):.1f}s / {pct(.95):.1f}s / {(lat[-1] if lat else 0):.1f}s")
    print(f"output tokens total : {toks}")
    print(f"aggregate throughput: {toks / wall:.1f} tok/s" if wall else "n/a")
    print("---------------- GPU memory (A100-40GB) ----------------")
    print(f"after model load    : {base_used} MiB / {total} MiB")
    print(f"peak under load     : {peak['used']} MiB / {total} MiB  ({100 * peak['used'] / total:.0f}%)")
    print("========================================================\n")
    proc.terminate()
    return {"succeeded": len(ok), "n": n, "wall_s": round(wall, 1),
            "p50_s": round(pct(.5), 1), "p95_s": round(pct(.95), 1),
            "agg_tok_s": round(toks / wall, 1) if wall else 0,
            "mem_load_mib": base_used, "mem_peak_mib": peak["used"], "mem_total_mib": total}


# ---------------- MULTI-TURN CONTEXT TEST: is a differing answer quant or context? ----------------
@app.function(image=image, gpu="A100-40GB", volumes={VOL_DIR: models}, timeout=60 * 25)
def ctx_compare(quants: str = "Q8_0,Q5_K_M,Q4_K_M", max_tokens: int = 220):
    """Replay the SAME multi-turn conversation on each quant (greedy temp 0) and print every answer,
    so we can see whether a differing reply is from QUANT precision or from CONVERSATION CONTEXT.
    Run: modal run deploy/modal_app.py::ctx_compare"""
    import asyncio
    from pathlib import Path
    import httpx

    sp = Path(f"{VOL_DIR}/system_prompt.txt")
    system = sp.read_text("utf-8").strip() if sp.exists() else ""
    USER_TURNS = [
        "স্মার্ট কার্ড হারিয়ে গেলে কী করনীয়,নতুন কার্ড কীভাবে পাবো?",
        "এনআইডি নষ্ট হয়ে গেছে,পুনঃইস্যু করা যাবে?",
        "নতুন NID কার্ড  কত টাকা লাগে",
        "কি কি ডকুমেন্ট লাগে",
        "কতদিন লাগে",
    ]

    # ensure each requested quant exists — rebuild bf16 master + quantize if missing
    import os
    bf16 = f"{VOL_DIR}/ec-qwen25-7b.bf16.gguf"
    want = [x.strip() for x in quants.split(",") if x.strip()]
    missing = [q for q in want if not Path(f"{VOL_DIR}/ec-qwen25-7b.{q}.gguf").exists()]
    if missing and not Path(bf16).exists():
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        merged = "/tmp/merged"
        print("building bf16 master (one-time) for quantizing ...")
        base = AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True)
        m = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
        m.save_pretrained(merged, safe_serialization=True, max_shard_size="5GB")
        AutoTokenizer.from_pretrained(ADAPTER).save_pretrained(merged)
        del base, m
        subprocess.run(["python", "/llama.cpp/convert_hf_to_gguf.py", merged,
                        "--outtype", "bf16", "--outfile", bf16], check=True)
        subprocess.run(["rm", "-rf", merged])
        models.commit()
    for q in missing:
        g = f"{VOL_DIR}/ec-qwen25-7b.{q}.gguf"
        print(f"quantizing -> {q} ...")
        tmp = g + ".tmp"
        subprocess.run([QUANTIZE_BIN, bf16, tmp, q, str(os.cpu_count() or 4)], check=True)
        os.replace(tmp, g)
        models.commit()

    out = {}
    for q in [x.strip() for x in quants.split(",") if x.strip()]:
        g = f"{VOL_DIR}/ec-qwen25-7b.{q}.gguf"
        if not Path(g).exists():
            print(f"[{q}] GGUF not on Volume — skipping")
            continue
        proc = subprocess.Popen(llama_cmd(g))
        try:
            wait_for_health(proc)
        except Exception as e:
            print(f"[{q}] llama-server failed: {e}")
            continue

        import json as _json
        history = []
        speeds = []

        async def ask(user):
            msgs = [{"role": "system", "content": system}] + history + [{"role": "user", "content": user}]
            payload = {"model": "ec", "stream": True, "temperature": 0.0, "top_p": 1.0,
                       "max_tokens": max_tokens, "messages": msgs}
            t0 = time.perf_counter(); ttft = None; n = 0; chunks = []
            async with httpx.AsyncClient(timeout=300) as c:
                async with c.stream("POST", f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions", json=payload) as r:
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        d = line[5:].strip()
                        if d == "[DONE]":
                            continue
                        try:
                            cc = _json.loads(d)["choices"][0]["delta"].get("content")
                            if cc:
                                if ttft is None:
                                    ttft = time.perf_counter() - t0
                                n += 1; chunks.append(cc)
                        except Exception:
                            pass
            dt = time.perf_counter() - t0
            dec = (n - 1) / (dt - ttft) if (n > 1 and ttft and dt > ttft) else 0.0
            if dec:
                speeds.append(dec)
            return "".join(chunks).strip()

        answers = []
        for u in USER_TURNS:
            a = asyncio.run(ask(u))
            history += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            answers.append(a)
        out[q] = {"answers": answers, "tok_s": round(sum(speeds) / len(speeds), 1) if speeds else 0.0}
        try: proc.terminate()
        except Exception: pass
        time.sleep(3)

    print("\n========= MULTI-TURN: VERBATIM + SPEED across quants (greedy temp 0) =========")
    print("SPEED: " + "  |  ".join(f"{q}={out[q]['tok_s']} tok/s" for q in out))
    for i, u in enumerate(USER_TURNS):
        print(f"\nUSER: {u}")
        for q in out:
            print(f"  [{q}] {out[q]['answers'][i]}")
    print("\n(Per turn: does Q6_K match Q8_0 VERBATIM? If yes -> Q6 = fidelity + more speed.)")
    print("=============================================================================\n")
    return {q: out[q]["tok_s"] for q in out}


# ---------------- QUANT COMPARE: quality (answers) + solo speed across quants ----------------
@app.function(image=image, gpu="A100-40GB", volumes={VOL_DIR: models}, timeout=60 * 40)
def compare_quants(quants: str = "Q4_K_M,Q5_K_M,Q6_K,Q8_0", max_tokens: int = 220):
    """Build (from a kept bf16 master) + benchmark each quant: solo DECODE tok/s + the actual answers,
    so we pick the fastest quant whose facts still match Q8. Run: modal run deploy/modal_app.py::compare_quants"""
    import asyncio
    import json
    import os
    from pathlib import Path
    import httpx

    quant_list = [q.strip() for q in quants.split(",") if q.strip()]
    bf16 = f"{VOL_DIR}/ec-qwen25-7b.bf16.gguf"

    # bf16 master (merge once, keep on Volume -> future quant swaps are a ~1-min llama-quantize)
    if not Path(bf16).exists():
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        merged = "/tmp/merged"
        print("merging LoRA -> bf16 master (one-time) ...")
        base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                                    device_map={"": 0}, low_cpu_mem_usage=True)
        m = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
        m.save_pretrained(merged, safe_serialization=True, max_shard_size="5GB")
        AutoTokenizer.from_pretrained(ADAPTER).save_pretrained(merged)
        del base, m
        subprocess.run(["python", "/llama.cpp/convert_hf_to_gguf.py", merged,
                        "--outtype", "bf16", "--outfile", bf16], check=True)
        subprocess.run(["rm", "-rf", merged])
        models.commit()

    # ensure each quant exists (fast: quantize from the bf16 master)
    for q in quant_list:
        g = f"{VOL_DIR}/ec-qwen25-7b.{q}.gguf"
        if not Path(g).exists():
            print(f"quantizing -> {q} ...")
            tmp = g + ".tmp"
            subprocess.run([QUANTIZE_BIN, bf16, tmp, q, str(os.cpu_count() or 4)], check=True)
            os.replace(tmp, g)
            models.commit()

    sp = Path(f"{VOL_DIR}/system_prompt.txt")
    system = sp.read_text("utf-8").strip() if sp.exists() else ""
    PROMPTS = ["পুনঃইস্যুর জন্য কত টাকা লাগে?",
               "নতুন ভোটার হতে কী কী কাগজপত্র লাগে?",
               "স্মার্ট কার্ড হারিয়ে গেলে কী করনীয়?"]
    results = {}

    for q in quant_list:
        g = f"{VOL_DIR}/ec-qwen25-7b.{q}.gguf"
        proc = subprocess.Popen(llama_cmd(g))
        try:
            wait_for_health(proc)
        except Exception as e:
            print(f"[{q}] llama-server failed: {e}")
            continue

        async def one(prompt):
            payload = {"model": "ec", "stream": True, "temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens,
                       "messages": ([{"role": "system", "content": system}] if system else [])
                       + [{"role": "user", "content": prompt}]}
            t0 = time.perf_counter(); ttft = None; n = 0; chunks = []
            async with httpx.AsyncClient(timeout=300) as c:
                async with c.stream("POST", f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions", json=payload) as r:
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            continue
                        try:
                            cc = json.loads(data)["choices"][0]["delta"].get("content")
                            if cc:
                                if ttft is None:
                                    ttft = time.perf_counter() - t0
                                n += 1; chunks.append(cc)
                        except Exception:
                            pass
            dt = time.perf_counter() - t0
            dec = (n - 1) / (dt - ttft) if (n > 1 and ttft and dt > ttft) else 0.0
            return dec, "".join(chunks)

        qr = [asyncio.run(one(p)) for p in PROMPTS]
        results[q] = {"tok_s": round(sum(d for d, _ in qr) / len(qr), 1),
                      "answers": [a for _, a in qr], "size_gb": round(Path(g).stat().st_size / 1e9, 2)}
        try: proc.terminate()
        except Exception: pass
        time.sleep(3)

    print("\n================ QUANT QUALITY + SOLO SPEED ================")
    print("SPEED: " + "  |  ".join(
        f"{q}={results[q]['tok_s']} tok/s ({results[q]['size_gb']}GB)" for q in quant_list if q in results))
    # grouped by PROMPT so Q8/Q5/Q4 align for a proper fact-by-fact juxtaposition (FULL answers)
    for i, p in enumerate(PROMPTS):
        print(f"\n#### PROMPT {i + 1}: {p}")
        for q in quant_list:
            if q in results:
                print(f"----- {q} -----")
                print(results[q]["answers"][i].strip())
    print("\n===========================================================\n")
    return {q: {"tok_s": results[q]["tok_s"], "answers": results[q]["answers"]} for q in results}


# ---------------- SPECULATIVE-DECODING A/B: baseline vs draft, solo decode tok/s ----------------
DRAFT_GGUF = f"{VOL_DIR}/qwen2.5-0.5b-instruct.Q8_0.gguf"


@app.function(image=image, gpu="A100-40GB", volumes={VOL_DIR: models}, timeout=60 * 20)
def spectest(max_tokens: int = 256):
    """A/B: solo DECODE tok/s with no draft vs a Qwen2.5-0.5B draft (speculative decoding).
    Lossless at temp 0. Run: modal run deploy/modal_app.py::spectest"""
    import asyncio
    import json
    from pathlib import Path
    import httpx

    # 1) build the draft GGUF once (Qwen2.5-0.5B-Instruct — SAME vocab as the 7B target, required)
    if not Path(DRAFT_GGUF).exists():
        from huggingface_hub import snapshot_download
        print("building draft GGUF (Qwen2.5-0.5B-Instruct) ...")
        d = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")
        subprocess.run(["python", "/llama.cpp/convert_hf_to_gguf.py", d,
                        "--outtype", "q8_0", "--outfile", DRAFT_GGUF], check=True)
        models.commit()

    # 2) print the EXACT draft/spec flags this llama-server build supports (avoid guessing)
    h = subprocess.run([LLAMA_BIN, "--help"], capture_output=True, text=True)
    print("=== draft/spec flags in this build ===")
    for line in (h.stdout + h.stderr).splitlines():
        if "draft" in line.lower() or "spec" in line.lower():
            print("  " + line.strip())
    print("======================================")

    sp = Path(f"{VOL_DIR}/system_prompt.txt")
    system = sp.read_text("utf-8").strip() if sp.exists() else ""
    QS = ["স্মার্ট কার্ড হারিয়ে গেলে কী করনীয়?",
          "নতুন ভোটার হতে কী কী কাগজপত্র লাগে?",
          "পুনঃইস্যুর জন্য কত টাকা লাগে?",
          "ভোটার এলাকা স্থানান্তর কীভাবে করব?"]

    def bench(extra, label):
        proc = subprocess.Popen(llama_cmd() + extra)
        try:
            wait_for_health(proc)
        except Exception as e:
            print(f"[{label}] llama-server did NOT start ({e}) — likely a wrong draft flag; see flags above.")
            try: proc.terminate()
            except Exception: pass
            return None

        async def one(q):
            payload = {"model": "ec", "stream": True, "temperature": 0.0, "top_p": 1.0, "max_tokens": max_tokens,
                       "messages": ([{"role": "system", "content": system}] if system else [])
                       + [{"role": "user", "content": q}]}
            t0 = time.perf_counter(); ttft = None; n = 0
            async with httpx.AsyncClient(timeout=300) as c:
                async with c.stream("POST", f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions", json=payload) as r:
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            continue
                        try:
                            d = json.loads(data)["choices"][0]["delta"].get("content")
                            if d:
                                if ttft is None:
                                    ttft = time.perf_counter() - t0
                                n += 1
                        except Exception:
                            pass
            dt = time.perf_counter() - t0
            decode = (n - 1) / (dt - ttft) if (n > 1 and ttft and dt > ttft) else 0.0
            return decode

        rates = [asyncio.run(one(q)) for q in QS]   # sequential = pure solo
        avg = sum(rates) / len(rates) if rates else 0.0
        print(f"[{label}] solo DECODE tok/s: {[round(x, 1) for x in rates]} -> avg {avg:.1f}")
        try: proc.terminate()
        except Exception: pass
        time.sleep(3)
        return avg

    base = bench([], "BASELINE (no draft)")
    # README example flags for current master; if these are wrong the help dump above shows the right ones.
    spec = bench(["--spec-draft-model", DRAFT_GGUF, "--spec-draft-ngl", "99",
                  "--spec-draft-n-max", "8", "--spec-type", "draft-simple"], "SPECULATIVE (0.5B draft)")

    print("\n================ SPECULATIVE DECODING A/B ================")
    print(f"baseline    : {base:.1f} tok/s" if base else "baseline    : FAILED")
    print(f"speculative : {spec:.1f} tok/s" if spec else "speculative : FAILED (check draft flags above)")
    if base and spec:
        print(f"speedup     : {spec / base:.2f}x  ({'KEEP' if spec > base * 1.05 else 'NOT WORTH IT'})")
    print("==========================================================\n")
    return {"baseline": base, "speculative": spec}


# ---------------- SERVING: llama-server (concurrent) + FastAPI chat page ----------------
@app.cls(image=image, gpu="A100-40GB", volumes={VOL_DIR: models},
         scaledown_window=120, timeout=60 * 60, max_containers=1)  # 120s idle tail = lower idle GPU cost
@modal.concurrent(max_inputs=30)   # 30 concurrent requests share ONE A100 container (no extra-GPU cost)
class Server:
    @modal.enter()
    def start(self):
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        import urllib.request

        if not Path(GGUF).exists():
            raise RuntimeError(
                "GGUF not found on the Volume. Run the one-time build first:\n"
                "    modal run deploy/modal_app.py::build_gguf")

        sp_vol = Path(f"{VOL_DIR}/system_prompt.txt")   # prefer the Volume copy (no HF dependency on the hot path)
        if sp_vol.exists():
            self.system_prompt = sp_vol.read_text("utf-8").strip()
        else:
            self.system_prompt = Path(hf_hub_download(ADAPTER, "system_prompt.txt")).read_text("utf-8").strip()

        # 30 slots x 8192 ctx = 245760 total; Q8_0 ~8GB weights + ~13GB KV ≈ 21GB -> fits A100 40GB.
        self.proc = subprocess.Popen(llama_cmd())
        wait_for_health(self.proc)
        print("llama-server ready")

    @modal.exit()
    def stop(self):
        try:
            self.proc.terminate()
        except Exception:
            pass

    @modal.asgi_app()
    def web(self):
        import json
        import time as _t
        import httpx
        from fastapi import FastAPI, Request
        from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

        api = FastAPI()
        SYS = self.system_prompt
        UPSTREAM = f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions"
        MAX_MSG_CHARS, MAX_TURNS = 4000, 16
        inflight: set[str] = set()   # IPs currently generating (1 per IP -> no single user grabs all slots)
        last_seen: dict[str, float] = {}

        @api.get("/")
        def index():
            return HTMLResponse(CHAT_HTML)

        @api.post("/chat")
        async def chat(req: Request):
            ip = req.client.host if req.client else "?"
            if self.proc.poll() is not None:                      # llama-server died -> tell Modal to recycle
                import os as _os, signal as _sig
                _os.kill(_os.getpid(), _sig.SIGTERM)
                return JSONResponse({"error": "model restarting, retry shortly"}, status_code=503)
            now = _t.monotonic()
            if now - last_seen.get(ip, 0.0) < 1.0:                # basic per-IP rate limit
                return JSONResponse({"error": "ধীরে / slow down"}, status_code=429)
            if ip in inflight:                                    # one in-flight generation per IP
                return JSONResponse({"error": "একটি অনুরোধ চলছে / one at a time"}, status_code=429)
            last_seen[ip] = now

            body = await req.json()
            raw = body.get("messages", [])
            if not isinstance(raw, list):
                return JSONResponse({"error": "bad request"}, status_code=422)
            msgs = []
            for m in raw[-(MAX_TURNS * 2):]:                      # whitelist roles, cap size, trim history
                if not isinstance(m, dict):
                    continue
                role, content = m.get("role"), m.get("content")
                if role in ("user", "assistant") and isinstance(content, str):
                    msgs.append({"role": role, "content": content[:MAX_MSG_CHARS]})
            if not msgs or msgs[-1]["role"] != "user":
                return JSONResponse({"error": "last message must be from user"}, status_code=422)

            # Token-budget trim (Bengali ≈ 1 token/char): always keep the system prompt + the MOST RECENT
            # turns (which carry follow-up context like "কতদিন লাগে") + the current question, within 8k.
            budget = 8192 - 512 - len(SYS)          # ctx - output reservation - system prompt
            while len(msgs) > 1 and sum(len(m["content"]) for m in msgs) > budget:
                msgs = msgs[2:]                     # drop the oldest user+assistant pair

            payload = {"model": "ec-nid",
                       "messages": [{"role": "system", "content": SYS}] + msgs,
                       "stream": True, "temperature": 0.0, "top_p": 1.0, "max_tokens": 512}  # greedy = most factual/deterministic

            async def gen():
                inflight.add(ip)
                try:
                    async with httpx.AsyncClient(timeout=180) as client:
                        async with client.stream("POST", UPSTREAM, json=payload) as r:
                            if r.status_code != 200:             # surface upstream errors instead of a stuck "..."
                                detail = (await r.aread()).decode("utf-8", "ignore")[:200]
                                yield "data: " + json.dumps({"error": detail or f"upstream {r.status_code}"}) + "\n\n"
                                return
                            async for line in r.aiter_lines():
                                if line:
                                    yield line + "\n"
                except Exception:
                    yield "data: " + json.dumps({"error": "stream failed"}) + "\n\n"
                finally:
                    inflight.discard(ip)

            return StreamingResponse(gen(), media_type="text/event-stream")

        return api


CHAT_HTML = """<!doctype html><html lang="bn"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EC/NID সহায়ক</title>
<style>
 body{font-family:system-ui,'Noto Sans Bengali',sans-serif;max-width:760px;margin:0 auto;padding:16px;background:#0b0f14;color:#e6edf3}
 h2{font-weight:600;margin:.2em 0}
 #log{min-height:48vh;border:1px solid #222;border-radius:10px;padding:12px;overflow-y:auto;white-space:pre-wrap}
 .u{color:#7ee787;margin-top:10px} .b{color:#cdd9e5;margin:4px 0 10px} .err{color:#ff7b72}
 #row{display:flex;gap:8px;margin-top:10px}
 #q{flex:1;padding:10px;border-radius:8px;border:1px solid #333;background:#11161c;color:#e6edf3}
 button{padding:10px 16px;border-radius:8px;border:0;background:#238636;color:#fff;cursor:pointer}
 button:disabled{background:#30363d;cursor:not-allowed}
</style></head><body>
<h2>EC/NID সহায়ক · Qwen2.5-7B</h2>
<div id="log"></div>
<div id="row"><input id="q" placeholder="আপনার প্রশ্ন লিখুন..." autofocus><button id="btn" onclick="send()">পাঠান</button></div>
<script>
const log=document.getElementById('log'), q=document.getElementById('q'), btn=document.getElementById('btn');
let history=[], busy=false;
function add(cls,txt){const d=document.createElement('div');d.className=cls;d.textContent=txt;log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
function setBusy(b){busy=b;btn.disabled=b;q.disabled=b;if(!b)q.focus();}
async function send(){
  if(busy) return;
  const text=q.value.trim(); if(!text) return; q.value='';
  setBusy(true);
  add('u','User: '+text);
  const bot=add('b','...'); let acc='', errored=false;
  history.push({role:'user',content:text});
  try{
    const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:history})});
    if(!res.ok){let m=res.status;try{m=(await res.json()).error||m;}catch(e){} bot.className='err';bot.textContent='[ত্রুটি / error: '+m+']';history.pop();return;}
    const reader=res.body.getReader(), dec=new TextDecoder(); let buf='';
    while(true){const {value,done}=await reader.read(); if(done) break;
      buf+=dec.decode(value,{stream:true});
      let i; while((i=buf.indexOf('\\n'))>=0){let line=buf.slice(0,i); buf=buf.slice(i+1);
        line=line.trim(); if(!line.startsWith('data:')) continue;
        const data=line.slice(5).trim(); if(data==='[DONE]') continue;
        try{const j=JSON.parse(data);
          if(j.error){errored=true;bot.className='err';bot.textContent='[ত্রুটি / error: '+j.error+']';continue;}
          const d=j.choices?.[0]?.delta?.content; if(d){acc+=d; bot.textContent=acc; log.scrollTop=log.scrollHeight;}
        }catch(e){}
      }
    }
    if(errored||!acc){history.pop();} else {history.push({role:'assistant',content:acc});}
  }catch(e){
    bot.className='err';bot.textContent='[সংযোগ ব্যর্থ / connection failed]';history.pop();
  }finally{setBusy(false);}
}
q.addEventListener('keydown',e=>{if(e.key==='Enter'&&!busy)send();});
</script></body></html>"""
