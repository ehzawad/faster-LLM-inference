#!/usr/bin/env python3
"""
Merge the ec-SFT LoRA adapter into Qwen2.5-7B-Instruct and write a plain,
vLLM-ready checkpoint.

Why merge instead of serving the LoRA dynamically?
  * Zero per-token adapter overhead (the LoRA deltas are folded into the base
    weights once, here, instead of being applied on every forward pass).
  * Lets vLLM use full CUDA graphs and FP8/AWQ quantization paths that the
    dynamic-LoRA path partially disables.
  * You always use this one adapter, so there is no reason to pay the dynamic
    cost. (If you needed to hot-swap many adapters, you'd keep them separate
    and use `vllm serve --enable-lora` instead — see serve.sh, Option B.)

Run this ONCE on the GPU host. Output goes to ./ec-qwen25-7b-merged.

    python merge_adapter.py
    python merge_adapter.py --out /models/ec-merged --dtype bfloat16
"""
import argparse
import inspect
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ADAPTER_ID_DEFAULT = "ehzawad/ec-SFT-qwen25-7b-lora"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-id", default=ADAPTER_ID_DEFAULT)
    ap.add_argument("--adapter-revision", default=None, help="optional commit/branch pin")
    ap.add_argument("--base-model", default=None, help="override; else read from adapter config")
    ap.add_argument("--out", default="./ec-qwen25-7b-merged")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    out_dir = Path(args.out).resolve()

    print(f"[1/5] downloading adapter {args.adapter_id} ...")
    adapter_dir = Path(
        snapshot_download(repo_id=args.adapter_id, revision=args.adapter_revision)
    )

    # The base model is recorded in adapter_config.json ("base_model_name_or_path").
    base_model = args.base_model
    if base_model is None:
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
        base_model = cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
    print(f"[2/5] base model = {base_model} (dtype={args.dtype})")

    # transformers renamed torch_dtype -> dtype in 4.49; accept either (mirrors model card).
    dtype_kw = (
        "dtype"
        if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
        else "torch_dtype"
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model, **{dtype_kw: dtype}, device_map="cpu", attn_implementation="sdpa"
    )

    print("[3/5] attaching adapter + merging weights ...")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()  # fold LoRA deltas into the base linear layers

    print(f"[4/5] saving merged model -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)

    # Save the tokenizer FROM THE ADAPTER DIR so the trained chat_template ships
    # with the merged model and vLLM renders prompts exactly as in training.
    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(str(out_dir))

    # Carry the trained system prompt alongside the weights so the client can find it.
    sys_prompt = adapter_dir / "system_prompt.txt"
    if sys_prompt.is_file():
        shutil.copy2(sys_prompt, out_dir / "system_prompt.txt")
        print("[5/5] copied system_prompt.txt into merged dir")
    else:
        print("[5/5] WARNING: system_prompt.txt not found in adapter repo")

    print(f"\nDone. Serve it with:\n    bash serve.sh {out_dir}\n")


if __name__ == "__main__":
    main()
