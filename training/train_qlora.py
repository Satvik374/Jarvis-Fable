#!/usr/bin/env python
"""QLoRA fine-tuning of a small base model into the Jarvis brain.

Trains a LoRA adapter on the dataset produced by ``dataset/build_dataset.py``
so the model learns to emit Jarvis's exact action JSON given a screen
observation.

Two backends, chosen automatically:
  * Unsloth (recommended on 4 GB VRAM) - 2x less memory, faster. Used if the
    ``unsloth`` package is importable.
  * Plain transformers + bitsandbytes 4-bit + PEFT - the portable fallback.

Hardware note (RTX 2050 / 4 GB VRAM):
  * Qwen2.5-3B QLoRA fits *only* with Unsloth and a short sequence length; it
    is tight. If you hit CUDA OOM, either:
      - switch to a 1.5B base: --model unsloth/Qwen2.5-1.5B-Instruct
      - lower --max-seq-len (e.g. 1024) and keep --batch 1,
      - or train on a free Colab T4 (see training/README.md), then copy the
        adapter back and export to Ollama.

Example:
    python training/train_qlora.py \
        --model unsloth/Qwen2.5-3B-Instruct \
        --train dataset/data/jarvis_sft.train.jsonl \
        --val   dataset/data/jarvis_sft.val.jsonl \
        --epochs 3 --batch 1 --grad-accum 16 --max-seq-len 1536
"""

from __future__ import annotations

import argparse
from pathlib import Path


QWEN_LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="QLoRA fine-tune the Jarvis brain")
    ap.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct",
                    help="base model (HF id). Use Qwen2.5-1.5B-Instruct on <=4GB VRAM.")
    ap.add_argument("--train", default="dataset/data/jarvis_sft.train.jsonl")
    ap.add_argument("--val", default="dataset/data/jarvis_sft.val.jsonl")
    ap.add_argument("--out", default="training/outputs/jarvis-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=1536)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--eval", action="store_true",
                    help="run in-loop evaluation. OFF by default: computing "
                         "logits over the full vocab OOMs a 4 GB GPU. Use "
                         "eval_adapter.py (generation-based) after training instead.")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 16],
                    help="4 = QLoRA (4-bit, for big models); 16 = plain LoRA in "
                         "bf16/fp16 (best for <=1.5B models, avoids bitsandbytes).")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the number of train examples (0 = all). Handy for "
                         "quick smoke tests on constrained machines.")
    ap.add_argument("--no-unsloth", action="store_true",
                    help="force the plain transformers path")
    ap.add_argument("--merge", action="store_true",
                    help="also save a merged fp16 model (needed for GGUF export)")
    return ap.parse_args(argv)


def _ensure_writable_hf_cache() -> None:
    """Redirect the HuggingFace cache if the default (~/.cache) isn't writable.

    On this machine ~/.cache/huggingface is ACL-locked (WinError 5), so model
    downloads fail. We fall back to %LOCALAPPDATA%\\Jarvis\\hf, which is writable.
    Must run before transformers/datasets are imported.
    """
    import os
    import tempfile

    if os.environ.get("HF_HOME"):
        return
    default_hub = os.path.expanduser("~/.cache/huggingface/hub")
    probe = os.path.join(default_hub, "__wtest")
    try:
        os.makedirs(probe, exist_ok=True)
        os.rmdir(probe)
        return  # default is fine
    except Exception:
        pass
    base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
    hf_home = os.path.join(base, "Jarvis", "hf")
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    print(f"[train] ~/.cache not writable; using HF_HOME={hf_home}")


def main(argv=None) -> int:
    import os

    args = parse_args(argv)
    _ensure_writable_hf_cache()
    # reduce CUDA fragmentation on small GPUs
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    use_unsloth = not args.no_unsloth and _can_import("unsloth")
    if use_unsloth:
        print("[train] using Unsloth backend (low-memory)")
        model, tokenizer = _load_unsloth(args)
    else:
        mode = "4-bit QLoRA" if args.bits == 4 else "bf16/fp16 LoRA"
        print(f"[train] using transformers backend ({mode})")
        model, tokenizer = _load_transformers(args)

    train_ds, val_ds = _load_data(args, tokenizer)
    trainer = _build_trainer(args, model, tokenizer, train_ds, val_ds, use_unsloth)

    print(f"[train] {len(train_ds)} train / {len(val_ds)} val examples; "
          f"starting for {args.epochs} epochs")
    trainer.train()

    adapter_dir = Path(args.out)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[train] LoRA adapter saved to {adapter_dir}")

    if args.merge:
        _merge_and_save(args, model, tokenizer, use_unsloth)
    return 0


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #

def _load_unsloth(args):
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=None,            # auto (bf16/fp16)
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=QWEN_LORA_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    _ensure_pad(tokenizer)
    return model, tokenizer


def _load_transformers(args):
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore

    compute_dtype = torch.bfloat16 if _bf16_ok() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    _ensure_pad(tokenizer)

    load_kwargs = dict(device_map="auto")
    # transformers >=5 renamed torch_dtype -> dtype; support both.
    load_kwargs.update(_dtype_kwarg(AutoModelForCausalLM, compute_dtype))

    if args.bits == 4:
        from transformers import BitsAndBytesConfig  # type: ignore
        from peft import prepare_model_for_kbit_training  # type: ignore

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # plain LoRA in bf16/fp16 - ideal for <=1.5B models, no bitsandbytes.
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=QWEN_LORA_MODULES,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


def _dtype_kwarg(cls, dtype) -> dict:
    """Return {'dtype': ...} or {'torch_dtype': ...} depending on transformers version."""
    import inspect
    try:
        params = inspect.signature(cls.from_pretrained).parameters
    except (ValueError, TypeError):
        params = {}
    if "dtype" in params:
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def _ensure_pad(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def _load_data(args, tokenizer):
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("json", data_files={"train": args.train, "val": args.val})

    def render(batch):
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False,
                                          add_generation_prompt=False)
            for m in batch["messages"]
        ]
        return {"text": texts}

    train = ds["train"].map(render, batched=True, remove_columns=ds["train"].column_names)
    val = ds["val"].map(render, batched=True, remove_columns=ds["val"].column_names)

    if args.limit and args.limit > 0:
        train = train.select(range(min(args.limit, len(train))))
        val = val.select(range(min(max(20, args.limit // 10), len(val))))
    return train, val


def _build_trainer(args, model, tokenizer, train_ds, val_ds, use_unsloth):
    """Build the SFT trainer, adapting to the installed trl/transformers API.

    trl has churned across versions: `max_seq_length`->`max_length`,
    `evaluation_strategy`->`eval_strategy`, and SFTTrainer's `tokenizer`->
    `processing_class`. We inspect the signatures and only pass what's accepted.
    """
    import inspect
    from trl import SFTTrainer, SFTConfig  # type: ignore

    cfg_params = set(inspect.signature(SFTConfig.__init__).parameters)
    want = dict(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_steps=args.save_steps,
        bf16=_bf16_ok(),
        fp16=not _bf16_ok(),
        dataset_text_field="text",
        packing=False,
        report_to="none",
        seed=42,
    )
    # sequence-length arg name differs by version
    if "max_length" in cfg_params:
        want["max_length"] = args.max_seq_len
    elif "max_seq_length" in cfg_params:
        want["max_seq_length"] = args.max_seq_len
    # eval strategy arg name differs by version; only enable when asked
    eval_mode = "steps" if args.eval else "no"
    if "eval_strategy" in cfg_params:
        want["eval_strategy"] = eval_mode
    elif "evaluation_strategy" in cfg_params:
        want["evaluation_strategy"] = eval_mode
    if args.eval and "per_device_eval_batch_size" in cfg_params:
        want["per_device_eval_batch_size"] = 1   # full-vocab logits OOM otherwise
    # 8-bit optimizer only when bitsandbytes is present and we're quantizing
    if "optim" in cfg_params:
        want["optim"] = ("adamw_8bit" if (args.bits == 4 and _can_import("bitsandbytes"))
                         else "adamw_torch")
    if "gradient_checkpointing" in cfg_params:
        want["gradient_checkpointing"] = not use_unsloth

    cfg = SFTConfig(**{k: v for k, v in want.items() if k in cfg_params})

    tr_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    tkwargs = dict(model=model, train_dataset=train_ds, args=cfg)
    if args.eval:
        tkwargs["eval_dataset"] = val_ds
    if "processing_class" in tr_params:
        tkwargs["processing_class"] = tokenizer
    elif "tokenizer" in tr_params:
        tkwargs["tokenizer"] = tokenizer
    return SFTTrainer(**tkwargs)


def _merge_and_save(args, model, tokenizer, use_unsloth):
    out = str(Path(args.out).parent / "jarvis-merged")
    print(f"[train] merging LoRA into base and saving fp16 to {out}")
    if use_unsloth:
        model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")
    else:
        merged = model.merge_and_unload()
        merged.save_pretrained(out, safe_serialization=True)
        tokenizer.save_pretrained(out)
    print(f"[train] merged model at {out} - now run training/export_ollama.py")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _can_import(mod: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def _bf16_ok() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
