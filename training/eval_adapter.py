#!/usr/bin/env python
"""Quick quality check of a fine-tuned Jarvis adapter (before wiring it into Ollama).

Loads the base model + LoRA adapter, replays validation examples, and reports:
  * JSON-validity rate  (does it emit one parseable action object?)
  * exact action-name match vs the gold decision,
  * element-id match for pointer actions (did it pick the right target?).

    python training/eval_adapter.py --adapter training/outputs/jarvis-lora --n 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from jarvis.agent.prompts import parse_decision  # noqa: E402


def _ensure_writable_hf_cache() -> None:
    """Mirror train_qlora.py: redirect HF cache when ~/.cache is not writable."""
    import os
    import tempfile

    if os.environ.get("HF_HOME"):
        return
    probe = os.path.expanduser("~/.cache/huggingface/hub/__wtest")
    try:
        os.makedirs(probe, exist_ok=True)
        os.rmdir(probe)
        return
    except Exception:
        base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
        os.environ["HF_HOME"] = os.path.join(base, "Jarvis", "hf")


def main(argv=None) -> int:
    _ensure_writable_hf_cache()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="training/outputs/jarvis-lora")
    ap.add_argument("--val", default="dataset/data/jarvis_sft.val.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args(argv)

    model, tokenizer = _load(args)
    rows = _read_val(args.val, args.n)

    valid = name_ok = elem_ok = elem_total = 0
    for ex in rows:
        msgs = ex["messages"]
        gold = json.loads(msgs[-1]["content"])
        prompt_msgs = msgs[:-1]   # everything up to (not including) the target
        text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True)
        reply = _generate(model, tokenizer, text, args.max_new_tokens)
        pred = parse_decision(reply)

        if pred.raw and _is_json(reply):
            valid += 1
        if pred.action == gold.get("action"):
            name_ok += 1
        if "element" in gold.get("args", {}):
            elem_total += 1
            if pred.args.get("element") == gold["args"]["element"]:
                elem_ok += 1

    n = len(rows)
    print(f"\nEvaluated {n} validation examples:")
    print(f"  JSON valid        : {valid}/{n}  ({100*valid/n:.1f}%)")
    print(f"  action name match : {name_ok}/{n}  ({100*name_ok/n:.1f}%)")
    if elem_total:
        print(f"  element id match  : {elem_ok}/{elem_total}  "
              f"({100*elem_ok/elem_total:.1f}%)")
    return 0


def _load(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from peft import PeftModel  # type: ignore
    import torch  # type: ignore

    tok = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    return model, tok


def _generate(model, tokenizer, text, max_new_tokens):
    import torch  # type: ignore

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=None, top_p=None,
                             pad_token_id=tokenizer.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def _read_val(path, n):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))
        if len(rows) >= n:
            break
    return rows


def _is_json(text: str) -> bool:
    try:
        json.loads(text.strip().strip("`"))
        return True
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
