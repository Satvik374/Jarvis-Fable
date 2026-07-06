#!/usr/bin/env python
"""Turn the fine-tuned model into a runnable Ollama model called ``jarvis``.

Pipeline:  merged fp16 HF model  ->  GGUF (quantised)  ->  Ollama Modelfile  ->
``ollama create jarvis``.

Two ways to get the GGUF:

  A) Unsloth (easiest, do it right after training):
        model.save_pretrained_gguf("training/outputs/jarvis-gguf",
                                   tokenizer, quantization_method="q4_k_m")
     That writes ``jarvis-gguf/unsloth.Q4_K_M.gguf``. Point --gguf at it below.

  B) llama.cpp: convert the merged model yourself:
        python <llama.cpp>/convert_hf_to_gguf.py training/outputs/jarvis-merged \
            --outfile training/outputs/jarvis-f16.gguf --outtype f16
        <llama.cpp>/llama-quantize training/outputs/jarvis-f16.gguf \
            training/outputs/jarvis-q4_k_m.gguf q4_k_m

This script's job is to write a correct Modelfile (with Jarvis's system prompt
and the Qwen chatml template baked in) and register it with Ollama:

    python training/export_ollama.py --gguf training/outputs/jarvis-q4_k_m.gguf
    # then, if you didn't pass --create:
    ollama create jarvis -f training/outputs/Modelfile
    # finally point the app at it:  set brain.model: jarvis  in config.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from jarvis.agent.prompts import build_system_prompt  # noqa: E402


# Qwen2.5 / ChatML template, so the bare GGUF chats correctly under Ollama.
_CHATML_TEMPLATE = """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""


def write_modelfile(gguf_path: str, out_path: Path, temperature: float,
                    num_ctx: int, num_predict: int) -> Path:
    system = build_system_prompt().replace('"""', '\\"\\"\\"')
    modelfile = f'''FROM {gguf_path}

TEMPLATE """{_CHATML_TEMPLATE}"""

PARAMETER temperature {temperature}
PARAMETER num_ctx {num_ctx}
PARAMETER num_predict {num_predict}
PARAMETER stop "<|im_end|>"

SYSTEM """{system}"""
'''
    out_path.write_text(modelfile, encoding="utf-8")
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Register the trained model with Ollama")
    ap.add_argument("--gguf", required=True,
                    help="path to the quantised .gguf (see options A/B in the header)")
    ap.add_argument("--name", default="jarvis", help="Ollama model name to create")
    ap.add_argument("--modelfile", default=str(_HERE / "outputs" / "Modelfile"))
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--num-predict", type=int, default=512)
    ap.add_argument("--create", action="store_true",
                    help="run 'ollama create' immediately")
    args = ap.parse_args(argv)

    gguf = Path(args.gguf)
    if not gguf.exists():
        print(f"[export] GGUF not found: {gguf}\n"
              f"         Produce it first (see the header of this file).")
        return 1

    mf = Path(args.modelfile)
    mf.parent.mkdir(parents=True, exist_ok=True)
    # Use an absolute FROM path so 'ollama create' works from any cwd.
    write_modelfile(str(gguf.resolve()), mf, args.temperature,
                    args.num_ctx, args.num_predict)
    print(f"[export] wrote Modelfile -> {mf}")

    create_cmd = ["ollama", "create", args.name, "-f", str(mf)]
    if args.create:
        print(f"[export] running: {' '.join(create_cmd)}")
        try:
            subprocess.run(create_cmd, check=True)
        except FileNotFoundError:
            print("[export] 'ollama' not on PATH - install from https://ollama.com")
            return 1
        except subprocess.CalledProcessError as exc:
            print(f"[export] ollama create failed: {exc}")
            return 1
        print(f"[export] created Ollama model '{args.name}'.")
    else:
        print(f"[export] next:  {' '.join(create_cmd)}")

    print(f"[export] then set  brain.model: {args.name}  in config.yaml "
          f"(or run: python run.py --model {args.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
