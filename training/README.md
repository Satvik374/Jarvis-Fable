# Training the Jarvis brain

Fine-tune a small base model (LoRA/QLoRA) so it emits Jarvis's action JSON, then
serve it back to the app via Ollama.

## Verified run on this machine (RTX 2050, 4 GB VRAM, 8 GB RAM)

This exact command was used to train a real adapter here:

```bash
set HF_HOME=%LOCALAPPDATA%\Jarvis\hf          &:: (the script sets this for you)
python training/train_qlora.py ^
  --model Qwen/Qwen2.5-0.5B-Instruct ^
  --bits 16 ^                # plain bf16 LoRA - 0.5B is small, no need for 4-bit
  --limit 600 ^              # subset that finishes in ~30 min on this rig
  --epochs 1 --batch 1 --grad-accum 6 --max-seq-len 1024 ^
  --save-steps 20 --merge --out training/outputs/jarvis-lora
```

Three things learned the hard way on this hardware, now handled automatically:

1. **HF cache is redirected.** `~/.cache/huggingface` is ACL-locked here
   (WinError 5), so downloads fail. The script detects this and sets
   `HF_HOME=%LOCALAPPDATA%\Jarvis\hf` for you.
2. **In-loop eval is OFF by default.** Computing eval logits over Qwen's 152k
   vocab needs ~4.6 GB and OOMs a 4 GB GPU. Evaluate *after* training with
   `eval_adapter.py` (generation-based, batch 1, low memory). Pass `--eval`
   only if you have more VRAM.
3. **`--bits 16` (plain LoRA) for <=1.5B models.** 4-bit QLoRA is for big
   models; for 0.5-1.5B, bf16 LoRA is simpler, avoids bitsandbytes quirks, and
   fits easily in 4 GB.

**RAM is the real bottleneck**, not VRAM. With <1 GB free, each step thrashes
swap (~22 s/step); with ~2 GB free it's ~18 s/step. Close other apps before a
big run. To train the **full** dataset (all ~6.5k examples, 3 epochs) you'll
want a machine with more RAM, or a free Colab T4 (see below) - then copy the
adapter back and export to Ollama.

To scale up on the same rig, raise `--limit` (or drop it for everything) and add
epochs; the `--save-steps` checkpoints mean a stopped run is never wasted.

## 0. Prerequisites
- The dataset: `python dataset/build_dataset.py --n 1500` (creates
  `dataset/data/jarvis_sft.{train,val}.jsonl`).
- A **separate** Python env for training (the deps are large and unrelated to
  the app):
  ```bash
  # CUDA build of torch first (cu121 shown; match your CUDA):
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install -r training/requirements-train.txt
  # STRONGLY recommended on 4 GB VRAM (halves memory, can export GGUF):
  pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
  ```

## 1. Train (QLoRA)
```bash
python training/train_qlora.py \
  --model unsloth/Qwen2.5-3B-Instruct \
  --epochs 3 --batch 1 --grad-accum 16 --max-seq-len 1536 \
  --merge
```
- Uses **Unsloth** automatically if installed, else transformers + bitsandbytes
  4-bit. Both train a LoRA adapter into `training/outputs/jarvis-lora`.
- `--merge` also writes a merged fp16 model to `training/outputs/jarvis-merged`
  (needed for GGUF export).

### Fitting 4 GB VRAM (RTX 2050)
A 3B QLoRA is *tight* on 4 GB. If you hit CUDA OOM:
- **Use Unsloth** (biggest single win).
- Drop to a 1.5B base: `--model unsloth/Qwen2.5-1.5B-Instruct`.
- Lower `--max-seq-len 1024`, keep `--batch 1`, raise `--grad-accum` to keep the
  effective batch size.
- Or train on a **free Colab T4 (16 GB)**: upload `jarvis_sft.*.jsonl`, run the
  same command there, download `jarvis-merged/`, and continue from step 2 locally.

## 2. Sanity-check the adapter
```bash
python training/eval_adapter.py --adapter training/outputs/jarvis-lora --n 100
```
Reports JSON-validity, action-name accuracy, and element-id accuracy on the
validation split. Expect high numbers — the task is narrow and consistent.

## 3. Convert to GGUF (for Ollama)
**Easiest (Unsloth), right after training** — add to a short script or run in a
REPL with the trained `model`, `tokenizer`:
```python
model.save_pretrained_gguf("training/outputs/jarvis-gguf",
                           tokenizer, quantization_method="q4_k_m")
# -> training/outputs/jarvis-gguf/unsloth.Q4_K_M.gguf
```
**Or with llama.cpp:**
```bash
python <llama.cpp>/convert_hf_to_gguf.py training/outputs/jarvis-merged \
    --outfile training/outputs/jarvis-f16.gguf --outtype f16
<llama.cpp>/llama-quantize training/outputs/jarvis-f16.gguf \
    training/outputs/jarvis-q4_k_m.gguf q4_k_m
```

## 4. Register with Ollama and use it
```bash
python training/export_ollama.py --gguf training/outputs/jarvis-gguf/unsloth.Q4_K_M.gguf --create
# writes a Modelfile (system prompt + chatml template baked in) and runs
# `ollama create jarvis`.
python run.py --model jarvis          # the app now runs on YOUR fine-tune
```
Or set `brain.model: jarvis` in `config.yaml`.

## 5. Improve it over time
1. Use Jarvis (with `:confirm on`) on real tasks — runs are logged to
   `dataset/data/trajectories/`.
2. Convert those to training examples and mix with the synthetic set.
3. Re-run steps 1–4. Each cycle grounds the model more in *your* apps.

## Files
| File | Purpose |
|------|---------|
| `train_qlora.py` | QLoRA fine-tune (Unsloth or transformers+bnb). |
| `eval_adapter.py` | Validation-set quality check. |
| `export_ollama.py` | GGUF → Modelfile → `ollama create jarvis`. |
| `requirements-train.txt` | Training-only dependencies. |
