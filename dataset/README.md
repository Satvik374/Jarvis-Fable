# Jarvis dataset

Two kinds of data feed the Jarvis brain:

### 1. Synthetic dataset (`build_dataset.py`)
Grounded, format-perfect examples generated from scenario templates in
`seeds.py`. Each example is one **turn**: the model is shown the task, the
history of prior actions + tool results, and the current screen observation
(numbered element list), and must output the next action as JSON.

Because the builder imports `jarvis.agent.prompts` (the same
`build_system_prompt`, `format_observation`, `format_decision` the live agent
uses), the training text is byte-identical to what the model sees at inference.

```bash
python dataset/build_dataset.py --n 1500          # ~5.9k examples
python dataset/build_dataset.py --n 4000 --seed 1 # more / different
```

Outputs (in `dataset/data/`):
- `jarvis_sft.train.jsonl` / `jarvis_sft.val.jsonl` — chat `messages` format,
  one JSON object per line: `{"messages": [{"role": "...", "content": "..."}, ...]}`.
- `jarvis_sft.sample.jsonl` — first 15 examples, safe to read / commit.
- `stats.json` — counts and action distribution.

**Add scenarios** by writing a generator in `seeds.py` (return
`(task, trajectory)` where a trajectory is a list of
`((active_window, [(role, name, cx, cy), ...]), (thought, action, args))`
turns) and registering it in `GENERATORS`. The builder handles element
numbering, tool-result simulation, splitting and shuffling.

### 2. Real trajectories (`data/trajectories/`)
Every task you run in the app is appended here as one JSON line (task, model,
and the full step list with observations, decisions and results). This is the
highest-value data — it's *your* apps and *your* tasks. To fold it into
training, convert each logged step into the same per-turn `messages` format and
concatenate with the synthetic `train.jsonl`. (The formats already align; a
small converter is a good next addition.)

## Format at a glance
```json
{"messages": [
  {"role": "system", "content": "You are JARVIS..."},
  {"role": "user", "content": "TASK: open notepad and save a note"},
  {"role": "assistant", "content": "{\"thought\": \"...\", \"action\": \"open_app\", \"args\": {\"name\": \"notepad\"}}"},
  {"role": "user", "content": "RESULT: launched 'notepad'"},
  {"role": "user", "content": "ACTIVE WINDOW: Untitled - Notepad\nSCREEN: 1920x1080\nELEMENTS:\n[0] ...\n\nReply with one action as JSON."},
  {"role": "assistant", "content": "{\"thought\": \"...\", \"action\": \"type\", \"args\": {\"text\": \"...\"}}"}
]}
```
The final `assistant` turn is the training target; everything before it is the
prompt.
