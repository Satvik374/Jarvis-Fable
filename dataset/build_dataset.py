#!/usr/bin/env python
"""Build the Jarvis-specific fine-tuning dataset.

Run from the project root:

    python dataset/build_dataset.py --n 1500

It expands the scenario generators in ``seeds.py`` into chat-format SFT
examples. Crucially, every example is constructed with the SAME system prompt,
observation formatting, and decision formatting that the live agent uses at
inference time (imported from ``jarvis.agent.prompts``), so there is zero
train/inference skew.

Design: one training example per *turn*. For a k-step trajectory we emit k
examples; example i contains the full prefix (task + prior decisions + prior
tool results + the current screen observation) and targets the i-th decision.
This mirrors the runtime loop, which drops stale screenshots and keeps only
(decision, result) history plus the current observation.

Outputs (under dataset/data/):
    jarvis_sft.train.jsonl   - training split  ({"messages": [...]} per line)
    jarvis_sft.val.jsonl     - validation split
    jarvis_sft.sample.jsonl  - first 15 examples, for eyeballing
    stats.json               - counts and action distribution
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

# make both the project root (for `jarvis`) and this dir (for `seeds`) importable
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from jarvis.agent.prompts import (build_system_prompt, format_observation,  # noqa: E402
                                  format_decision)
from jarvis.perception.elements import Element, Observation  # noqa: E402
import seeds  # noqa: E402

_RESOLUTIONS = [(1280, 720), (1366, 768), (1920, 1080), (1600, 900)]
_KEY_ALIASES = {"win": "winleft", "esc": "escape", "return": "enter"}


# --------------------------------------------------------------------------- #
# turning a seed observation into the exact runtime user-turn text
# --------------------------------------------------------------------------- #

def _observation(active_window: str, elements, screen_size) -> Observation:
    els = []
    for i, (role, name, cx, cy) in enumerate(elements):
        els.append(Element(id=i, role=role, name=name,
                           bbox=(cx - 12, cy - 8, cx + 12, cy + 8),
                           center=(cx, cy)))
    return Observation(elements=els, screen_size=screen_size,
                       active_window=active_window)


def _resolve_center(args: dict, obs: Observation):
    if args.get("element") is not None:
        el = obs.by_id(int(args["element"]))
        return el.center if el else (0, 0)
    if args.get("x") is not None and args.get("y") is not None:
        return int(args["x"]), int(args["y"])
    return None


def simulate_result(action: str, args: dict, obs: Observation) -> str:
    """Reproduce the tool return string the runtime would feed back as RESULT."""
    if action in {"click", "double_click", "triple_click", "right_click"}:
        pt = _resolve_center(args, obs) or (0, 0)
        button = "right" if action == "right_click" else "left"
        clicks = 1
        if action == "double_click":
            clicks = 2
        elif action == "triple_click":
            clicks = 3
        elif action == "click":
            clicks = int(args.get("count", 1))
        suffix = f" x{clicks}" if clicks > 1 else ""
        return f"{button}-clicked ({pt[0]},{pt[1]}){suffix}"
    if action == "key_sequence":
        keys = args.get("keys", [])
        return "pressed sequence: " + " -> ".join(str(k) for k in keys)
    if action == "move":
        return f"moved mouse to ({args.get('x')},{args.get('y')})"
    if action == "drag":
        src = _resolve_center({"element": args.get("from_element"),
                               "x": args.get("x1"), "y": args.get("y1")}, obs)
        dst = _resolve_center({"element": args.get("to_element"),
                               "x": args.get("x2"), "y": args.get("y2")}, obs)
        return f"dragged {src} -> {dst}"
    if action == "scroll":
        return f"scrolled dy={args.get('dy', 3)} dx={args.get('dx', 0)}"
    if action == "type":
        t = str(args.get("text", ""))
        preview = t if len(t) <= 40 else t[:37] + "..."
        return f'typed "{preview}"'
    if action == "press":
        parts = [_KEY_ALIASES.get(k.strip().lower(), k.strip().lower())
                 for k in str(args.get("keys", "")).split("+") if k.strip()]
        return f"pressed {'+'.join(parts)}"
    if action == "open_app":
        return f"launched '{args.get('name')}'"
    if action == "focus_window":
        return f"focused '{args.get('title')}'"
    if action == "open_url":
        url = str(args.get("url", ""))
        if not url.startswith(("http://", "https://", "file://")):
            url = "https://" + url
        return f"opened {url}"
    if action == "run_command":
        return "exit code 0\nstdout:\n<command output>"
    if action == "write_file":
        return f"wrote {len(str(args.get('content','')))} chars to {args.get('path')}"
    if action == "read_file":
        return "<file contents>"
    if action == "list_dir":
        return f"{args.get('path', '.')}  (entries listed)"
    if action == "clipboard_write":
        return "copied to clipboard"
    if action == "clipboard_read":
        return "clipboard: <text>"
    if action == "wait":
        return f"waited {args.get('seconds', 1.0)}s"
    if action == "observe":
        return "re-reading the screen"
    return "ok"


# --------------------------------------------------------------------------- #
# expand a trajectory into per-turn training examples
# --------------------------------------------------------------------------- #

def trajectory_examples(system: str, task: str, trajectory) -> list[dict]:
    examples: list[dict] = []
    # persistent history: task, then (decision, result) pairs - NO stale obs,
    # exactly like the runtime loop.
    history: list[dict] = [{"role": "user", "content": f"TASK: {task}"}]

    for (active_window, elements), (thought, action, args) in trajectory:
        screen_size = random.choice(_RESOLUTIONS)
        obs = _observation(active_window, elements, screen_size)
        obs_turn = {"role": "user",
                    "content": format_observation(active_window, screen_size,
                                                  obs.menu())}
        target = format_decision(thought, action, args)

        messages = ([{"role": "system", "content": system}]
                    + history + [obs_turn]
                    + [{"role": "assistant", "content": target}])
        examples.append({"messages": messages})

        # advance persistent history for the next turn
        history.append({"role": "assistant", "content": target})
        if action not in {"finish", "ask"}:
            history.append({"role": "user",
                            "content": f"RESULT: {simulate_result(action, args, obs)}"})
    return examples


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the Jarvis SFT dataset")
    ap.add_argument("--n", type=int, default=1500,
                    help="number of trajectories to generate")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out", default=str(_HERE / "data"))
    args = ap.parse_args(argv)

    random.seed(args.seed)
    system = build_system_prompt()
    generators = seeds.weighted_generators()

    all_examples: list[dict] = []
    action_counts: Counter = Counter()
    traj_len_counts: Counter = Counter()

    for _ in range(args.n):
        gen = random.choice(generators)
        task, trajectory = gen("")
        traj_len_counts[len(trajectory)] += 1
        exs = trajectory_examples(system, task, trajectory)
        for ex in exs:
            action_counts[json.loads(ex["messages"][-1]["content"])["action"]] += 1
        all_examples.extend(exs)

    random.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * args.val_frac))
    val, train = all_examples[:n_val], all_examples[n_val:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "jarvis_sft.train.jsonl", train)
    _write_jsonl(out / "jarvis_sft.val.jsonl", val)
    _write_jsonl(out / "jarvis_sft.sample.jsonl", all_examples[:15])

    stats = {
        "trajectories": args.n,
        "examples_total": len(all_examples),
        "train": len(train),
        "val": len(val),
        "avg_turns_per_trajectory": round(
            sum(k * v for k, v in traj_len_counts.items()) / max(1, args.n), 2),
        "action_distribution": dict(action_counts.most_common()),
        "trajectory_length_distribution": dict(sorted(traj_len_counts.items())),
        "system_prompt_chars": len(system),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Generated {len(all_examples)} examples from {args.n} trajectories")
    print(f"  train: {len(train)}   val: {len(val)}")
    print(f"  actions: {dict(action_counts.most_common())}")
    print(f"  written to {out}/")
    return 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
