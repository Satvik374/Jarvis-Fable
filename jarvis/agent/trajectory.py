"""Record real agent runs as training data.

Every task Jarvis performs is appended to a JSONL file as an ordered list of
(observation, decision, result) steps. This is the *real* agentic dataset: as
you use Jarvis, you accumulate grounded examples of screen-state -> action that
can be mixed into the synthetic dataset and fine-tuned on (see training/).

Each line is one full task:
  {
    "task": "...",
    "backend": "ollama", "model": "ornith:9b",
    "steps": [
      {"active_window": "...", "elements": [...], "menu": "...",
       "thought": "...", "action": "click", "args": {...},
       "result": "left-clicked (512,40)", "ok": true},
      ...
    ],
    "outcome": "finish", "summary": "...", "success": null
  }
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class Step:
    active_window: str
    elements: list[dict]
    menu: str
    thought: str
    action: str
    args: dict[str, Any]
    result: str
    ok: bool


@dataclass
class Trajectory:
    task: str
    backend: str
    model: str
    steps: list[Step] = field(default_factory=list)
    outcome: str = ""
    summary: str = ""
    started: float = field(default_factory=time.time)
    # Reward label: set by the success verifier after a claimed finish
    # (True = verified success, False = claimed but failed verification,
    # None = unverified). This is what makes trajectories usable as RL data.
    success: Any = None

    def add(self, step: Step) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(time.time() - self.started, 2)
        return d


class TrajectoryWriter:
    def __init__(self, directory: str | Path, enabled: bool = True):
        self.enabled = enabled
        self.dir = Path(directory)
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"traj_{time.strftime('%Y%m%d')}.jsonl"

    def save(self, traj: Trajectory) -> Path | None:
        if not self.enabled:
            return None
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
        return self.path
