"""Configuration loading.

Order of precedence (highest first):
  1. environment variables (``JARVIS_*``)
  2. ``config.yaml`` next to the project root
  3. built-in defaults below

The defaults are chosen for a modest machine (RTX 2050 / 4 GB VRAM / 8 GB RAM):
a small quantised model served by Ollama, with graceful fallbacks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


@dataclass
class BrainConfig:
    # backend: "gemini" | "ollama" | "hf" | "llamacpp" | "openai" | "anthropic"
    #   gemini - Google Cloud Vertex AI (default)
    #   ollama - local server
    #   hf     - load a HuggingFace model + your trained LoRA adapter directly,
    #            no server needed. Best way to run your fine-tune immediately.
    backend: str = "gemini"
    # Model name/tag. For gemini this is e.g. "gemini-3.5-flash".
    # For ollama this is the pulled model, e.g. "ornith:9b".
    # For hf this is the base model id, e.g. "Qwen/Qwen2.5-0.5B-Instruct".
    # Once you fine-tune, point this at "jarvis" (see training/export_ollama.py).
    model: str = "gemini-3.5-flash"
    # Google Cloud Vertex AI region/location (used by "gemini" backend).
    location: str = "global"
    # Path to a trained LoRA adapter (used by the "hf" backend). Empty = base only.
    adapter_path: str = ""
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"   # used by openai/anthropic backends
    api_key: str = ""                     # direct API key value from env/config
    temperature: float = 0.2
    max_tokens: int = 50000
    # If true, send the annotated screenshot to a vision model each step.
    # Defaults to True for Gemini backend.
    use_vision: bool = True
    request_timeout: int = 120
    # Let Jarvis answer plain conversation (greetings, small talk, general
    # questions) directly, with no tools/perception. Set false to force every
    # input through the computer-control loop.
    conversational: bool = True


@dataclass
class PerceptionConfig:
    # Use the Windows UI Automation tree to label elements (fast, no GPU).
    use_uia: bool = True
    # OCR fallback for apps with no accessibility info (games, canvases).
    # Heavy (pulls torch); off by default.
    use_ocr: bool = False
    # Cap on labelled elements handed to the model, to keep the prompt small.
    max_elements: int = 60
    # Save annotated screenshots for debugging / dataset collection.
    save_screenshots: bool = True


@dataclass
class SafetyConfig:
    # Ask for confirmation before each action (recommended while learning).
    confirm_each_action: bool = False
    # Hard stop after this many steps to avoid runaway loops.
    max_steps: int = 50
    # Refuse shell commands matching these (case-insensitive substrings).
    blocked_command_patterns: tuple[str, ...] = (
        "format ", "del /", "rmdir /s", "rm -rf", "diskpart", "mkfs",
        ":(){", "shutdown", "reg delete", "cipher /w",
    )
    # Directories writes/deletes are confined to (empty = home dir only guard off).
    allow_paths: tuple[str, ...] = ()


@dataclass
class DataConfig:
    # Log every (observation, action) step to build a real agentic dataset.
    collect_trajectories: bool = True
    trajectory_dir: str = "dataset/data/trajectories"
    # Reinforcement learning: after the model claims a task is finished, run a
    # verification pass (model judges the final screen state) before rewarding
    # the plan. Prevents fake successes from polluting memory.txt.
    verify_success: bool = True
    # Cap on memory.txt size (chars). Oldest learned entries are dropped first
    # so the system prompt never bloats.
    memory_max_chars: int = 4000
    # Conversational memory: how many recent (user prompt, Jarvis response)
    # exchanges to feed back in for continuity across tasks/sessions.
    chat_history_turns: int = 10


@dataclass
class Config:
    brain: BrainConfig = field(default_factory=BrainConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply(dc: Any, values: dict[str, Any]) -> None:
    for key, val in (values or {}).items():
        if hasattr(dc, key):
            setattr(dc, key, val)


def load_config(path: Path | str | None = None) -> Config:
    # Load .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # Fallback manual parser to prevent new dependencies
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v

    cfg = Config()
    p = Path(path) if path else CONFIG_PATH
    if p.exists():
        data = _read_yaml(p)
        _apply(cfg.brain, data.get("brain", {}))
        _apply(cfg.perception, data.get("perception", {}))
        _apply(cfg.safety, data.get("safety", {}))
        _apply(cfg.data, data.get("data", {}))

    # Environment overrides for the most common knobs.
    env = os.environ
    if v := env.get("JARVIS_BACKEND") or env.get("BACKEND"):
        cfg.brain.backend = v
    if v := env.get("JARVIS_MODEL") or env.get("MODEL_ID") or env.get("MODEL"):
        cfg.brain.model = v
    if v := env.get("JARVIS_BASE_URL") or env.get("BASE_URL"):
        cfg.brain.base_url = v
    if v := env.get("JARVIS_LOCATION") or env.get("LOCATION"):
        cfg.brain.location = v
    if v := env.get("JARVIS_API_KEY") or env.get("API_KEY") or env.get("OPENAI_API_KEY"):
        cfg.brain.api_key = v
        # Ensure it is set in environment for standard libraries
        os.environ["OPENAI_API_KEY"] = v

    # Automatic fallback to openai backend if custom URL or API key is set
    # but backend is still the default ollama
    if cfg.brain.backend == "ollama":
        has_custom_key = bool(cfg.brain.api_key)
        has_custom_url = cfg.brain.base_url and "11434" not in cfg.brain.base_url
        if has_custom_key or has_custom_url:
            cfg.brain.backend = "openai"

    if env.get("JARVIS_VISION") in {"1", "true", "True"}:
        cfg.brain.use_vision = True
    if env.get("JARVIS_CONFIRM") in {"1", "true", "True"}:
        cfg.safety.confirm_each_action = True
    return cfg


def _read_yaml(p: Path) -> dict:
    try:
        import yaml  # type: ignore

        with p.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        # Fall back to a minimal parser so the app runs before deps install.
        return _read_yaml_minimal(p)


def _read_yaml_minimal(p: Path) -> dict:
    """A tiny nested key: value reader for the simple config we ship.

    Only supports two levels of ``key:`` sections with scalar leaves - enough
    for config.yaml when PyYAML is not yet installed.
    """
    result: dict[str, dict] = {}
    section: str | None = None
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.rstrip().endswith(":"):
            section = line.strip()[:-1]
            result[section] = {}
        elif section and ":" in line:
            k, _, v = line.strip().partition(":")
            result[section][k.strip()] = _coerce(v.strip())
    return result


def _coerce(v: str) -> Any:
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v.strip('"\'')
