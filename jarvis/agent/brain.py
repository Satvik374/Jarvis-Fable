"""The pluggable LLM 'brain'.

A Brain takes the running message history and returns the model's next raw
reply (a JSON action string, per prompts.py). Four backends ship:

  * ``ollama``   - local, default. Talks to the Ollama server (localhost:11434).
  * ``llamacpp`` - local, via a llama.cpp / llama-cpp-python OpenAI server.
  * ``openai``   - any OpenAI-compatible endpoint (LM Studio, vLLM, OpenRouter...).
  * ``anthropic``- Claude API, useful as the strong-vision reference brain.

All backends share one interface so the rest of the app never branches on which
model is in use. Vision (sending the annotated screenshot) is supported by the
ollama and openai/anthropic backends when ``use_vision`` is on.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any

from ..config import BrainConfig


class BrainError(RuntimeError):
    pass


class Brain:
    """Base interface."""

    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg

    def complete(self, system: str, messages: list[dict],
                 image=None) -> str:
        """Return the model's raw reply.

        ``messages`` is a list of {role, content} dicts (user/assistant).
        ``image`` is an optional PIL image for the current turn (vision mode).
        """
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------
    @staticmethod
    def _png_b64(image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _prepare_vision_b64(image, max_dim: int = 1280,
                            quality: int = 80) -> tuple[str, str]:
        """Resize and JPEG-compress a screenshot for vision API calls.

        Returns ``(base64_string, mime_type)``.  A typical 1920x1080 PNG
        screenshot is 3-7 MB base64; after resizing to 1280 px and JPEG
        compression it drops to ~100-300 KB — small enough that the HTTP
        upload never times out.
        """
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


def _coalesce_roles(messages: list[dict]) -> list[dict]:
    """Merge consecutive same-role messages into one (Anthropic needs alternation)."""
    out: list[dict] = []
    for m in messages:
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] = out[-1]["content"] + "\n\n" + m["content"]
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def make_brain(cfg: BrainConfig) -> Brain:
    backend = cfg.backend.lower()
    if backend in {"gemini", "vertex"}:
        return GeminiVertexBrain(cfg)
    if backend == "ollama":
        return OllamaBrain(cfg)
    if backend in {"hf", "transformers", "local"}:
        return HFLocalBrain(cfg)
    if backend in {"llamacpp", "openai", "lmstudio", "vllm"}:
        return OpenAICompatBrain(cfg)
    if backend == "anthropic":
        return AnthropicBrain(cfg)
    raise BrainError(f"unknown brain backend: {cfg.backend}")


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #

class OllamaBrain(Brain):
    """Local models served by Ollama. The recommended default for this rig."""

    def complete(self, system, messages, image=None) -> str:
        import requests  # type: ignore

        msgs: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            msg = {"role": m["role"], "content": m["content"]}
            msgs.append(msg)
        if image is not None and self.cfg.use_vision and msgs:
            msgs[-1]["images"] = [self._png_b64(image)]

        payload = {
            "model": self.cfg.model,
            "messages": msgs,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.max_tokens,
            },
            "format": "json",   # ask Ollama to constrain output to JSON
        }
        try:
            r = requests.post(f"{self.cfg.base_url}/api/chat", json=payload,
                              timeout=self.cfg.request_timeout)
        except Exception as exc:
            raise BrainError(
                f"cannot reach Ollama at {self.cfg.base_url} ({exc}). "
                f"Is it running? Try:  ollama serve   and   "
                f"ollama pull {self.cfg.model}") from exc
        if r.status_code == 404:
            raise BrainError(
                f"Ollama has no model '{self.cfg.model}'. Run: "
                f"ollama pull {self.cfg.model}")
        r.raise_for_status()
        data = r.json()
        return (data.get("message", {}) or {}).get("content", "")


# --------------------------------------------------------------------------- #
# HuggingFace local (base model + your trained LoRA adapter, no server)
# --------------------------------------------------------------------------- #

class HFLocalBrain(Brain):
    """Run a fine-tuned model directly with transformers - the fastest way to
    try YOUR trained adapter without installing Ollama or converting to GGUF.

    Set in config.yaml:
        brain.backend: hf
        brain.model: Qwen/Qwen2.5-0.5B-Instruct        # the base
        brain.adapter_path: training/outputs/jarvis-lora  # your LoRA
    """

    _model = None
    _tokenizer = None
    _loaded_key = None

    def _ensure_loaded(self):
        key = (self.cfg.model, self.cfg.adapter_path)
        if HFLocalBrain._model is not None and HFLocalBrain._loaded_key == key:
            return
        self._redirect_hf_cache()
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(self.cfg.adapter_path or self.cfg.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model, device_map="auto", **_dtype_kwarg(dtype))
        if self.cfg.adapter_path:
            from peft import PeftModel  # type: ignore

            model = PeftModel.from_pretrained(model, self.cfg.adapter_path)
        model.eval()
        HFLocalBrain._model, HFLocalBrain._tokenizer = model, tok
        HFLocalBrain._loaded_key = key

    def complete(self, system, messages, image=None) -> str:
        self._ensure_loaded()
        import torch  # type: ignore

        model, tok = HFLocalBrain._model, HFLocalBrain._tokenizer
        msgs = [{"role": "system", "content": system}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in messages]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        do_sample = self.cfg.temperature and self.cfg.temperature > 0
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=self.cfg.max_tokens,
                do_sample=bool(do_sample),
                temperature=self.cfg.temperature if do_sample else None,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True)

    @staticmethod
    def _redirect_hf_cache():
        import os
        import tempfile

        if os.environ.get("HF_HOME"):
            return
        probe = os.path.expanduser("~/.cache/huggingface/hub/__wtest")
        try:
            os.makedirs(probe, exist_ok=True)
            os.rmdir(probe)
        except Exception:
            base = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
            os.environ["HF_HOME"] = os.path.join(base, "Jarvis", "hf")


def _dtype_kwarg(dtype) -> dict:
    """transformers >=5 uses `dtype`; older uses `torch_dtype`."""
    import inspect

    from transformers import AutoModelForCausalLM  # type: ignore
    try:
        params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
        return {"dtype": dtype} if "dtype" in params else {"torch_dtype": dtype}
    except Exception:
        return {"torch_dtype": dtype}


# --------------------------------------------------------------------------- #
# OpenAI-compatible (llama.cpp server, LM Studio, vLLM, OpenRouter, ...)
# --------------------------------------------------------------------------- #

class OpenAICompatBrain(Brain):
    def complete(self, system, messages, image=None) -> str:
        import requests  # type: ignore

        msgs: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            msgs.append({"role": m["role"], "content": m["content"]})
        if image is not None and self.cfg.use_vision and msgs:
            last = msgs[-1]
            b64 = self._png_b64(image)
            last["content"] = [
                {"type": "text", "text": last["content"]},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]

        headers = {"Content-Type": "application/json"}
        key = getattr(self.cfg, "api_key", "") or os.environ.get(self.cfg.api_key_env, "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": self.cfg.model,
            "messages": msgs,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        base = self.cfg.base_url.rstrip("/")
        url = base + ("/chat/completions" if base.endswith("/v1")
                      else "/v1/chat/completions")
        try:
            r = requests.post(url, json=payload, headers=headers,
                              timeout=self.cfg.request_timeout)
        except Exception as exc:
            raise BrainError(f"cannot reach endpoint {url}: {exc}") from exc
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# Anthropic (strong vision reference brain)
# --------------------------------------------------------------------------- #

class AnthropicBrain(Brain):
    def complete(self, system, messages, image=None) -> str:
        import requests  # type: ignore

        key = getattr(self.cfg, "api_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise BrainError("set ANTHROPIC_API_KEY or api_key in config/env to use the anthropic backend")

        # Anthropic requires strictly alternating roles; our loop emits two
        # consecutive user turns (RESULT then the fresh observation), so merge
        # any consecutive same-role messages first.
        api_msgs = _coalesce_roles(messages)
        if image is not None and self.cfg.use_vision and api_msgs:
            b64 = self._png_b64(image)
            last = api_msgs[-1]
            last["content"] = [
                {"type": "text", "text": last["content"]},
                {"type": "image", "source": {"type": "base64",
                 "media_type": "image/png", "data": b64}},
            ]

        payload = {
            "model": self.cfg.model,
            "system": system,
            "messages": api_msgs,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
        }
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        r = requests.post("https://api.anthropic.com/v1/messages",
                          json=payload, headers=headers,
                          timeout=self.cfg.request_timeout)
        r.raise_for_status()
        data = r.json()
        parts = [b.get("text", "") for b in data.get("content", [])
                 if b.get("type") == "text"]
        return "".join(parts)


# --------------------------------------------------------------------------- #
# Google Cloud Vertex AI (Gemini)
# --------------------------------------------------------------------------- #

class GeminiVertexBrain(Brain):
    """Google Cloud Vertex AI backend using Application Default Credentials (ADC)."""

    def __init__(self, cfg: BrainConfig):
        super().__init__(cfg)
        self.project_id = None
        self._cached_token = None
        self._token_expiry = 0

    def _get_access_token_and_project(self) -> tuple[str, str]:
        if self._cached_token and time.time() < self._token_expiry - 60:
            return self._cached_token, self.project_id

        # Try using google-auth library if installed
        try:
            import google.auth  # type: ignore
            import google.auth.transport.requests  # type: ignore
            credentials, project = google.auth.default()
            request = google.auth.transport.requests.Request()
            credentials.refresh(request)
            self._cached_token = credentials.token
            self.project_id = project or getattr(credentials, 'project_id', None)
            self._token_expiry = time.time() + 3500
            if self._cached_token and self.project_id:
                return self._cached_token, self.project_id
        except ImportError:
            pass
        except Exception:
            pass

        # Manual parsing from ADC JSON
        adc_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not adc_path:
            adc_path = os.path.expandvars(r'%APPDATA%\gcloud\application_default_credentials.json')

        if not os.path.exists(adc_path):
            raise BrainError(
                f"Google Cloud Application Default Credentials (ADC) file not found. "
                f"Please run 'gcloud auth application-default login' or set GOOGLE_APPLICATION_CREDENTIALS env var."
            )

        try:
            with open(adc_path, 'r', encoding='utf-8') as f:
                creds = json.load(f)
        except Exception as exc:
            raise BrainError(f"Failed to read ADC file at {adc_path}: {exc}")

        cred_type = creds.get('type')
        project_id = creds.get('quota_project_id') or creds.get('project_id') or os.environ.get('GOOGLE_CLOUD_PROJECT')
        if not project_id:
            raise BrainError("No project ID found in credentials or environment. Please set GOOGLE_CLOUD_PROJECT or configure your quota project.")

        if cred_type == 'authorized_user':
            token_url = "https://oauth2.googleapis.com/token"
            data = urllib.parse.urlencode({
                'client_id': creds['client_id'],
                'client_secret': creds['client_secret'],
                'refresh_token': creds['refresh_token'],
                'grant_type': 'refresh_token'
            }).encode('utf-8')

            req = urllib.request.Request(token_url, data=data, headers={'Content-Type': 'application/x-www-form-urlencoded'})
            try:
                with urllib.request.urlopen(req) as response:
                    res = json.loads(response.read().decode('utf-8'))
                    self._cached_token = res['access_token']
                    self.project_id = project_id
                    self._token_expiry = time.time() + res.get('expires_in', 3600)
                    return self._cached_token, self.project_id
            except Exception as e:
                err_msg = ""
                if hasattr(e, 'read'):
                    err_msg = f": {e.read().decode()}"
                raise BrainError(f"Failed to refresh access token using ADC credentials{err_msg}") from e
        else:
            raise BrainError(
                f"Unsupported ADC credential type '{cred_type}'. "
                f"Please install the 'google-auth' package via: pip install google-auth"
            )

    def complete(self, system: str, messages: list[dict], image=None) -> str:
        access_token, project_id = self._get_access_token_and_project()

        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            parts = []
            if isinstance(m["content"], str):
                parts.append({"text": m["content"]})
            elif isinstance(m["content"], list):
                for part in m["content"]:
                    if part.get("type") == "text":
                        parts.append({"text": part["text"]})
                    elif part.get("type") == "image_url":
                        pass
            contents.append({"role": role, "parts": parts})

        if image is not None and self.cfg.use_vision and contents:
            for msg in reversed(contents):
                if msg["role"] == "user":
                    b64, mime = self._prepare_vision_b64(image)
                    msg["parts"].append({
                        "inlineData": {
                            "mimeType": mime,
                            "data": b64
                        }
                    })
                    break
            # Gemini's native pointing convention is 0-1000 normalized; pin it
            # down so raw coordinates are unambiguous (denormalized in
            # tools/registry.py). Element ids are still preferred and exact.
            system = (system or "") + (
                "\n\nVISION COORDINATES: prefer clicking by element id from the "
                "list. If you must give raw x/y coordinates, normalize them to "
                "a 0-1000 scale relative to the screenshot: (0,0) is the "
                "top-left corner, (1000,1000) the bottom-right. Never copy "
                "pixel coordinates from the element list into x/y - use the "
                "element id instead.")

        max_tokens = self.cfg.max_tokens
        if max_tokens is None or max_tokens <= 0 or max_tokens > 50000:
            max_tokens = 50000

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.cfg.temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json"
            }
        }

        if system:
            payload["systemInstruction"] = {
                "parts": [{"text": system}]
            }

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        loc = getattr(self.cfg, "location", "global")
        url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{loc}/publishers/google/models/{self.cfg.model}:generateContent"

        import requests as _req  # type: ignore

        try:
            r = _req.post(url, json=payload, headers=headers,
                          timeout=self.cfg.request_timeout)
        except Exception as exc:
            raise BrainError(
                f"Vertex AI Gemini API request failed: {exc}") from exc
        if not r.ok:
            try:
                err_details = f" - Details: {r.text[:500]}"
            except Exception:
                err_details = ""
            raise BrainError(
                f"Vertex AI Gemini API returned HTTP {r.status_code}{err_details}")
        data = r.json()

        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise BrainError("No candidates returned from Gemini API. The response might have been blocked.")
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                finish_reason = candidates[0].get("finishReason")
                raise BrainError(f"Empty content from Gemini. Finish reason: {finish_reason}")
            return parts[0].get("text", "")
        except KeyError as e:
            raise BrainError(f"Unexpected response format from Vertex AI: {data}") from e

