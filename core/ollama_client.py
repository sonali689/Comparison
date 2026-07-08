"""
core/ollama_client.py

Thin wrapper around the local Ollama HTTP API. This only ever talks to
http://localhost:11434 (or wherever your Ollama instance runs). No cloud
calls. Requires `ollama serve` running, and the model already pulled:
    ollama pull qwen2.5vl:7b
    ollama pull qwen2.5:7b
"""

import base64
import json
import re
import requests

import config


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _strip_fences(raw: str) -> str:
    return re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()


def chat_json(prompt: str, image_paths: list = None, model: str = None,
              temperature: float = 0.0, timeout: int = 600) -> dict:
    """
    Sends a prompt (optionally with images) to a local Ollama model and
    parses the response as JSON. Raises ValueError if the model didn't
    return clean JSON -- surface that to the user rather than guessing.
    """
    model = model or config.OLLAMA_TEXT_MODEL
    
    message = {"role": "user", "content": prompt}
    if image_paths:
        message["images"] = [_b64(p) for p in image_paths]

    payload = {
        "model": model,
        "messages": [message],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 16384,
            "num_predict": 4096
        },
    }
    
    resp = requests.post(f"{config.OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
    
    # --- CRITICAL FIX: Force the exact error to show in Streamlit ---
    if resp.status_code != 200:
        raise RuntimeError(f"OLLAMA REJECTED REQUEST: {resp.text}")
    # ----------------------------------------------------------------
        
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]
    cleaned = _strip_fences(raw)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model '{model}' did not return clean JSON: {e}\nRaw:\n{raw[:1000]}")


def is_available() -> bool:
    try:
        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False