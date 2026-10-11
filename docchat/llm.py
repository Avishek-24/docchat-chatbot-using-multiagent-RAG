
from __future__ import annotations

import json
import re
from typing import Iterator, List, Optional

import requests

DEFAULT_HOST = "http://localhost:11434"


class LLMError(RuntimeError):
    """Raised with a human-friendly message when the local model can't be used."""


def _drop_think(tokens: Iterator[str]) -> Iterator[str]:
    """Hide <think>...</think> blocks that some reasoning models emit."""
    buf, hidden = "", False
    for tok in tokens:
        buf += tok
        out = ""
        while buf:
            if hidden:
                i = buf.find("</think>")
                if i < 0:
                    buf = buf[-8:]  # keep a tail in case the tag is split
                    break
                buf, hidden = buf[i + 8:], False
            else:
                i = buf.find("<think>")
                if i >= 0:
                    out += buf[:i]
                    buf, hidden = buf[i + 7:], True
                elif "<" in buf[-7:]:  # maybe the start of a tag; wait for more
                    j = buf.rfind("<")
                    out += buf[:j]
                    buf = buf[j:]
                    break
                else:
                    out += buf
                    buf = ""
        if out:
            yield out
    if buf and not hidden:
        yield buf


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return re.sub(r"<think>.*$", "", text, flags=re.S).strip()


class OllamaClient:
    def __init__(self, host: str = DEFAULT_HOST, model: str = "",
                 timeout: int = 600, num_ctx: int = 6144):
        self.host = (host or DEFAULT_HOST).strip().rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx

    # ---- discovery -------------------------------------------------------
    def list_models(self) -> Optional[List[str]]:
        """Installed model names; None if the Ollama server can't be reached."""
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=3)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return sorted(n for n in names if n)
        except Exception:
            return None

    # ---- internals -------------------------------------------------------
    def _payload(self, prompt: str, system: Optional[str], temperature: float,
                 max_tokens: int, stream: bool, json_mode: bool = False) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        if json_mode:
            payload["format"] = "json"
        return payload

    def _explain(self, resp: requests.Response) -> str:
        try:
            msg = resp.json().get("error", resp.text)
        except Exception:
            msg = resp.text
        if "not found" in str(msg).lower():
            return (f"The model '{self.model}' isn't installed. "
                    f"Run `ollama pull {self.model}` in a terminal, then try again.")
        return f"Ollama returned an error: {msg}"

    def _post(self, payload: dict, stream: bool) -> requests.Response:
        try:
            resp = requests.post(f"{self.host}/api/chat", json=payload,
                                 stream=stream, timeout=(5, self.timeout))
        except requests.exceptions.ConnectionError:
            raise LLMError(
                f"Can't reach Ollama at {self.host}. Open the Ollama app "
                "(or run `ollama serve`) and try again.") from None
        except requests.exceptions.Timeout:
            raise LLMError("Ollama took too long to respond. Try a smaller model.") from None
        if resp.status_code != 200:
            raise LLMError(self._explain(resp))
        return resp

    # ---- public API ------------------------------------------------------
    def stream(self, prompt: str, system: Optional[str] = None,
               temperature: float = 0.2, max_tokens: int = 700) -> Iterator[str]:
        resp = self._post(self._payload(prompt, system, temperature, max_tokens, True), True)

        def raw() -> Iterator[str]:
            try:
                with resp:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        if "error" in data:
                            raise LLMError(f"Ollama error: {data['error']}")
                        piece = data.get("message", {}).get("content", "")
                        if piece:
                            yield piece
                        if data.get("done"):
                            break
            except requests.exceptions.RequestException:
                raise LLMError("The connection to Ollama was interrupted.") from None

        return _drop_think(raw())

    def complete(self, prompt: str, system: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 400,
                 json_mode: bool = False) -> str:
        resp = self._post(
            self._payload(prompt, system, temperature, max_tokens, False, json_mode), False)
        try:
            text = resp.json().get("message", {}).get("content", "")
        except ValueError:
            raise LLMError("Ollama sent an unreadable response.") from None
        return _strip_think(text)
