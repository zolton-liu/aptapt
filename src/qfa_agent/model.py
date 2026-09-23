"""Model adapters for the organizer endpoint, a local command, and replay tests."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from .types import Message


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelReply:
    content: str | dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""


class Model(Protocol):
    name: str

    def complete(self, messages: Sequence[Message], *, timeout_sec: float) -> ModelReply: ...


class ReplayModel:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        responses = payload.get("responses") if isinstance(payload, dict) else payload
        if not isinstance(responses, list) or not responses:
            raise ModelError("replay file must contain a non-empty response list")
        self._responses = list(responses)
        self._index = 0
        self.name = f"replay:{path.name}"

    def complete(self, messages: Sequence[Message], *, timeout_sec: float) -> ModelReply:
        del messages, timeout_sec
        if self._index >= len(self._responses):
            raise ModelError("replay responses exhausted")
        value = self._responses[self._index]
        self._index += 1
        if not isinstance(value, (str, dict)):
            raise ModelError("each replay response must be a string or object")
        return ModelReply(content=value)


class CommandModel:
    """Local adapter: JSON messages on stdin, JSON/string action on stdout."""

    def __init__(self, command: str):
        self.argv = shlex.split(command)
        if not self.argv:
            raise ModelError("QFA_MODEL_COMMAND is empty")
        self.name = f"command:{self.argv[0]}"

    def complete(self, messages: Sequence[Message], *, timeout_sec: float) -> ModelReply:
        payload = json.dumps({"messages": list(messages)}, ensure_ascii=False).encode()
        try:
            result = subprocess.run(
                self.argv,
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=max(0.1, timeout_sec),
                check=False,
                shell=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ModelError(f"local model command failed: {type(exc).__name__}") from exc
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[-1000:]
            raise ModelError(f"local model command exited {result.returncode}: {stderr}")
        text = result.stdout.decode("utf-8", errors="replace").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return ModelReply(content=text)
        if isinstance(parsed, dict) and "content" in parsed:
            return ModelReply(content=parsed["content"])
        if isinstance(parsed, dict):
            return ModelReply(content=parsed)
        return ModelReply(content=text)


class HouseEndpointModel:
    """Minimal OpenAI-compatible client using only the standard library."""

    def __init__(self, endpoint: str, model_name: str, seed: int, token: str = ""):
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/v1/chat/completions"):
            pass
        elif endpoint.endswith("/chat/completions"):
            endpoint = endpoint.removesuffix("/chat/completions") + "/v1/chat/completions"
        elif endpoint.endswith("/v1"):
            endpoint += "/chat/completions"
        else:
            endpoint += "/v1/chat/completions"
        self.endpoint = endpoint
        self.name = model_name
        self.seed = seed
        self.token = token

    def complete(self, messages: Sequence[Message], *, timeout_sec: float) -> ModelReply:
        payload = {
            "model": self.name,
            "messages": list(messages),
            "temperature": 0,
            "seed": self.seed,
            "max_tokens": min(4000, max(1, int(os.environ.get("QFA_MAX_RESPONSE_TOKENS", "4000")))),
            # Deliberately no `tools`, retrieval, web search, or remote code execution.
        }
        if os.environ.get("QFA_JSON_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
            # Useful for local OpenAI-compatible servers (including Ollama) whose
            # chat models otherwise tend to wrap tool actions in Markdown.
            # It stays opt-in because the organizer's endpoint contract does not
            # require support for response_format.
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.monotonic()
        body: bytes | None = None
        last_error: BaseException | None = None
        max_attempts = 3
        for attempt in range(max_attempts):
            remaining = float(timeout_sec) - (time.monotonic() - started)
            if remaining <= 0:
                break
            try:
                with urllib.request.urlopen(request, timeout=max(0.1, remaining)) as response:
                    body = response.read(16 * 1024 * 1024)
                break
            except urllib.error.HTTPError as exc:
                last_error = exc
                if attempt == max_attempts - 1 or exc.code not in {408, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt == max_attempts - 1:
                    break
            # Two bounded retries handle transient local/House disconnects
            # without creating an unbounded request loop.
            time.sleep(min(0.5 * (attempt + 1), max(0.0, remaining / 20.0)))
        if body is None:
            kind = type(last_error).__name__ if last_error is not None else "TimeoutError"
            raise ModelError(f"house model request failed after bounded retries: {kind}") from last_error
        try:
            decoded = json.loads(body)
            content = decoded["choices"][0]["message"]["content"]
            finish_reason = str(decoded["choices"][0].get("finish_reason") or "")
            usage = decoded.get("usage") or {}
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ModelError("house endpoint returned an invalid chat-completions response") from exc
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        if not isinstance(content, str):
            raise ModelError("house endpoint returned non-text message content")
        return ModelReply(
            content=content,
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            finish_reason=finish_reason,
        )


def model_from_environment() -> Model:
    replay = os.environ.get("QFA_REPLAY_FILE")
    if replay:
        return ReplayModel(Path(replay))
    command = os.environ.get("QFA_MODEL_COMMAND")
    if command:
        return CommandModel(command)
    endpoint = os.environ.get("MODEL_ENDPOINT")
    name = os.environ.get("MODEL_NAME")
    if endpoint and name:
        seed = int(os.environ.get("QFBENCH_SEED", "0"))
        token = os.environ.get("MODEL_TOKEN", "")
        if os.environ.get("QFBENCH_NETWORK") == "restricted" and not token:
            raise ModelError("MODEL_TOKEN is required in the official restricted runtime")
        return HouseEndpointModel(endpoint, name, seed, token)
    raise ModelError(
        "no model configured: official runs must provide MODEL_ENDPOINT and MODEL_NAME; "
        "local runs can set QFA_REPLAY_FILE or QFA_MODEL_COMMAND"
    )
