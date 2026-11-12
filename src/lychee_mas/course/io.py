"""Atomic artifacts and asynchronous OpenAI-compatible CPU client."""
from __future__ import annotations

import asyncio
import contextvars
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

TASK = contextvars.ContextVar("course_task", default="")
PURPOSE = contextvars.ContextVar("course_purpose", default="")


def write_json(path: str | Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class JSONLines:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, data: dict) -> None:
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False) + "\n")
            stream.flush()


class ChatClient:
    def __init__(self, base_url: str, model: str, concurrency: int, log_path: Path,
                 max_tokens: int = 768, timeout: float = 180, api_key: str = "", *,
                 proposal_max_tokens: int = 1536):
        if type(proposal_max_tokens) is not int or proposal_max_tokens < 1:
            raise ValueError("proposal_max_tokens must be a positive integer")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.max_tokens = model, max_tokens
        self.proposal_max_tokens = proposal_max_tokens
        self.timeout, self.api_key = timeout, api_key
        self.semaphore = asyncio.Semaphore(concurrency)
        self.events = JSONLines(log_path)
        self.calls: list[dict] = []

    def _request(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.url, json.dumps(payload).encode(), headers)
        # Loopback inference must not inherit an external HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    async def call(self, prompt: str, *, temperature: float = 0.0,
                   purpose: str | None = None, json_mode: bool = False,
                   max_tokens: int | None = None) -> str:
        payload = {
            "model": self.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens or self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        started = time.monotonic()
        for attempt in range(3):
            try:
                async with self.semaphore:
                    response = await asyncio.to_thread(self._request, payload)
                choice = response["choices"][0]
                message = choice["message"]
                content = message.get("content") or ""
                if not content.strip():
                    raise ValueError("model returned empty content")
                if message.get("reasoning_content"):
                    raise ValueError("server returned reasoning although enable_thinking=false")
                event = {
                    "task_id": TASK.get(), "purpose": purpose or PURPOSE.get(),
                    "model": self.model, "attempt": attempt, "prompt": prompt,
                    "content": content, "usage": response.get("usage", {}),
                    "finish_reason": choice.get("finish_reason"),
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "enable_thinking": False, "temperature": temperature,
                    "max_tokens": payload["max_tokens"],
                }
                self.calls.append(event)
                self.events.append(event)
                if choice.get("finish_reason") == "length":
                    raise ValueError(f"generation reached max_tokens={payload['max_tokens']} "
                                     f"({event['purpose']}); output is incomplete")
                return content.strip()
            except (urllib.error.URLError, TimeoutError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code not in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {error.code}: {error.read().decode()[:800]}") from error
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    async def generate(self, prompt: str) -> str:
        return await self.call(prompt)

    async def compare(self, prompt: str) -> str:
        from ..methods.prerun.maspo.textops import parse_comparison_result
        response = await self.call(prompt, purpose="maspo_compare", max_tokens=512)
        try:
            parse_comparison_result(response)
        except ValueError:
            response = await self.call(
                prompt + "\nYour comparison must end with one valid choice tag. "
                "Return only <choose>A</choose> or <choose>B</choose>.",
                purpose="maspo_compare_format_retry", max_tokens=128)
            parse_comparison_result(response)
        return response

    async def propose(self, prompt: str) -> str:
        return await self.call(prompt, purpose="maspo_propose", temperature=0.7,
                               max_tokens=self.proposal_max_tokens)

    async def judge(self, prompt: str) -> str:
        return await self.call(prompt, purpose="adv2_judge", json_mode=True, max_tokens=768)

    def statistics(self, task_id: str | None = None) -> dict:
        events = self.calls if task_id is None else [c for c in self.calls if c["task_id"] == task_id]
        return {
            "model_calls": len(events),
            "prompt_tokens": sum(c["usage"].get("prompt_tokens", 0) for c in events),
            "completion_tokens": sum(c["usage"].get("completion_tokens", 0) for c in events),
            "call_elapsed_s": round(sum(c["elapsed_s"] for c in events), 3),
            "truncated_calls": sum(c["finish_reason"] == "length" for c in events),
        }
