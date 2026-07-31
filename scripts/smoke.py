"""Smoke test: run one task through the Anthropic adapter, with caching.

Usage: python scripts/smoke.py --task eob-001 [--no-cache]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.adapters.anthropic import AnthropicAdapter
from harness.cache import ResponseCache, make_cache_key
from harness.config import MODELS
from harness.prompts import load_prompt, prompt_hash
from harness.tasks import load_tasks

PROMPT_NAME = "extract_v1"
MAX_TOKENS = 2000
TEMPERATURE = 0.0


async def run(task_id: str, use_cache: bool) -> None:
    tasks = load_tasks("tasks/")
    task = next((t for t in tasks if t.id == task_id), None)
    if task is None:
        raise SystemExit(f"No task with id {task_id!r} found under tasks/")

    prompt = load_prompt(PROMPT_NAME).replace("{document}", task.input)
    template_hash = prompt_hash(PROMPT_NAME)

    adapter = AnthropicAdapter(model_id=MODELS["sonnet"])
    params = {"max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}
    key = make_cache_key(prompt, adapter.model_id, params, template_hash)

    cache = ResponseCache(enabled=use_cache)
    cached = cache.get(key)
    if cached is not None:
        response = cached
        cache_status = "HIT"
    else:
        response = await adapter.complete(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        cache.set(key, response)
        cache_status = "MISS"

    print(f"task_id:       {task.id}")
    print(f"model_id:      {response.model_id}")
    print(f"finish_reason: {response.finish_reason}")
    print(f"tokens_in:     {response.tokens_in}")
    print(f"tokens_out:    {response.tokens_out}")
    print(f"latency_ms:    {response.latency_ms:.2f}")
    print(f"cost_usd:      {response.cost_usd:.6f}")
    print(f"cache:         {cache_status}")
    print("text:")
    print(response.text)
    print("raw:")
    print(json.dumps(response.raw, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task id, e.g. eob-001")
    parser.add_argument("--no-cache", action="store_true", help="Disable the response cache")
    args = parser.parse_args()

    asyncio.run(run(args.task, use_cache=not args.no_cache))


if __name__ == "__main__":
    main()
