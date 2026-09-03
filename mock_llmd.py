"""Stand-in for the llm-d inference gateway: echoes the flow control headers.

Returns an OpenAI-shaped chat completion whose message content is a JSON object
of every x-llm-d-* header the request arrived with, so the POC can assert on
what the proxy actually sent rather than on what it logged.
"""

from __future__ import annotations

import json
import time

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()


@app.get("/v1/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [{"id": "RedHatAI/gemma-4-12B-it-FP8-Dynamic", "object": "model", "owned_by": "llm-d"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict:
    body = await request.json()
    seen = {k.lower(): v for k, v in request.headers.items() if k.lower().startswith(("x-llm-d-", "x-litellm-"))}
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(seen)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
