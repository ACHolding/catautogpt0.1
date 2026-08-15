#!/usr/bin/env python3
"""Vibe-train CatSeek-GPU 0.1 and dump everything into auto_gpt_workspace.

Runs a short corpus of prompts through the real 14B GGUF, scores replies, and
writes a soul prompt + training log under:

    auto_gpt_workspace/catseek-gpu-0.1/vibe_train/
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from colorama import Fore, init as colorama_init

colorama_init(autoreset=True)

VIBE_CORPUS = [
    {
        "id": "hello",
        "messages": [
            {"role": "system", "content": "You are CatSeek-GPU 0.1, a local agent LLM."},
            {"role": "user", "content": "Reply with exactly: CATSEEK_OK"},
        ],
        "expect_substr": "CATSEEK",
    },
    {
        "id": "json_cmd",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Auto-GPT's brain. Always reply with a single JSON object "
                    "containing keys thought and command."
                ),
            },
            {
                "role": "user",
                "content": 'Goal: list files. Emit JSON only, e.g. '
                '{"thought":"...","command":{"name":"list_files","args":{}}}',
            },
        ],
        "expect_substr": "{",
    },
    {
        "id": "workspace",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a one-sentence plan to save all agent outputs into "
                    "auto_gpt_workspace. Be concrete."
                ),
            }
        ],
        "expect_substr": "workspace",
    },
    {
        "id": "code",
        "messages": [
            {
                "role": "user",
                "content": "Write a Python function add(a,b) that returns a+b. Code only.",
            }
        ],
        "expect_substr": "def ",
    },
]


def main() -> int:
    from autogpt.llm.providers import catseek_bake, catseek_engine

    print(Fore.CYAN + f"==> CatSeek-GPU 0.1 vibe-train in {ROOT}")
    gguf = catseek_bake.ensure_catseek_gguf(project_root_path=ROOT)
    catseek_bake.write_env_model_path(gguf, ROOT / ".env")
    path = catseek_engine.warm_start()
    ws = catseek_engine.workspace_root() / "vibe_train"
    ws.mkdir(parents=True, exist_ok=True)

    results = []
    passed = 0
    for item in VIBE_CORPUS:
        t0 = time.time()
        print(Fore.YELLOW + f"  vibe · {item['id']} …")
        reply = catseek_engine.create_chat_completion_raw(
            item["messages"], temperature=0.2, max_tokens=256
        )
        content = reply.choices[0].message["content"]
        ok = item["expect_substr"].lower() in (content or "").lower()
        passed += int(ok)
        row = {
            "id": item["id"],
            "ok": ok,
            "latency_s": round(time.time() - t0, 3),
            "prompt_tokens": reply.usage.prompt_tokens,
            "completion_tokens": reply.usage.completion_tokens,
            "content": content,
        }
        results.append(row)
        (ws / f"{item['id']}.txt").write_text(content or "", encoding="utf-8")
        status = Fore.GREEN + "PASS" if ok else Fore.RED + "FAIL"
        print(f"    {status}{Fore.RESET} · {row['latency_s']}s · {content[:80]!r}")

    soul = (
        "You are CatSeek-GPU 0.1 — a local DeepSeek-R1-Distill-Qwen-14B agent LLM "
        "running inside catautogpt. Prefer concise, actionable replies. When Auto-GPT "
        "asks for commands, emit strict JSON. Write all durable artifacts under "
        "auto_gpt_workspace/. Never invent API keys. Think briefly, then act.\n"
    )
    (ws / "soul.txt").write_text(soul, encoding="utf-8")
    # Also drop a copy the agent can read as a file.
    agent_soul = ROOT / "auto_gpt_workspace" / "catseek_soul.txt"
    agent_soul.parent.mkdir(parents=True, exist_ok=True)
    agent_soul.write_text(soul, encoding="utf-8")

    report = {
        "model_id": catseek_engine.MODEL_ID,
        "gguf": str(path),
        "passed": passed,
        "total": len(VIBE_CORPUS),
        "score": round(passed / max(1, len(VIBE_CORPUS)), 3),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    (ws / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        Fore.GREEN
        + f"==> vibe-train done · {passed}/{len(VIBE_CORPUS)} · "
        + f"artifacts → {ws}"
    )
    return 0 if passed >= 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
