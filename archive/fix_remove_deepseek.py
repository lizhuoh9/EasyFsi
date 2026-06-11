"""Remove the DeepSeek routing block from ~/.claude/settings.json.

Why: the env block routes Claude Code's permission safety classifier
(haiku alias) and subagents (sonnet alias) to deepseek-v4-pro via
api.deepseek.com. When that endpoint is down, every Bash/PowerShell call
and every subagent fails with "deepseek-v4-pro[1m] is temporarily
unavailable". Removing the block reverts all model aliases to genuine
Anthropic models using your existing login.

What it does:
1. Backs up settings.json next to itself (settings.json.bak-deepseek-<timestamp>).
2. Deletes the entire "env" block (the 6 ANTHROPIC_* keys, including the
   DeepSeek ANTHROPIC_AUTH_TOKEN, which must not be sent to the real
   Anthropic API).
3. Leaves everything else (hooks, permissions, model, availableModels) untouched.

Run once, then RESTART the Claude Code session so the new environment takes effect:
    "D:/TOOL/Anaconda/python.exe" fix_remove_deepseek.py
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

EXPECTED_KEYS = {
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
}


def main() -> None:
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        print(f"not found: {settings_path}")
        return

    raw = settings_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    env = data.get("env")
    if not isinstance(env, dict) or not env:
        print("no 'env' block in settings.json - nothing to do")
        return

    unexpected = sorted(set(env) - EXPECTED_KEYS)
    if unexpected:
        print(
            "WARNING: env contains keys beyond the DeepSeek block, they will "
            f"be removed too: {unexpected}"
        )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = settings_path.with_name(f"settings.json.bak-deepseek-{timestamp}")
    shutil.copy2(settings_path, backup_path)

    removed_keys = sorted(env)
    del data["env"]
    settings_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"backup written: {backup_path}")
    print(f"removed env keys: {', '.join(removed_keys)}")
    print("done. Restart your Claude Code session for the change to take effect.")


if __name__ == "__main__":
    main()
