from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


CONTROL_TOOL_NAMES = frozenset(
    {
        "spawn_agent",
        "wait_agent",
        "list_agents",
        "send_message",
        "followup_task",
        "interrupt_agent",
    }
)


def parse_timestamp(value: str | None) -> float | None:
    if not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def json_dumps(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def parent_agent_name(agent_name: str | None) -> str | None:
    if not agent_name or agent_name == "/root":
        return None
    normalized = agent_name.rstrip("/")
    if "/" not in normalized[1:]:
        return "/root" if normalized.startswith("/root/") else None
    return normalized.rsplit("/", 1)[0]


def iter_jsonl_binary(path: Path) -> Iterator[tuple[int, int, dict[str, Any]]]:
    """Yield byte offset, byte length, and decoded JSON object for each line."""
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                return
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at {path}:{offset}")
            yield offset, len(line), value


def bounded_json(value: Any, max_bytes: int = 131_072) -> tuple[str, bool]:
    encoded = json_dumps(value)
    raw = encoded.encode("utf-8")
    if len(raw) <= max_bytes:
        return encoded, False
    clipped = raw[:max_bytes].decode("utf-8", errors="ignore")
    return clipped, True

