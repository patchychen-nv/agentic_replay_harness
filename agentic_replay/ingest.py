from __future__ import annotations

import json
import hashlib
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .database import connect, initialize
from .util import (
    CONTROL_TOOL_NAMES,
    bounded_json,
    iter_jsonl_binary,
    json_dumps,
    parent_agent_name,
    parse_json_object,
    parse_timestamp,
)


SUPPORTED_ROUTES = frozenset({"/v1/responses", "/v1/chat/completions"})

# Fields emitted on response-side objects but normally stripped before the
# same item is submitted as request input.  Removing them makes the content
# fingerprint stable without weakening explicit item_id/call_id evidence.
_TRANSIENT_ITEM_FIELDS = frozenset(
    {
        "status",
        "phase",
        "logprobs",
        "annotations",
        "encrypted_content",
        "async_",
        "caller",
        "namespace",
    }
)


def _canonical_item(value: Any, *, include_identity: bool = True) -> Any:
    def normalize(item: Any, depth: int) -> Any:
        if isinstance(item, dict):
            normalized: dict[str, Any] = {}
            for key, child in item.items():
                if key in _TRANSIENT_ITEM_FIELDS:
                    continue
                if depth == 0 and not include_identity and key in {"id", "call_id"}:
                    continue
                normalized[str(key)] = normalize(child, depth + 1)
            return normalized
        if isinstance(item, list):
            return [normalize(child, depth + 1) for child in item]
        return item

    return normalize(value, 0)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json_dumps(value).encode("utf-8")).hexdigest()


def _item_index_values(
    item: Any,
) -> tuple[str, str | None, str | None, str | None, str | None, str, str]:
    if isinstance(item, dict):
        item_type = str(item.get("type") or ("message" if item.get("role") else "object"))
        item_id = str(item["id"]) if item.get("id") is not None else None
        call_id = str(item["call_id"]) if item.get("call_id") is not None else None
        role = str(item["role"]) if item.get("role") is not None else None
        name = str(item["name"]) if item.get("name") is not None else None
    else:
        item_type = "text" if isinstance(item, str) else type(item).__name__
        item_id = call_id = role = name = None
    canonical = _canonical_item(item)
    content = _canonical_item(item, include_identity=False)
    return (
        item_type,
        item_id,
        call_id,
        role,
        name,
        _sha256_json(canonical),
        _sha256_json(content),
    )


def _record_indexed_items(
    connection: sqlite3.Connection,
    table: str,
    request_id: str,
    items: Any,
) -> int:
    if table not in {"request_input_items", "response_output_items"}:
        raise ValueError(f"unsupported lineage item table: {table}")
    connection.execute(f"DELETE FROM {table} WHERE request_id=?", (request_id,))
    if not isinstance(items, list):
        return 0
    rows = []
    for ordinal, item in enumerate(items):
        rows.append((request_id, ordinal, *_item_index_values(item)))
    connection.executemany(
        f"""
        INSERT INTO {table}(
          request_id, ordinal, item_type, item_id, call_id, role, name,
          canonical_sha256, content_sha256
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def _message_type(content: Any) -> str | None:
    if isinstance(content, str):
        text = content
    else:
        try:
            text = json_dumps(content)
        except (TypeError, ValueError):
            return None
    match = re.search(r"Message Type:\s*([A-Z_]+)", text)
    if match:
        return match.group(1)
    for candidate in ("FINAL_ANSWER", "MESSAGE"):
        if candidate in text:
            return candidate
    return None


def _record_agent_messages(
    connection: sqlite3.Connection,
    trace_id: str,
    recorded_at: str | None,
    send_ts: float | None,
    request_input: Any,
) -> int:
    if not isinstance(request_input, list):
        return 0
    count = 0
    for item in request_input:
        if not isinstance(item, dict) or item.get("type") != "agent_message":
            continue
        author = item.get("author")
        recipient = item.get("recipient")
        content = item.get("content")
        content_bytes = json_dumps(content).encode("utf-8")
        content_sha = hashlib.sha256(content_bytes).hexdigest()
        message_id = item.get("id") or hashlib.sha256(
            f"{author}\0{recipient}\0{content_sha}".encode("utf-8")
        ).hexdigest()
        item_metadata = {
            "id": item.get("id"),
            "type": "agent_message",
            "author": author,
            "recipient": recipient,
        }
        connection.execute(
            """
            INSERT INTO agent_messages(
              message_id, surfaced_in_request_id, surfaced_at, surfaced_ts,
              author_agent_name, recipient_agent_name, message_type,
              content_sha256, item_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
              surfaced_in_request_id=excluded.surfaced_in_request_id,
              surfaced_at=excluded.surfaced_at,
              surfaced_ts=excluded.surfaced_ts,
              author_agent_name=excluded.author_agent_name,
              recipient_agent_name=excluded.recipient_agent_name,
              message_type=excluded.message_type,
              content_sha256=excluded.content_sha256,
              item_json=excluded.item_json
            WHERE excluded.surfaced_ts < agent_messages.surfaced_ts
            """,
            (
                str(message_id),
                trace_id,
                recorded_at,
                send_ts,
                author,
                recipient,
                _message_type(content),
                content_sha,
                json_dumps(item_metadata),
            ),
        )
        count += 1
    return count


def _discover(attempt_dir: Path) -> tuple[list[Path], list[Path], list[Path]]:
    request_root = attempt_dir / "request_trace"
    if attempt_dir.name == "request_trace":
        request_root = attempt_dir
        attempt_dir = attempt_dir.parent
    if not request_root.exists():
        raise FileNotFoundError(f"request trace directory does not exist: {request_root}")
    request_files = sorted(request_root.glob("*/requests-*.jsonl"))
    response_files = sorted(request_root.glob("*/responses-*.jsonl"))
    perf_files = sorted((attempt_dir / "perf_metrics").glob("perf_metrics-disagg-*.jsonl"))
    if not request_files:
        raise FileNotFoundError(f"no requests-*.jsonl files below {request_root}")
    if not response_files:
        raise FileNotFoundError(f"no responses-*.jsonl files below {request_root}")
    return request_files, response_files, perf_files


def _source_relative(path: Path, attempt_dir: Path) -> str:
    try:
        return str(path.relative_to(attempt_dir))
    except ValueError:
        return str(path)


def _register_source(
    connection: sqlite3.Connection,
    path: Path,
    attempt_dir: Path,
    kind: str,
) -> int:
    stat = path.stat()
    relative = _source_relative(path, attempt_dir)
    connection.execute(
        """
        INSERT INTO sources(relative_path, absolute_path, kind, size_bytes, mtime_ns)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(relative_path) DO UPDATE SET
          absolute_path=excluded.absolute_path,
          kind=excluded.kind,
          size_bytes=excluded.size_bytes,
          mtime_ns=excluded.mtime_ns
        """,
        (relative, str(path.absolute()), kind, stat.st_size, stat.st_mtime_ns),
    )
    row = connection.execute(
        "SELECT source_id FROM sources WHERE relative_path=?", (relative,)
    ).fetchone()
    assert row is not None
    return int(row[0])


def _iter_sse_events(body: Any) -> Iterable[dict[str, Any]]:
    chunks: list[str] = []
    if isinstance(body, str):
        chunks = [body]
    elif isinstance(body, list):
        chunks = [part for part in body if isinstance(part, str)]
    for chunk in chunks:
        for block in chunk.split("\n\n"):
            for line in block.splitlines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].lstrip()
                if not data or data == "[DONE]":
                    continue
                # Token deltas dominate large traces and contain none of the
                # indexing fields. Avoid millions of redundant JSON decodes.
                relevant = (
                    '"response.completed"' in data
                    or '"response.failed"' in data
                    or '"response.incomplete"' in data
                    or '"response.output_item.done"' in data
                    or '"usage"' in data
                    or '"finish_reason"' in data
                )
                if not relevant:
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event


def _response_payload(record: dict[str, Any]) -> tuple[Any, str | None]:
    response = record.get("response")
    if not isinstance(response, dict):
        return None, None
    kind = response.get("kind")
    if "body" in response:
        return response.get("body"), kind
    if "frames" in response:
        return response.get("frames"), kind
    return response, kind


def _usage_values(usage: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    details = usage.get("input_tokens_details", usage.get("prompt_tokens_details", {}))
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return _as_int(input_tokens), _as_int(output_tokens), _as_int(cached)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _parse_response(
    record: dict[str, Any],
) -> tuple[
    int | None,
    int | None,
    int | None,
    str | None,
    list[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str | None,
]:
    payload, _ = _response_payload(record)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    finish_reason: str | None = None
    output_types: list[str] = []
    output_items: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    parse_error: str | None = None

    events = list(_iter_sse_events(payload))
    if not events and isinstance(payload, dict):
        events = [payload]

    try:
        for event in events:
            event_type = event.get("type")
            if event_type == "response.output_item.done":
                item = event.get("item")
                if not isinstance(item, dict):
                    continue
                output_items.append(item)
                item_type = str(item.get("type") or "unknown")
                output_types.append(item_type)
                name = item.get("name")
                if name:
                    tool_calls.append(
                        {
                            "call_id": item.get("call_id") or item.get("id"),
                            "item_type": item_type,
                            "name": str(name),
                            "arguments": item.get("arguments", item.get("input")),
                        }
                    )
            if event_type in {"response.completed", "response.failed", "response.incomplete"}:
                final_response = event.get("response")
                if isinstance(final_response, dict):
                    usage = final_response.get("usage")
                    if isinstance(usage, dict):
                        input_tokens, output_tokens, cached_tokens = _usage_values(usage)
                    finish_reason = final_response.get("finish_reason") or finish_reason
                    output = final_response.get("output")
                    if isinstance(output, list) and not output_types:
                        fallback_items = [item for item in output if isinstance(item, dict)]
                        output_items.extend(fallback_items)
                        output_types.extend(
                            str(item.get("type") or "unknown") for item in fallback_items
                        )

            usage = event.get("usage")
            if isinstance(usage, dict):
                values = _usage_values(usage)
                input_tokens = values[0] if values[0] is not None else input_tokens
                output_tokens = values[1] if values[1] is not None else output_tokens
                cached_tokens = values[2] if values[2] is not None else cached_tokens

            choices = event.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
        return (
            input_tokens,
            output_tokens,
            cached_tokens,
            finish_reason,
            output_types,
            output_items,
            tool_calls,
            parse_error,
        )
    except (TypeError, ValueError, KeyError) as error:
        parse_error = f"{type(error).__name__}: {error}"
        return (
            input_tokens,
            output_tokens,
            cached_tokens,
            finish_reason,
            output_types,
            output_items,
            tool_calls,
            parse_error,
        )


def _progress(kind: str, count: int, byte_count: int, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    gib = byte_count / (1024**3)
    print(
        f"[{kind}] {count:,} records, {gib:.2f} GiB, {gib / elapsed:.2f} GiB/s",
        file=sys.stderr,
        flush=True,
    )


def _ingest_responses(
    connection: sqlite3.Connection,
    attempt_dir: Path,
    files: list[Path],
    progress_every: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    started = time.monotonic()
    bytes_seen = 0
    records_seen = 0
    for path in files:
        source_id = _register_source(connection, path, attempt_dir, "response")
        for offset, length, record in iter_jsonl_binary(path):
            records_seen += 1
            bytes_seen += length
            trace_id = record.get("trace_id")
            if not trace_id:
                counts["missing_trace_id"] += 1
                continue
            payload, response_kind = _response_payload(record)
            (
                input_tokens,
                output_tokens,
                cached_tokens,
                finish_reason,
                output_types,
                output_items,
                tool_calls,
                parse_error,
            ) = _parse_response(record)
            connection.execute(
                """
                INSERT INTO responses(
                  trace_id, response_status, finished_at, finish_ts,
                  disagg_request_id, termination_json, response_kind,
                  input_tokens, output_tokens, cached_tokens, finish_reason,
                  output_types_json, source_id, line_offset, line_length, parse_error
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                  response_status=excluded.response_status,
                  finished_at=excluded.finished_at,
                  finish_ts=excluded.finish_ts,
                  disagg_request_id=excluded.disagg_request_id,
                  termination_json=excluded.termination_json,
                  response_kind=excluded.response_kind,
                  input_tokens=excluded.input_tokens,
                  output_tokens=excluded.output_tokens,
                  cached_tokens=excluded.cached_tokens,
                  finish_reason=excluded.finish_reason,
                  output_types_json=excluded.output_types_json,
                  source_id=excluded.source_id,
                  line_offset=excluded.line_offset,
                  line_length=excluded.line_length,
                  parse_error=excluded.parse_error
                """,
                (
                    str(trace_id),
                    record.get("status"),
                    record.get("finished_at"),
                    parse_timestamp(record.get("finished_at")),
                    _as_int(record.get("disagg_request_id")),
                    json_dumps(record.get("termination"))
                    if record.get("termination") is not None
                    else None,
                    response_kind,
                    input_tokens,
                    output_tokens,
                    cached_tokens,
                    finish_reason,
                    json_dumps(output_types),
                    source_id,
                    offset,
                    length,
                    parse_error,
                ),
            )
            _record_indexed_items(
                connection,
                "response_output_items",
                str(trace_id),
                output_items,
            )
            for ordinal, tool_call in enumerate(tool_calls):
                name = str(tool_call.get("name") or "")
                call_id = tool_call.get("call_id") or f"{trace_id}:{ordinal}"
                arguments = tool_call.get("arguments")
                arguments_json = None
                if name in CONTROL_TOOL_NAMES and arguments is not None:
                    arguments_json = (
                        arguments if isinstance(arguments, str) else json_dumps(arguments)
                    )
                connection.execute(
                    """
                    INSERT OR REPLACE INTO response_tool_calls(
                      call_id, trace_id, ordinal, item_type, name,
                      arguments_json, is_control
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(call_id),
                        str(trace_id),
                        ordinal,
                        tool_call.get("item_type"),
                        name,
                        arguments_json,
                        int(name in CONTROL_TOOL_NAMES),
                    ),
                )
            counts[str(record.get("status") or "unknown")] += 1
            if parse_error:
                counts["parse_error"] += 1
            if progress_every and records_seen % progress_every == 0:
                connection.commit()
                _progress("responses", records_seen, bytes_seen, started)
        connection.commit()
    _progress("responses", records_seen, bytes_seen, started)
    return counts


def _merged_metadata(body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    client = body.get("client_metadata")
    if not isinstance(client, dict):
        return {}, False
    nested_raw = client.get("x-codex-turn-metadata")
    nested_ok = True
    nested: dict[str, Any] = {}
    if nested_raw:
        nested = parse_json_object(nested_raw)
        nested_ok = bool(nested)
    merged = dict(nested)
    for key, value in client.items():
        if key != "x-codex-turn-metadata" and value is not None:
            merged[key] = value
    return merged, nested_ok


def _request_input(body: dict[str, Any]) -> tuple[Any, int]:
    if "input" in body:
        value = body.get("input")
    else:
        value = body.get("messages")
    return value, len(value) if isinstance(value, list) else int(value is not None)


def _ingest_requests(
    connection: sqlite3.Connection,
    attempt_dir: Path,
    files: list[Path],
    progress_every: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    control_ids = {
        str(row[0])
        for row in connection.execute(
            "SELECT call_id FROM response_tool_calls WHERE is_control=1"
        )
    }
    started = time.monotonic()
    bytes_seen = 0
    records_seen = 0
    for path in files:
        source_id = _register_source(connection, path, attempt_dir, "request")
        for offset, length, record in iter_jsonl_binary(path):
            records_seen += 1
            bytes_seen += length
            trace_id = record.get("trace_id")
            if not trace_id:
                counts["missing_trace_id"] += 1
                continue
            body = record.get("body")
            if not isinstance(body, dict):
                body = {}
            metadata, nested_ok = _merged_metadata(body)
            raw_client_metadata = body.get("client_metadata")
            if (
                not nested_ok
                and isinstance(raw_client_metadata, dict)
                and raw_client_metadata.get("x-codex-turn-metadata")
            ):
                counts["invalid_nested_metadata"] += 1
            session_id = metadata.get("session_id") or record.get("session")
            thread_id = metadata.get("thread_id")
            turn_id = metadata.get("turn_id")
            root_turn_id = metadata.get("root_turn_id") or turn_id
            tree_id = (
                str(root_turn_id)
                if root_turn_id
                else f"session:{session_id}"
                if session_id
                else f"thread:{thread_id}"
                if thread_id
                else f"request:{trace_id}"
            )
            agent_name = metadata.get("agent_name")
            request_input, input_count = _request_input(body)
            route = record.get("route")
            eligibility_reason = "pending_response_validation"
            replay_eligible = int(
                route in SUPPORTED_ROUTES and bool(thread_id) and bool(request_input)
            )
            if route not in SUPPORTED_ROUTES:
                eligibility_reason = "unsupported_route"
            elif not thread_id:
                eligibility_reason = "missing_thread_id"
            elif not request_input:
                eligibility_reason = "missing_input"

            connection.execute(
                """
                INSERT INTO requests(
                  trace_id, route, request_status, recorded_at, send_ts,
                  server_arrival_time, trace_session_id, session_id, thread_id,
                  turn_id, root_turn_id, tree_id, agent_name, parent_agent_name,
                  request_kind, context_window_id, window_id,
                  turn_started_at_unix_ms, model, stream, prompt_cache_key,
                  input_item_count, source_id, line_offset, line_length,
                  replay_eligible, eligibility_reason
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                  route=excluded.route,
                  request_status=excluded.request_status,
                  recorded_at=excluded.recorded_at,
                  send_ts=excluded.send_ts,
                  server_arrival_time=excluded.server_arrival_time,
                  trace_session_id=excluded.trace_session_id,
                  session_id=excluded.session_id,
                  thread_id=excluded.thread_id,
                  turn_id=excluded.turn_id,
                  root_turn_id=excluded.root_turn_id,
                  tree_id=excluded.tree_id,
                  agent_name=excluded.agent_name,
                  parent_agent_name=excluded.parent_agent_name,
                  request_kind=excluded.request_kind,
                  context_window_id=excluded.context_window_id,
                  window_id=excluded.window_id,
                  turn_started_at_unix_ms=excluded.turn_started_at_unix_ms,
                  model=excluded.model,
                  stream=excluded.stream,
                  prompt_cache_key=excluded.prompt_cache_key,
                  input_item_count=excluded.input_item_count,
                  source_id=excluded.source_id,
                  line_offset=excluded.line_offset,
                  line_length=excluded.line_length,
                  replay_eligible=excluded.replay_eligible,
                  eligibility_reason=excluded.eligibility_reason
                """,
                (
                    str(trace_id),
                    route,
                    record.get("status"),
                    record.get("recorded_at"),
                    parse_timestamp(record.get("recorded_at")),
                    record.get("server_arrival_time"),
                    record.get("session"),
                    session_id,
                    thread_id,
                    turn_id,
                    root_turn_id,
                    tree_id,
                    agent_name,
                    parent_agent_name(str(agent_name)) if agent_name else None,
                    metadata.get("request_kind"),
                    metadata.get("context_window_id"),
                    metadata.get("window_id")
                    or metadata.get("x-codex-window-id"),
                    _as_int(metadata.get("turn_started_at_unix_ms")),
                    body.get("model"),
                    int(bool(body.get("stream"))),
                    body.get("prompt_cache_key"),
                    input_count,
                    source_id,
                    offset,
                    length,
                    replay_eligible,
                    eligibility_reason,
                ),
            )

            _record_indexed_items(
                connection,
                "request_input_items",
                str(trace_id),
                request_input,
            )

            counts["agent_message_occurrences"] += _record_agent_messages(
                connection,
                str(trace_id),
                record.get("recorded_at"),
                parse_timestamp(record.get("recorded_at")),
                request_input,
            )

            if isinstance(request_input, list) and control_ids:
                for item in request_input:
                    if not isinstance(item, dict):
                        continue
                    call_id = item.get("call_id")
                    item_type = str(item.get("type") or "")
                    if call_id not in control_ids or not item_type.endswith("_output"):
                        continue
                    output_json, truncated = bounded_json(item.get("output"))
                    observed_ts = parse_timestamp(record.get("recorded_at"))
                    connection.execute(
                        """
                        INSERT INTO control_tool_results(
                          call_id, observed_in_request_id, observed_at,
                          observed_ts, output_json, output_truncated
                        ) VALUES(?, ?, ?, ?, ?, ?)
                        ON CONFLICT(call_id) DO UPDATE SET
                          observed_in_request_id=CASE
                            WHEN excluded.observed_ts < observed_ts THEN excluded.observed_in_request_id
                            ELSE observed_in_request_id END,
                          observed_at=CASE
                            WHEN excluded.observed_ts < observed_ts THEN excluded.observed_at
                            ELSE observed_at END,
                          observed_ts=MIN(observed_ts, excluded.observed_ts),
                          output_json=CASE
                            WHEN excluded.observed_ts < observed_ts THEN excluded.output_json
                            ELSE output_json END,
                          output_truncated=CASE
                            WHEN excluded.observed_ts < observed_ts THEN excluded.output_truncated
                            ELSE output_truncated END
                        """,
                        (
                            str(call_id),
                            str(trace_id),
                            record.get("recorded_at"),
                            observed_ts,
                            output_json,
                            int(truncated),
                        ),
                    )
            counts[str(route or "unknown")] += 1
            if progress_every and records_seen % progress_every == 0:
                connection.commit()
                _progress("requests", records_seen, bytes_seen, started)
        connection.commit()
    _progress("requests", records_seen, bytes_seen, started)
    return counts


def _read_indexed_record(
    handles: dict[Path, Any],
    path: Path,
    offset: int,
    length: int,
) -> dict[str, Any]:
    handle = handles.get(path)
    if handle is None:
        handle = path.open("rb")
        handles[path] = handle
    handle.seek(offset)
    raw = handle.read(length)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}:{offset}")
    return value


def enrich_lineage_items(
    database_path: Path,
    *,
    tree_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Backfill prompt-lineage item fingerprints using stored byte locators.

    This upgrades an existing index without rescanning or rebuilding the raw
    capture.  Passing ``tree_ids`` makes the I/O proportional to the selected
    trees (the 10-request canary reads only its 20 request/response records).
    """

    connection = connect(database_path)
    initialize(connection)
    selected = sorted({str(value) for value in tree_ids or []})
    where = ""
    parameters: list[Any] = []
    if selected:
        placeholders = ",".join("?" for _ in selected)
        where = f"WHERE q.tree_id IN ({placeholders})"
        parameters.extend(selected)
    rows = list(
        connection.execute(
            f"""
            SELECT q.trace_id, q.tree_id,
                   qs.absolute_path AS request_path,
                   q.line_offset AS request_offset,
                   q.line_length AS request_length,
                   rs.absolute_path AS response_path,
                   s.line_offset AS response_offset,
                   s.line_length AS response_length
            FROM requests q
            JOIN sources qs ON qs.source_id=q.source_id
            LEFT JOIN responses s ON s.trace_id=q.trace_id
            LEFT JOIN sources rs ON rs.source_id=s.source_id
            {where}
            ORDER BY q.tree_id, q.send_ts, q.trace_id
            """,
            parameters,
        )
    )
    request_ids = [str(row["trace_id"]) for row in rows]
    if request_ids:
        connection.execute(
            "CREATE TEMP TABLE IF NOT EXISTS selected_lineage_requests "
            "(request_id TEXT PRIMARY KEY)"
        )
        connection.execute("DELETE FROM selected_lineage_requests")
        connection.executemany(
            "INSERT INTO selected_lineage_requests(request_id) VALUES(?)",
            ((request_id,) for request_id in request_ids),
        )
        connection.execute(
            "DELETE FROM request_input_items WHERE request_id IN "
            "(SELECT request_id FROM selected_lineage_requests)"
        )
        connection.execute(
            "DELETE FROM response_output_items WHERE request_id IN "
            "(SELECT request_id FROM selected_lineage_requests)"
        )
        connection.execute(
            "DELETE FROM prompt_chains WHERE target_request_id IN "
            "(SELECT request_id FROM selected_lineage_requests) "
            "OR source_request_id IN "
            "(SELECT request_id FROM selected_lineage_requests)"
        )

    handles: dict[Path, Any] = {}
    input_items = 0
    output_items = 0
    responses = 0
    try:
        for row in rows:
            request_id = str(row["trace_id"])
            request_record = _read_indexed_record(
                handles,
                Path(str(row["request_path"])),
                int(row["request_offset"]),
                int(row["request_length"]),
            )
            body = request_record.get("body")
            request_input, _ = _request_input(body if isinstance(body, dict) else {})
            input_items += _record_indexed_items(
                connection, "request_input_items", request_id, request_input
            )

            if row["response_path"] is None:
                continue
            response_record = _read_indexed_record(
                handles,
                Path(str(row["response_path"])),
                int(row["response_offset"]),
                int(row["response_length"]),
            )
            parsed = _parse_response(response_record)
            response_items = parsed[5]
            output_items += _record_indexed_items(
                connection, "response_output_items", request_id, response_items
            )
            responses += 1
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "trees": len(selected)
        if selected
        else len({str(row["tree_id"]) for row in rows}),
        "requests": len(rows),
        "responses": responses,
        "request_input_items": input_items,
        "response_output_items": output_items,
        "tree_ids": selected,
    }
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('lineage_enrichment_summary', ?)",
        (json_dumps(summary),),
    )
    connection.commit()
    connection.close()
    return summary


def enrich_agent_messages(
    database_path: Path,
    *,
    progress_every: int = 10_000,
) -> dict[str, int | float]:
    """Add structured cross-agent messages to an index built by an older pass.

    The byte prefilter avoids decoding request lines whose history has never
    contained an ``agent_message`` item.
    """
    connection = connect(database_path)
    initialize(connection)
    row = connection.execute(
        "SELECT value FROM metadata WHERE key='attempt_dir'"
    ).fetchone()
    if row is None:
        connection.close()
        raise ValueError("database metadata has no attempt_dir")
    attempt_dir = Path(str(row[0]))
    request_files, _, _ = _discover(attempt_dir)
    connection.execute("DELETE FROM agent_messages")
    connection.commit()
    started = time.monotonic()
    bytes_seen = 0
    records_seen = 0
    matched_records = 0
    occurrences = 0
    for path in request_files:
        with path.open("rb") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                records_seen += 1
                bytes_seen += len(line)
                if b"agent_message" not in line:
                    if progress_every and records_seen % progress_every == 0:
                        _progress("agent_messages", records_seen, bytes_seen, started)
                    continue
                record = json.loads(line)
                body = record.get("body")
                request_input = body.get("input") if isinstance(body, dict) else None
                found = _record_agent_messages(
                    connection,
                    str(record.get("trace_id")),
                    record.get("recorded_at"),
                    parse_timestamp(record.get("recorded_at")),
                    request_input,
                )
                if found:
                    matched_records += 1
                    occurrences += found
                if progress_every and records_seen % progress_every == 0:
                    connection.commit()
                    _progress("agent_messages", records_seen, bytes_seen, started)
        connection.commit()
    _progress("agent_messages", records_seen, bytes_seen, started)
    unique_messages = int(
        connection.execute("SELECT COUNT(*) FROM agent_messages").fetchone()[0]
    )
    summary: dict[str, int | float] = {
        "records_seen": records_seen,
        "matched_records": matched_records,
        "message_occurrences": occurrences,
        "unique_messages": unique_messages,
        "bytes_seen": bytes_seen,
        "elapsed_s": time.monotonic() - started,
    }
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('agent_message_summary', ?)",
        (json_dumps(summary),),
    )
    connection.commit()
    connection.close()
    return summary


def _ingest_perf(
    connection: sqlite3.Connection,
    attempt_dir: Path,
    files: list[Path],
    progress_every: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    started = time.monotonic()
    bytes_seen = 0
    records_seen = 0
    for path in files:
        source_id = _register_source(connection, path, attempt_dir, "perf")
        for offset, length, record in iter_jsonl_binary(path):
            records_seen += 1
            bytes_seen += length
            request_id = _as_int(record.get("disagg_request_id"))
            if request_id is None:
                counts["missing_disagg_request_id"] += 1
                continue
            arrival = record.get("disagg_server_arrival_time")
            first = record.get("disagg_server_first_token_time")
            ttft = (
                float(first) - float(arrival)
                if isinstance(first, (int, float)) and isinstance(arrival, (int, float))
                else None
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO perf_metrics(
                  disagg_request_id, status, arrival_time, first_token_time,
                  ttft_s, source_id, line_offset, line_length
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    record.get("status"),
                    arrival,
                    first,
                    ttft,
                    source_id,
                    offset,
                    length,
                ),
            )
            counts[str(record.get("status") or "unknown")] += 1
            if progress_every and records_seen % progress_every == 0:
                connection.commit()
                _progress("perf", records_seen, bytes_seen, started)
        connection.commit()
    if files:
        _progress("perf", records_seen, bytes_seen, started)
    return counts


def _validate_eligibility(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        UPDATE requests
        SET replay_eligible=0, eligibility_reason='missing_response'
        WHERE replay_eligible=1
          AND NOT EXISTS (SELECT 1 FROM responses s WHERE s.trace_id=requests.trace_id)
        """
    )
    connection.execute(
        """
        UPDATE requests
        SET replay_eligible=0,
            eligibility_reason='response_status:' || COALESCE(
              (SELECT response_status FROM responses s WHERE s.trace_id=requests.trace_id),
              'unknown')
        WHERE replay_eligible=1
          AND COALESCE((SELECT response_status FROM responses s WHERE s.trace_id=requests.trace_id), '')
              NOT IN ('completed', 'success')
        """
    )
    connection.execute(
        """
        UPDATE requests
        SET replay_eligible=0, eligibility_reason='missing_recorded_at'
        WHERE replay_eligible=1 AND send_ts IS NULL
        """
    )
    connection.execute(
        """
        UPDATE requests
        SET replay_eligible=0, eligibility_reason='missing_input_tokens'
        WHERE replay_eligible=1
          AND COALESCE((SELECT input_tokens FROM responses s WHERE s.trace_id=requests.trace_id), 0) <= 0
        """
    )
    connection.execute(
        """
        UPDATE requests
        SET replay_eligible=0, eligibility_reason='missing_output_tokens'
        WHERE replay_eligible=1
          AND COALESCE((SELECT output_tokens FROM responses s WHERE s.trace_id=requests.trace_id), 0) <= 0
        """
    )
    connection.execute(
        """
        UPDATE requests
        SET eligibility_reason='eligible'
        WHERE replay_eligible=1
        """
    )


def build_index(
    attempt_dir: Path,
    database_path: Path,
    *,
    include_perf: bool = True,
    progress_every: int = 1000,
    rebuild: bool = False,
) -> dict[str, Any]:
    attempt_dir = attempt_dir.absolute()
    request_files, response_files, perf_files = _discover(attempt_dir)
    if database_path.exists():
        if not rebuild:
            raise FileExistsError(
                f"output database already exists: {database_path}; pass --rebuild to replace it"
            )
        database_path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(database_path) + suffix)
            if sidecar.exists():
                sidecar.unlink()

    connection = connect(database_path)
    initialize(connection)
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('attempt_dir', ?)",
        (str(attempt_dir),),
    )
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('build_pid', ?)",
        (str(os.getpid()),),
    )
    connection.commit()

    response_counts = _ingest_responses(
        connection, attempt_dir, response_files, progress_every
    )
    request_counts = _ingest_requests(connection, attempt_dir, request_files, progress_every)
    perf_counts: Counter[str] = Counter()
    if include_perf:
        perf_counts = _ingest_perf(connection, attempt_dir, perf_files, progress_every)
    _validate_eligibility(connection)
    connection.commit()

    summary = {
        "attempt_dir": str(attempt_dir),
        "database": str(database_path.absolute()),
        "request_files": len(request_files),
        "response_files": len(response_files),
        "perf_files": len(perf_files) if include_perf else 0,
        "request_counts": dict(request_counts),
        "response_counts": dict(response_counts),
        "perf_counts": dict(perf_counts),
    }
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('ingest_summary', ?)",
        (json_dumps(summary),),
    )
    connection.commit()
    connection.close()
    return summary
