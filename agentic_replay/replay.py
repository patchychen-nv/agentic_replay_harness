from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from .database import connect
from .util import json_dumps


@dataclass(frozen=True)
class ReplayDependency:
    source_request_id: str
    kind: str
    delay_s: float


@dataclass
class ReplayNode:
    request_id: str
    tree_id: str
    route: str
    session_id: str | None
    agent_name: str | None
    send_ts: float
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    source_path: Path
    source_size_bytes: int
    source_mtime_ns: int
    line_offset: int
    line_length: int
    dependencies: list[ReplayDependency] = field(default_factory=list)
    chain_parent_request_id: str | None = None
    chain_relation_mode: str | None = None
    chain_confidence: str | None = None
    chain_delta_start: int | None = None
    chain_response_item_count: int | None = None
    chain_tool_result_items: int | None = None
    chain_delay_s: float = 0.0


@dataclass
class ReplayOptions:
    base_url: str
    api_key: str | None = None
    x_api_key: str | None = None
    model: str | None = None
    endpoint: str | None = None
    schedule: str = "causal"
    root_schedule: str = "simultaneous"
    speedup: float = 1.0
    concurrency: int = 128
    timeout_s: float = 3600.0
    fixed_output_length: bool = False
    fallback_output_tokens: int | None = None
    max_seq_len: int | None = None
    max_input_tokens: int | None = None
    source_root: Path | None = None
    allow_source_changes: bool = False
    allow_partial: bool = False
    payload_mode: str = "recorded"
    chain_mismatch_policy: str = "error"
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.concurrency < 0:
            raise ValueError("concurrency must be non-negative (0 means unlimited)")


class ChainMismatchError(ValueError):
    """A live response cannot safely replace its recorded response slice."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


class ChainBlockedError(ChainMismatchError):
    """A parent failed before it could publish terminal live-chain state."""


def _resolve_source_path(row: Any, source_root: Path | None) -> Path:
    if source_root is not None:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute():
            raise ValueError(
                f"indexed source has no relocatable relative path: {relative}"
            )
        candidate = source_root / relative
        if not candidate.exists():
            raise FileNotFoundError(
                f"indexed source is missing below --source-root: {candidate}"
            )
        return candidate
    return Path(str(row["absolute_path"]))


def load_replay_nodes(
    database_path: Path,
    *,
    tree_ids: list[str] | None = None,
    limit_trees: int | None = None,
    options: ReplayOptions,
) -> tuple[dict[str, ReplayNode], list[str]]:
    connection = connect(database_path, readonly=True)
    if tree_ids:
        placeholders = ",".join("?" for _ in tree_ids)
        rows = list(
            connection.execute(
                f"SELECT * FROM trees WHERE tree_id IN ({placeholders}) ORDER BY start_ts, tree_id",
                tree_ids,
            )
        )
        missing = sorted(set(tree_ids) - {str(row["tree_id"]) for row in rows})
        if missing:
            raise ValueError(f"tree IDs not found: {', '.join(missing)}")
    else:
        if options.allow_partial and options.fallback_output_tokens is not None:
            predicate = """
              EXISTS (
                SELECT 1 FROM requests q
                WHERE q.tree_id=trees.tree_id
                  AND (q.replay_eligible=1 OR q.eligibility_reason='missing_output_tokens')
              )
            """
        else:
            predicate = (
                "eligible_request_count > 0" if options.allow_partial else "replayable=1"
            )
        query = f"SELECT * FROM trees WHERE {predicate} ORDER BY start_ts, tree_id"
        params: tuple[Any, ...] = ()
        if limit_trees is not None:
            query += " LIMIT ?"
            params = (limit_trees,)
        rows = list(connection.execute(query, params))
    if limit_trees is not None and tree_ids:
        rows = rows[:limit_trees]
    selected_tree_ids = [str(row["tree_id"]) for row in rows]
    if not selected_tree_ids:
        connection.close()
        raise ValueError("no trees selected for replay")
    if not options.allow_partial:
        rejected = [str(row["tree_id"]) for row in rows if not bool(row["replayable"])]
        if rejected:
            connection.close()
            raise ValueError(
                "selected trees are not fully replayable; pass --allow-partial to use their "
                f"eligible subset: {', '.join(rejected[:10])}"
            )

    placeholders = ",".join("?" for _ in selected_tree_ids)
    request_predicate = "q.replay_eligible=1"
    if options.allow_partial and options.fallback_output_tokens is not None:
        request_predicate = (
            "(q.replay_eligible=1 OR q.eligibility_reason='missing_output_tokens')"
        )
    request_query = f"""
      SELECT q.trace_id, q.tree_id, q.route, q.session_id, q.agent_name,
             q.send_ts, q.line_offset, q.line_length,
             s.input_tokens, s.output_tokens, s.cached_tokens,
             src.relative_path, src.absolute_path, src.size_bytes, src.mtime_ns
      FROM requests q
      JOIN sources src ON src.source_id=q.source_id
      LEFT JOIN responses s ON s.trace_id=q.trace_id
      WHERE q.tree_id IN ({placeholders}) AND {request_predicate}
      ORDER BY q.send_ts, q.trace_id
    """
    nodes: dict[str, ReplayNode] = {}
    for row in connection.execute(request_query, selected_tree_ids):
        source_path = _resolve_source_path(row, options.source_root)
        raw_output_tokens = (
            int(row["output_tokens"]) if row["output_tokens"] is not None else None
        )
        node = ReplayNode(
            request_id=str(row["trace_id"]),
            tree_id=str(row["tree_id"]),
            route=str(row["route"]),
            session_id=row["session_id"],
            agent_name=row["agent_name"],
            send_ts=float(row["send_ts"]),
            input_tokens=int(row["input_tokens"]) if row["input_tokens"] is not None else None,
            output_tokens=(
                raw_output_tokens
                if raw_output_tokens is not None and raw_output_tokens > 0
                else options.fallback_output_tokens
            ),
            cached_tokens=(
                int(row["cached_tokens"])
                if row["cached_tokens"] is not None
                else None
            ),
            source_path=source_path,
            source_size_bytes=int(row["size_bytes"]),
            source_mtime_ns=int(row["mtime_ns"]),
            line_offset=int(row["line_offset"]),
            line_length=int(row["line_length"]),
        )
        nodes[node.request_id] = node

    edge_query = f"""
      SELECT e.* FROM dependency_edges e
      JOIN requests q ON q.trace_id=e.target_request_id
      WHERE q.tree_id IN ({placeholders})
      ORDER BY e.target_request_id, e.edge_kind, e.source_request_id
    """
    dropped_dependencies: list[tuple[str, str]] = []
    for row in connection.execute(edge_query, selected_tree_ids):
        target = str(row["target_request_id"])
        source = str(row["source_request_id"])
        if target not in nodes:
            continue
        if source not in nodes:
            dropped_dependencies.append((source, target))
            continue
        nodes[target].dependencies.append(
            ReplayDependency(
                source_request_id=source,
                kind=str(row["edge_kind"]),
                delay_s=max(float(row["delay_s"] or 0.0), 0.0),
            )
        )

    if options.payload_mode == "live-chain":
        has_prompt_chains = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prompt_chains'"
        ).fetchone()
        if has_prompt_chains is None:
            connection.close()
            raise ValueError(
                "--payload-mode=live-chain requires a prompt_chains index; "
                "rebuild the index or run `agentic-replay infer` first"
            )
        chain_query = f"""
          SELECT pc.target_request_id, pc.source_request_id,
                 pc.relation_mode, pc.confidence, pc.delta_start,
                 pc.response_item_count, pc.tool_result_items, pc.delay_s
          FROM prompt_chains pc
          JOIN requests q ON q.trace_id=pc.target_request_id
          WHERE q.tree_id IN ({placeholders})
          ORDER BY pc.target_request_id
        """
        missing_chain_parents: list[tuple[str, str]] = []
        for row in connection.execute(chain_query, selected_tree_ids):
            target = str(row["target_request_id"])
            source = str(row["source_request_id"])
            if target not in nodes:
                continue
            if source not in nodes:
                missing_chain_parents.append((source, target))
                continue
            node = nodes[target]
            node.chain_parent_request_id = source
            node.chain_relation_mode = str(row["relation_mode"])
            node.chain_confidence = str(row["confidence"])
            node.chain_delta_start = int(row["delta_start"])
            node.chain_response_item_count = int(row["response_item_count"])
            node.chain_tool_result_items = int(row["tool_result_items"])
            node.chain_delay_s = max(float(row["delay_s"] or 0.0), 0.0)
        if missing_chain_parents:
            connection.close()
            example = ", ".join(
                f"{source}->{target}"
                for source, target in missing_chain_parents[:5]
            )
            raise ValueError(
                "live-chain selection excludes prompt-chain parents: " + example
            )
    connection.close()
    if dropped_dependencies and not options.allow_partial:
        example = ", ".join(f"{s}->{t}" for s, t in dropped_dependencies[:5])
        raise ValueError(f"selected graph has dependencies on excluded nodes: {example}")
    _validate_nodes(nodes, options)
    _topological_order(nodes)
    return nodes, selected_tree_ids


def _validate_nodes(nodes: dict[str, ReplayNode], options: ReplayOptions) -> None:
    if options.payload_mode not in {"recorded", "live-chain"}:
        raise ValueError(f"unsupported payload mode: {options.payload_mode}")
    if options.chain_mismatch_policy not in {"error", "recorded"}:
        raise ValueError(
            f"unsupported chain mismatch policy: {options.chain_mismatch_policy}"
        )
    if options.payload_mode == "live-chain":
        effective_routes = {
            options.endpoint or node.route for node in nodes.values()
        }
        unsupported = sorted(route for route in effective_routes if route != "/v1/responses")
        if unsupported:
            raise ValueError(
                "live-chain currently requires /v1/responses; unsupported routes: "
                + ", ".join(unsupported)
            )
    source_stats: dict[Path, tuple[int, int]] = {}
    for node in nodes.values():
        if node.output_tokens is None or node.output_tokens <= 0:
            raise ValueError(f"request {node.request_id} has no positive output-token budget")
        if options.max_input_tokens is not None and node.input_tokens is not None:
            if node.input_tokens > options.max_input_tokens:
                raise ValueError(
                    f"request {node.request_id} input_tokens={node.input_tokens} exceeds "
                    f"--max-input-tokens={options.max_input_tokens}"
                )
        if options.max_seq_len is not None and node.input_tokens is not None:
            if node.input_tokens + node.output_tokens > options.max_seq_len:
                raise ValueError(
                    f"request {node.request_id} input+output="
                    f"{node.input_tokens + node.output_tokens} exceeds "
                    f"--max-seq-len={options.max_seq_len}"
                )
        if node.source_path not in source_stats:
            stat = node.source_path.stat()
            source_stats[node.source_path] = (stat.st_size, stat.st_mtime_ns)
        actual_size, actual_mtime = source_stats[node.source_path]
        if not options.allow_source_changes and (
            actual_size != node.source_size_bytes or actual_mtime != node.source_mtime_ns
        ):
            raise ValueError(
                f"source changed after indexing: {node.source_path}; "
                "rebuild the index or pass --allow-source-changes"
            )


def _topological_order(nodes: dict[str, ReplayNode]) -> list[str]:
    indegree = {request_id: 0 for request_id in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for node in nodes.values():
        seen_sources: set[str] = set()
        for dependency in node.dependencies:
            indegree[node.request_id] += 1
            children[dependency.source_request_id].append(node.request_id)
            seen_sources.add(dependency.source_request_id)
        if (
            node.chain_parent_request_id is not None
            and node.chain_parent_request_id not in seen_sources
        ):
            indegree[node.request_id] += 1
            children[node.chain_parent_request_id].append(node.request_id)
    queue = deque(sorted(request_id for request_id, degree in indegree.items() if degree == 0))
    order: list[str] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for child in children.get(current, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(nodes):
        cyclic = sorted(request_id for request_id, degree in indegree.items() if degree > 0)
        raise ValueError(f"dependency graph contains a cycle involving: {', '.join(cyclic[:10])}")
    return order


def replay_plan(nodes: dict[str, ReplayNode], tree_ids: list[str]) -> dict[str, Any]:
    edge_counts: dict[str, int] = defaultdict(int)
    for node in nodes.values():
        for dependency in node.dependencies:
            edge_counts[dependency.kind] += 1
    send_times = [node.send_ts for node in nodes.values()]
    return {
        "trees": len(tree_ids),
        "tree_ids": tree_ids,
        "requests": len(nodes),
        "edges": dict(sorted(edge_counts.items())),
        "prompt_chains": sum(
            node.chain_parent_request_id is not None for node in nodes.values()
        ),
        "captured_span_s": max(send_times) - min(send_times) if send_times else 0.0,
        "input_tokens": sum(node.input_tokens or 0 for node in nodes.values()),
        "output_token_budget": sum(node.output_tokens or 0 for node in nodes.values()),
        "max_input_tokens": max((node.input_tokens or 0 for node in nodes.values()), default=0),
        "max_sequence_tokens": max(
            ((node.input_tokens or 0) + (node.output_tokens or 0) for node in nodes.values()),
            default=0,
        ),
    }


def _read_payload(node: ReplayNode) -> dict[str, Any]:
    with node.source_path.open("rb") as handle:
        handle.seek(node.line_offset)
        raw = handle.read(node.line_length)
    record = json.loads(raw)
    if str(record.get("trace_id")) != node.request_id:
        raise ValueError(
            f"payload locator mismatch for {node.request_id} at {node.source_path}:"
            f"{node.line_offset}"
        )
    body = record.get("body")
    if not isinstance(body, dict):
        raise ValueError(f"request {node.request_id} has no JSON object body")
    return body


def _prepare_payload(
    body: dict[str, Any], node: ReplayNode, options: ReplayOptions
) -> dict[str, Any]:
    if options.model:
        body["model"] = options.model
    if options.fixed_output_length:
        effective_route = options.endpoint or node.route
        if effective_route == "/v1/responses":
            body["max_output_tokens"] = node.output_tokens
            body.pop("max_tokens", None)
        else:
            body["max_tokens"] = node.output_tokens
            body.pop("max_output_tokens", None)
        body["ignore_eos"] = True
    return body


def _encode_payload(body: dict[str, Any]) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_OUTPUT_ONLY_ITEM_KEYS = {
    "annotations",
    "async_",
    "caller",
    "logprobs",
    "phase",
    "status",
}


def _strip_output_only_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_output_only_fields(child)
            for key, child in value.items()
            if key not in _OUTPUT_ONLY_ITEM_KEYS
            and not (key == "encrypted_content" and child is None)
        }
    if isinstance(value, list):
        return [_strip_output_only_fields(child) for child in value]
    return copy.deepcopy(value)


def _project_live_item(live: dict[str, Any], recorded: dict[str, Any]) -> dict[str, Any]:
    """Convert an output event item into the shape accepted in Responses input.

    The recorded request is the strongest available description of the client-side
    projection.  In particular, TensorRT-LLM response events contain fields such as
    ``status``, ``phase`` and ``logprobs`` that the captured client removed before
    sending the next request.  Values always come from the live item; the recorded
    item supplies only the allowed shape.
    """

    def project(live_value: Any, template: Any) -> Any:
        if isinstance(live_value, dict) and isinstance(template, dict):
            return {
                key: project(child, template[key])
                for key, child in live_value.items()
                if key in template and key not in _OUTPUT_ONLY_ITEM_KEYS
                and not (key == "encrypted_content" and child is None)
            }
        if isinstance(live_value, list) and isinstance(template, list):
            if not template:
                return _strip_output_only_fields(live_value)
            projected: list[Any] = []
            for index, child in enumerate(live_value):
                exemplar: Any = template[min(index, len(template) - 1)]
                if isinstance(child, dict):
                    child_type = child.get("type")
                    typed = next(
                        (
                            item
                            for item in template
                            if isinstance(item, dict)
                            and item.get("type") == child_type
                        ),
                        None,
                    )
                    if typed is not None:
                        exemplar = typed
                projected.append(project(child, exemplar))
            return projected
        return copy.deepcopy(live_value)

    projected = project(live, recorded)
    if not isinstance(projected, dict):  # defensive; both inputs are dicts
        raise TypeError("projected response item is not an object")
    if _is_tool_call_type(live.get("type")):
        namespace = live.get("namespace")
        if isinstance(namespace, str) and namespace:
            # Current TRT-LLM Responses input understands namespaced calls and
            # needs this field to reconstruct the qualified tool name.
            projected["namespace"] = namespace
    return projected


def _is_tool_call_type(item_type: Any) -> bool:
    return (
        isinstance(item_type, str)
        and item_type.endswith("_call")
        and not item_type.endswith("_call_output")
    )


def _is_tool_result_type(item_type: Any) -> bool:
    return isinstance(item_type, str) and item_type.endswith("_call_output")


def _rewrite_live_chain_payload(
    body: dict[str, Any],
    node: ReplayNode,
    parent_sent_input: list[Any],
    parent_live_items: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a cumulative live prompt for one append-only Responses transition."""

    reasons: list[str] = []
    if node.chain_relation_mode != "append":
        reasons.append(
            f"unsupported relation_mode={node.chain_relation_mode!r}; only append is replayable"
        )
    recorded_input = body.get("input")
    if not isinstance(recorded_input, list):
        reasons.append("target Responses payload input is not an item list")
        recorded_input = []
    delta_start = node.chain_delta_start
    response_count = node.chain_response_item_count
    if delta_start is None or response_count is None:
        reasons.append("prompt-chain slice metadata is missing")
        delta_start = 0
        response_count = 0
    if delta_start < 0 or response_count < 0:
        reasons.append("prompt-chain slice metadata is negative")
    response_end = delta_start + response_count
    if response_end > len(recorded_input):
        reasons.append(
            "recorded response slice exceeds target input: "
            f"end={response_end}, items={len(recorded_input)}"
        )

    recorded_response_items = recorded_input[delta_start:response_end]
    projected_live_items: list[dict[str, Any]] = []
    recorded_templates_by_type: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for ordinal, item in enumerate(recorded_response_items):
        if not isinstance(item, dict):
            reasons.append(f"recorded response item {ordinal} is not an object")
            continue
        item_type = item.get("type")
        if isinstance(item_type, str):
            recorded_templates_by_type[item_type].append(item)
    template_offsets: defaultdict[str, int] = defaultdict(int)
    for live in parent_live_items:
        live_type = live.get("type")
        templates = (
            recorded_templates_by_type.get(live_type, [])
            if isinstance(live_type, str)
            else []
        )
        if templates:
            offset = min(template_offsets[live_type], len(templates) - 1)
            projected_live_items.append(_project_live_item(live, templates[offset]))
            template_offsets[live_type] += 1
        else:
            projected = _strip_output_only_fields(live)
            if not isinstance(projected, dict):  # defensive
                reasons.append("live response item did not normalize to an object")
                continue
            projected_live_items.append(projected)

    recorded_tool_calls = [
        item
        for item in recorded_response_items
        if isinstance(item, dict) and _is_tool_call_type(item.get("type"))
    ]
    live_tool_calls = [
        item
        for item in parent_live_items
        if _is_tool_call_type(item.get("type"))
    ]
    if len(live_tool_calls) != len(recorded_tool_calls):
        reasons.append(
            "live/recorded tool-call count differs: "
            f"live={len(live_tool_calls)}, recorded={len(recorded_tool_calls)}"
        )

    call_id_map: dict[str, str] = {}
    tool_argument_mismatches = 0
    for ordinal, (recorded, live) in enumerate(
        zip(recorded_tool_calls, live_tool_calls)
    ):
        live_type = live.get("type")
        recorded_type = recorded.get("type")
        if live_type != recorded_type:
            reasons.append(
                f"tool call {ordinal} type differs: "
                f"live={live_type!r}, recorded={recorded_type!r}"
            )
            continue
        recorded_name = recorded.get("name")
        live_name = live.get("name")
        if recorded_name != live_name:
            reasons.append(
                f"tool call {ordinal} name differs: "
                f"live={live_name!r}, recorded={recorded_name!r}"
            )
        recorded_namespace = recorded.get("namespace")
        live_namespace = live.get("namespace")
        if (
            isinstance(recorded_namespace, str)
            and recorded_namespace
            and recorded_namespace != live_namespace
        ):
            reasons.append(
                f"tool call {ordinal} namespace differs: "
                f"live={live_namespace!r}, recorded={recorded_namespace!r}"
            )
        recorded_call_id = recorded.get("call_id")
        live_call_id = live.get("call_id")
        if not isinstance(recorded_call_id, str) or not recorded_call_id:
            reasons.append(f"recorded tool call {ordinal} has no call_id")
        elif not isinstance(live_call_id, str) or not live_call_id:
            reasons.append(f"live tool call {ordinal} has no call_id")
        else:
            call_id_map[recorded_call_id] = live_call_id
        argument_key = "arguments" if "arguments" in recorded else "input"
        if recorded.get(argument_key) != live.get(argument_key):
            tool_argument_mismatches += 1

    external_suffix = copy.deepcopy(recorded_input[response_end:])
    rebound = 0
    tool_results_seen = 0
    for ordinal, item in enumerate(external_suffix):
        if not isinstance(item, dict) or not _is_tool_result_type(item.get("type")):
            continue
        tool_results_seen += 1
        recorded_call_id = item.get("call_id")
        if not isinstance(recorded_call_id, str) or recorded_call_id not in call_id_map:
            reasons.append(
                f"tool result suffix item {ordinal} cannot bind call_id={recorded_call_id!r}"
            )
            continue
        item["call_id"] = call_id_map[recorded_call_id]
        rebound += 1

    expected_tool_results = node.chain_tool_result_items
    if expected_tool_results is not None and tool_results_seen != expected_tool_results:
        reasons.append(
            "tool-result count differs from lineage metadata: "
            f"payload={tool_results_seen}, indexed={expected_tool_results}"
        )
    if reasons:
        raise ChainMismatchError(reasons)

    live_input = (
        copy.deepcopy(parent_sent_input)
        + projected_live_items
        + external_suffix
    )
    body["input"] = live_input
    metrics = {
        "chain_status": "applied",
        "chain_parent_request_id": node.chain_parent_request_id,
        "chain_relation_mode": node.chain_relation_mode,
        "chain_confidence": node.chain_confidence,
        "chain_recorded_response_items": len(recorded_response_items),
        "chain_live_response_items": len(projected_live_items),
        "chain_live_response_types": [item.get("type") for item in projected_live_items],
        "chain_tool_results_rebound": rebound,
        "chain_tool_argument_mismatches": tool_argument_mismatches,
        "chain_recorded_input_items": len(recorded_input),
        "chain_live_input_items": len(live_input),
        "chain_live_input_sha256": _json_sha256(live_input),
    }
    return body, metrics


def _extract_usage_from_event(event: dict[str, Any]) -> int | None:
    usage = _usage_object_from_event(event)
    if usage is None:
        return None
    value = usage.get("output_tokens", usage.get("completion_tokens"))
    return int(value) if isinstance(value, (int, float)) else None


def _usage_object_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    usage = event.get("usage")
    if not isinstance(usage, dict) and isinstance(event.get("response"), dict):
        usage = event["response"].get("usage")
    return usage if isinstance(usage, dict) else None


def _number_from_usage(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


class _ResponseParser:
    def __init__(self, route: str, *, capture_output_items: bool = True) -> None:
        self.route = route
        self.capture_output_items = capture_output_items
        self.buffer = b""
        self.mode: str | None = None
        self.output_tokens: int | None = None
        self.input_tokens: int | None = None
        self.cached_tokens: int | None = None
        self.terminal_event: str | None = None
        self.response_status: str | None = None
        self.error_detail: Any = None
        self.saw_done = False
        self.saw_chat_finish = False
        self.output_item_count = 0
        self.output_items: list[dict[str, Any]] = []
        self.output_item_indices: list[int | None] = []
        self.output_capture_errors: list[str] = []

    def feed(self, chunk: bytes) -> None:
        self.buffer += chunk
        if self.mode is None:
            stripped = self.buffer.lstrip()
            if stripped.startswith((b"{", b"[")):
                self.mode = "json"
            elif b"data:" in self.buffer or b"event:" in self.buffer:
                self.mode = "sse"
        if self.mode == "sse":
            while b"\n\n" in self.buffer:
                block, self.buffer = self.buffer.split(b"\n\n", 1)
                self._parse_block(block)
        if len(self.buffer) > 32 * 1024 * 1024:
            # Do not let a malformed endpoint response grow without bound.
            self.error_detail = "response parser buffer exceeded 32 MiB"
            self.buffer = self.buffer[-1024 * 1024 :]
            self.mode = "invalid"

    def finish(self) -> None:
        if self.mode == "json":
            try:
                value = json.loads(self.buffer)
            except json.JSONDecodeError as error:
                self.error_detail = f"invalid JSON response: {error}"
            else:
                if isinstance(value, dict):
                    self._parse_event(value, raw_json=True)
                else:
                    self.error_detail = "JSON response is not an object"
        elif self.buffer and self.mode != "invalid":
            self._parse_block(self.buffer)
        self.buffer = b""
        indexed = [index for index in self.output_item_indices if index is not None]
        if indexed and indexed != list(range(len(indexed))):
            self.output_capture_errors.append(
                f"non-contiguous response.output_item.done indices: {indexed}"
            )

    def _parse_block(self, block: bytes) -> None:
        for line in block.splitlines():
            if not line.startswith(b"data:"):
                continue
            data = line[5:].lstrip()
            if data == b"[DONE]":
                self.saw_done = True
                continue
            if not data:
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                self._parse_event(event)

    def _parse_event(self, event: dict[str, Any], *, raw_json: bool = False) -> None:
        usage = _usage_object_from_event(event)
        if usage is not None:
            parsed_output = _number_from_usage(
                usage, "output_tokens", "completion_tokens"
            )
            if parsed_output is not None:
                self.output_tokens = parsed_output
            parsed_input = _number_from_usage(usage, "input_tokens", "prompt_tokens")
            if parsed_input is not None:
                self.input_tokens = parsed_input
            details = usage.get("input_tokens_details")
            if not isinstance(details, dict):
                details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                parsed_cached = _number_from_usage(details, "cached_tokens")
                if parsed_cached is not None:
                    self.cached_tokens = parsed_cached

        event_type = event.get("type")
        response = event.get("response")
        if event_type == "response.output_item.done":
            item = event.get("item")
            if not isinstance(item, dict):
                self.output_capture_errors.append(
                    "response.output_item.done has no object item"
                )
            else:
                output_index = event.get("output_index")
                if not isinstance(output_index, int):
                    output_index = None
                if output_index is not None and output_index in self.output_item_indices:
                    self.output_capture_errors.append(
                        f"duplicate response.output_item.done index: {output_index}"
                    )
                else:
                    self.output_item_count += 1
                    if self.capture_output_items:
                        self.output_items.append(copy.deepcopy(item))
                    self.output_item_indices.append(output_index)
        if isinstance(event_type, str) and event_type in {
            "response.completed",
            "response.failed",
            "response.incomplete",
        }:
            self.terminal_event = event_type
            if isinstance(response, dict):
                status = response.get("status")
                self.response_status = str(status) if status is not None else None
                self.error_detail = response.get("error") or response.get(
                    "incomplete_details"
                )
            if self.response_status is None:
                self.response_status = event_type.removeprefix("response.")
            return
        if event_type == "error" or (
            event.get("error") is not None and not isinstance(response, dict)
        ):
            self.terminal_event = "error"
            self.response_status = "failed"
            self.error_detail = event.get("error")
            return

        if raw_json:
            if self.route == "/v1/responses":
                output = event.get("output")
                if isinstance(output, list):
                    object_items = [item for item in output if isinstance(item, dict)]
                    self.output_item_count = len(object_items)
                    if self.capture_output_items:
                        self.output_items = [copy.deepcopy(item) for item in object_items]
                    self.output_item_indices = list(range(len(object_items)))
            status = event.get("status")
            if status is not None:
                self.response_status = str(status)
                self.terminal_event = f"response.{status}"
                self.error_detail = event.get("error") or event.get(
                    "incomplete_details"
                )
            choices = event.get("choices")
            if isinstance(choices, list):
                self.terminal_event = self.terminal_event or "chat.completion"
                self.response_status = self.response_status or "completed"
                self.saw_chat_finish = any(
                    isinstance(choice, dict) and choice.get("finish_reason") is not None
                    for choice in choices
                )
        else:
            choices = event.get("choices")
            if isinstance(choices, list) and any(
                isinstance(choice, dict) and choice.get("finish_reason") is not None
                for choice in choices
            ):
                self.saw_chat_finish = True

    def result_status(self, http_status: int) -> str:
        if not 200 <= http_status < 300:
            return "http_error"
        normalized = (self.response_status or "").lower()
        if normalized == "failed" or self.terminal_event in {
            "response.failed",
            "error",
        }:
            return "response_failed"
        if normalized == "incomplete" or self.terminal_event == "response.incomplete":
            return "response_incomplete"
        if self.route == "/v1/responses":
            if self.terminal_event == "response.completed" and normalized in {
                "",
                "completed",
            }:
                return "completed"
            return "missing_terminal_event"
        if self.saw_done or self.saw_chat_finish or self.terminal_event == "chat.completion":
            return "completed"
        return "missing_terminal_event"


class _ResultWriter:
    def __init__(self, handle: TextIO) -> None:
        self.handle = handle
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.task = asyncio.create_task(self._run())

    async def write(self, value: dict[str, Any]) -> None:
        self.queue.put_nowait(json_dumps(value) + "\n")

    async def close(self) -> None:
        self.queue.put_nowait(None)
        await self.task

    def _write_batch(self, lines: list[str]) -> None:
        self.handle.write("".join(lines))
        self.handle.flush()

    async def _run(self) -> None:
        batch: list[str] = []
        while True:
            line = await self.queue.get()
            if line is None:
                if batch:
                    await asyncio.to_thread(self._write_batch, batch)
                return
            batch.append(line)
            # Flush promptly when the producer has caught up.  Small or
            # dependency-serial replays may run for minutes with far fewer
            # than 128 results; retaining that whole run in memory leaves an
            # empty artifact if the process is interrupted or times out.
            if len(batch) >= 128 or self.queue.empty():
                await asyncio.to_thread(self._write_batch, batch)
                batch = []


@dataclass
class _PreparedPayload:
    data: bytes
    sent_input: list[Any] | None
    chain_metrics: dict[str, Any]


def _tool_result_item_count(input_items: Any) -> int | None:
    if not isinstance(input_items, list):
        return None
    return sum(
        1
        for item in input_items
        if isinstance(item, dict)
        and isinstance(item.get("type"), str)
        and str(item["type"]).endswith("_call_output")
    )


async def run_replay(
    nodes: dict[str, ReplayNode],
    tree_ids: list[str],
    options: ReplayOptions,
    output_path: Path,
) -> dict[str, Any]:
    if options.dry_run:
        return {"dry_run": True, **replay_plan(nodes, tree_ids)}
    try:
        import aiohttp
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError("replay requires aiohttp; install the project dependencies") from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    for candidate in (output_path, summary_path):
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite replay output: {candidate}; choose a new --output path"
            )
    run_id = str(uuid.uuid4())
    root_nodes = [
        node
        for node in nodes.values()
        if not node.dependencies and node.chain_parent_request_id is None
    ]
    sent_inputs: dict[str, list[Any]] = {}
    live_output_items: dict[str, list[dict[str, Any]]] = {}

    async def prepare_payload(node: ReplayNode) -> _PreparedPayload:
        body = await asyncio.to_thread(_read_payload, node)
        recorded_input = body.get("input")
        recorded_input_sha256 = (
            _json_sha256(recorded_input) if "input" in body else None
        )
        payload_metrics: dict[str, Any] = {
            "prompt_source": (
                "recorded" if options.payload_mode == "recorded" else "live_chain"
            ),
            "live_output_policy": (
                "metrics_only"
                if options.payload_mode == "recorded"
                else "next_prompt"
            ),
            "recorded_input_items": (
                len(recorded_input) if isinstance(recorded_input, list) else None
            ),
            "recorded_tool_result_items": _tool_result_item_count(recorded_input),
            "recorded_input_sha256": recorded_input_sha256,
        }
        chain_metrics: dict[str, Any] = {}
        if options.payload_mode == "live-chain":
            parent_id = node.chain_parent_request_id
            if parent_id is None:
                input_items = body.get("input")
                chain_metrics = {
                    "chain_status": "root_recorded",
                    "chain_live_input_items": len(input_items)
                    if isinstance(input_items, list)
                    else None,
                    "chain_live_input_sha256": _json_sha256(input_items)
                    if isinstance(input_items, list)
                    else None,
                }
            else:
                parent_input = sent_inputs.get(parent_id)
                parent_output = live_output_items.get(parent_id)
                missing: list[str] = []
                if parent_input is None:
                    missing.append(f"parent {parent_id} has no committed sent input")
                if parent_output is None:
                    missing.append(f"parent {parent_id} has no committed live output items")
                try:
                    if missing:
                        raise ChainBlockedError(missing)
                    body, chain_metrics = _rewrite_live_chain_payload(
                        body,
                        node,
                        parent_input or [],
                        parent_output or [],
                    )
                except ChainBlockedError as error:
                    if options.chain_mismatch_policy != "recorded":
                        raise
                    recorded_input = body.get("input")
                    chain_metrics = {
                        "chain_status": "recorded_fallback",
                        "chain_parent_request_id": parent_id,
                        "chain_mismatch_reasons": error.reasons,
                        "chain_live_input_items": len(recorded_input)
                        if isinstance(recorded_input, list)
                        else None,
                        "chain_live_input_sha256": _json_sha256(recorded_input)
                        if isinstance(recorded_input, list)
                        else None,
                    }
                except ChainMismatchError as error:
                    if options.chain_mismatch_policy != "recorded":
                        raise
                    recorded_input = body.get("input")
                    chain_metrics = {
                        "chain_status": "recorded_fallback",
                        "chain_parent_request_id": parent_id,
                        "chain_mismatch_reasons": error.reasons,
                        "chain_live_input_items": len(recorded_input)
                        if isinstance(recorded_input, list)
                        else None,
                        "chain_live_input_sha256": _json_sha256(recorded_input)
                        if isinstance(recorded_input, list)
                        else None,
                    }
        body = _prepare_payload(body, node, options)
        effective_route = options.endpoint or node.route
        requested_output_token_limit = (
            body.get("max_output_tokens")
            if effective_route == "/v1/responses"
            else body.get("max_tokens")
        )
        input_items = body.get("input")
        sent_input = copy.deepcopy(input_items) if isinstance(input_items, list) else None
        sent_input_sha256 = _json_sha256(input_items) if "input" in body else None
        payload_metrics.update(
            {
                "sent_input_items": (
                    len(input_items) if isinstance(input_items, list) else None
                ),
                "sent_tool_result_items": _tool_result_item_count(input_items),
                "sent_input_sha256": sent_input_sha256,
                "requested_ignore_eos": bool(body.get("ignore_eos", False)),
                "requested_output_token_limit": (
                    requested_output_token_limit
                    if isinstance(requested_output_token_limit, int)
                    and not isinstance(requested_output_token_limit, bool)
                    else None
                ),
            }
        )
        if (
            options.payload_mode == "recorded"
            and recorded_input_sha256 != sent_input_sha256
        ):
            raise RuntimeError(
                f"recorded replay mutated input for request {node.request_id}"
            )
        payload_metrics.update(chain_metrics)
        data = await asyncio.to_thread(_encode_payload, body)
        return _PreparedPayload(
            data=data, sent_input=sent_input, chain_metrics=payload_metrics
        )

    # Dependency-free requests define the replay time origin. Prepare them
    # before starting the clock; otherwise source I/O and JSON encoding smear
    # a simultaneous root release by seconds on a large capture.
    preload_started = time.monotonic()
    root_payload_values = await asyncio.gather(
        *(prepare_payload(node) for node in root_nodes)
    )
    root_payloads: dict[str, _PreparedPayload] = {
        node.request_id: payload
        for node, payload in zip(root_nodes, root_payload_values, strict=True)
    }
    root_payload_count = len(root_payloads)
    root_preload_s = time.monotonic() - preload_started
    root_payload_bytes = sum(len(payload.data) for payload in root_payload_values)
    min_send_ts = min(node.send_ts for node in nodes.values())
    tree_start: dict[str, float] = {}
    for node in nodes.values():
        tree_start[node.tree_id] = min(
            node.send_ts, tree_start.get(node.tree_id, node.send_ts)
        )
    finish_times: dict[str, float] = {}
    completion_events = {request_id: asyncio.Event() for request_id in nodes}
    start_gate = asyncio.Event()
    tasks_ready = asyncio.Event()
    ready_count = 0
    wall_started = 0.0
    loop_started = 0.0
    # A zero-valued Semaphore never releases requests; unlimited mode must
    # bypass it entirely as well as disabling the HTTP connector's limits.
    request_slots = (
        asyncio.Semaphore(options.concurrency)
        if options.concurrency > 0
        else nullcontext()
    )
    counters: defaultdict[str, int] = defaultdict(int)
    output_fidelity: defaultdict[str, int] = defaultdict(int)
    chain_fidelity: defaultdict[str, int] = defaultdict(int)
    usage_totals: defaultdict[str, int] = defaultdict(int)
    request_control_totals: defaultdict[str, int] = defaultdict(int)
    request_timeout = aiohttp.ClientTimeout(total=options.timeout_s)
    connector = aiohttp.TCPConnector(limit=options.concurrency, limit_per_host=0)

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if options.api_key:
        headers["Authorization"] = f"Bearer {options.api_key}"
    if options.x_api_key:
        headers["x-api-key"] = options.x_api_key

    with output_path.open("x", encoding="utf-8") as result_handle:
        writer = _ResultWriter(result_handle)
        async with aiohttp.ClientSession(timeout=request_timeout, connector=connector) as session:

            async def execute(node: ReplayNode) -> None:
                nonlocal ready_count
                replay_request_id = str(uuid.uuid4())
                result: dict[str, Any] = {
                    "schema": "agentic_replay_result/v1",
                    "run_id": run_id,
                    "source_request_id": node.request_id,
                    "replay_request_id": replay_request_id,
                    "tree_id": node.tree_id,
                    "agent_name": node.agent_name,
                    "route": options.endpoint or node.route,
                    "recorded_input_tokens": node.input_tokens,
                    "recorded_output_tokens": node.output_tokens,
                    "recorded_cached_tokens": node.cached_tokens,
                    "recorded_cache_hit_ratio": (
                        node.cached_tokens / node.input_tokens
                        if node.cached_tokens is not None
                        and node.input_tokens is not None
                        and node.input_tokens > 0
                        else None
                    ),
                    "requested_ignore_eos": None,
                    "requested_output_token_limit": None,
                    "status": "pending",
                }
                ready_count += 1
                if ready_count == len(nodes):
                    tasks_ready.set()
                try:
                    for dependency in node.dependencies:
                        await completion_events[dependency.source_request_id].wait()
                    dependency_sources = {
                        dependency.source_request_id for dependency in node.dependencies
                    }
                    if (
                        node.chain_parent_request_id is not None
                        and node.chain_parent_request_id not in dependency_sources
                    ):
                        await completion_events[node.chain_parent_request_id].wait()
                    # All dependency-free nodes stop here while the remaining
                    # tasks settle on dependency Events. The clock starts only
                    # after that setup, so a large DAG does not consume its
                    # first recorded offsets during task construction.
                    await start_gate.wait()

                    if options.schedule == "fixed":
                        due = loop_started + (node.send_ts - min_send_ts) / options.speedup
                    elif options.schedule == "asap":
                        due = loop_started
                    else:
                        dependency_due = [
                            finish_times[dependency.source_request_id]
                            + dependency.delay_s / options.speedup
                            for dependency in node.dependencies
                        ]
                        if (
                            node.chain_parent_request_id is not None
                            and node.chain_parent_request_id not in dependency_sources
                        ):
                            dependency_due.append(
                                finish_times[node.chain_parent_request_id]
                                + node.chain_delay_s / options.speedup
                            )
                        if dependency_due:
                            due = max(dependency_due)
                        else:
                            tree_base = loop_started
                            if options.root_schedule == "recorded":
                                tree_base += (
                                    tree_start[node.tree_id] - min_send_ts
                                ) / options.speedup
                            # A tree can have multiple dependency-free roots
                            # (e.g. an inbound agent_message interrupts an
                            # in-flight request, or a captured retry). Preserve
                            # their within-tree release offsets.
                            due = tree_base + (
                                node.send_ts - tree_start[node.tree_id]
                            ) / options.speedup

                    # Pop preloaded roots on first use so the full root corpus
                    # is not retained for the duration of a long replay.
                    prepared = root_payloads.pop(node.request_id, None)
                    if prepared is None:
                        # Decode and encode dependent requests during their
                        # recorded external gap. aiohttp's ``json=`` path would
                        # serialize synchronously on the event loop.
                        prepared = await prepare_payload(node)
                    if prepared.chain_metrics:
                        result.update(prepared.chain_metrics)
                        chain_status = prepared.chain_metrics.get("chain_status")
                        if isinstance(chain_status, str):
                            chain_fidelity[chain_status] += 1
                    if result.get("requested_ignore_eos") is True:
                        request_control_totals["requests_with_ignore_eos"] += 1
                    if isinstance(result.get("requested_output_token_limit"), int):
                        request_control_totals[
                            "requests_with_output_token_limit"
                        ] += 1
                    remaining = due - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(remaining)

                    request_headers = dict(headers)
                    request_headers["X-Request-ID"] = replay_request_id
                    request_headers["X-Source-Trace-ID"] = node.request_id
                    request_headers["X-Replay-Run-ID"] = run_id
                    if node.session_id:
                        request_headers["X-Session-ID"] = node.session_id
                        request_headers["X-Correlation-ID"] = node.session_id
                    url = options.base_url.rstrip("/") + (options.endpoint or node.route)
                    async with request_slots:
                        request_started = time.monotonic()
                        result["scheduled_offset_s"] = due - loop_started
                        result["actual_start_offset_s"] = request_started - loop_started
                        parser = _ResponseParser(
                            options.endpoint or node.route,
                            capture_output_items=options.payload_mode == "live-chain",
                        )
                        response_bytes = 0
                        first_byte_at: float | None = None
                        error_preview = bytearray()
                        async with session.post(
                            url, data=prepared.data, headers=request_headers
                        ) as response:
                            result["http_status"] = response.status
                            async for chunk in response.content.iter_chunked(64 * 1024):
                                if first_byte_at is None:
                                    first_byte_at = time.monotonic()
                                response_bytes += len(chunk)
                                parser.feed(chunk)
                                if response.status >= 400 and len(error_preview) < 4096:
                                    error_preview.extend(chunk[: 4096 - len(error_preview)])
                            parser.finish()
                            request_finished = time.monotonic()
                            result.update(
                                {
                                    "status": parser.result_status(response.status),
                                    "latency_s": request_finished - request_started,
                                    "ttfb_s": first_byte_at - request_started
                                    if first_byte_at is not None
                                    else None,
                                    "response_bytes": response_bytes,
                                    "replay_input_tokens": parser.input_tokens,
                                    "replay_output_tokens": parser.output_tokens,
                                    "replay_cached_tokens": parser.cached_tokens,
                                    "replay_cache_hit_ratio": (
                                        parser.cached_tokens / parser.input_tokens
                                        if parser.cached_tokens is not None
                                        and parser.input_tokens is not None
                                        and parser.input_tokens > 0
                                        else None
                                    ),
                                    "response_terminal_event": parser.terminal_event,
                                    "response_status": parser.response_status,
                                    "live_output_items": parser.output_item_count,
                                }
                            )
                            if options.payload_mode == "live-chain":
                                result.update(
                                    {
                                        "live_output_item_types": [
                                            item.get("type")
                                            for item in parser.output_items
                                        ],
                                        "live_output_item_ids": [
                                            item.get("id") for item in parser.output_items
                                        ],
                                        "live_output_call_ids": [
                                            item.get("call_id")
                                            for item in parser.output_items
                                            if item.get("call_id") is not None
                                        ],
                                        "live_output_sha256": _json_sha256(
                                            parser.output_items
                                        ),
                                    }
                                )
                            if (
                                options.payload_mode == "live-chain"
                                and parser.output_capture_errors
                            ):
                                result["output_item_capture_errors"] = (
                                    parser.output_capture_errors
                                )
                            if parser.error_detail is not None:
                                detail = (
                                    parser.error_detail
                                    if isinstance(parser.error_detail, str)
                                    else json_dumps(parser.error_detail)
                                )
                                result["response_error"] = detail[:4096]
                            if error_preview:
                                result["error_body_preview"] = error_preview.decode(
                                    "utf-8", errors="replace"
                                )
                        if parser.input_tokens is not None:
                            usage_totals["requests_with_input_usage"] += 1
                            usage_totals["input_tokens"] += parser.input_tokens
                        if parser.cached_tokens is not None:
                            usage_totals["requests_with_cached_usage"] += 1
                            usage_totals["cached_tokens"] += parser.cached_tokens
                            if parser.input_tokens is not None:
                                usage_totals["cache_ratio_input_tokens"] += (
                                    parser.input_tokens
                                )
                                usage_totals["cache_ratio_cached_tokens"] += (
                                    parser.cached_tokens
                                )
                        if (
                            options.payload_mode == "live-chain"
                            and result["status"] in {"completed", "response_incomplete"}
                            and not parser.output_capture_errors
                            and prepared.sent_input is not None
                        ):
                            # Commit only after the HTTP response reached a terminal
                            # event. Descendants are released by completion_events
                            # below and therefore cannot observe partial SSE state.
                            sent_inputs[node.request_id] = prepared.sent_input
                            live_output_items[node.request_id] = parser.output_items
                            result["chain_output_committed"] = True
                        elif options.payload_mode == "live-chain":
                            result["chain_output_committed"] = False
                        finish_times[node.request_id] = request_finished
                        counters[result["status"]] += 1
                except ChainBlockedError as error:
                    finish_times[node.request_id] = time.monotonic()
                    result.update(
                        {
                            "status": "chain_blocked",
                            "chain_status": "blocked",
                            "chain_parent_request_id": node.chain_parent_request_id,
                            "chain_mismatch_reasons": error.reasons,
                        }
                    )
                    counters["chain_blocked"] += 1
                    chain_fidelity["blocked"] += 1
                except ChainMismatchError as error:
                    finish_times[node.request_id] = time.monotonic()
                    result.update(
                        {
                            "status": "chain_mismatch",
                            "chain_status": "mismatch",
                            "chain_parent_request_id": node.chain_parent_request_id,
                            "chain_mismatch_reasons": error.reasons,
                        }
                    )
                    counters["chain_mismatch"] += 1
                    chain_fidelity["mismatch"] += 1
                except Exception as error:
                    finish_times[node.request_id] = time.monotonic()
                    result.update(
                        {
                            "status": "exception",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    counters["exception"] += 1
                finally:
                    # OSL comparison is an observation, independent of whether
                    # replay requested the recorded OSL as a hard generation cap.
                    replay_tokens = result.get("replay_output_tokens")
                    if not isinstance(replay_tokens, int):
                        result["output_length_match"] = None
                        fidelity_status = (
                            "not_sent"
                            if result.get("status")
                            in {"chain_mismatch", "chain_blocked"}
                            else "missing_usage"
                        )
                        output_fidelity[fidelity_status] += 1
                    else:
                        delta = replay_tokens - int(node.output_tokens or 0)
                        matched = delta == 0
                        result["output_token_delta"] = delta
                        result["output_length_match"] = matched
                        output_fidelity["matched" if matched else "mismatched"] += 1
                    result["finished_offset_s"] = finish_times[node.request_id] - loop_started
                    completion_events[node.request_id].set()
                    await writer.write(result)

            tasks = [asyncio.create_task(execute(node)) for node in nodes.values()]
            # Wait until every task has started and reached either a dependency
            # Event or start_gate before establishing the replay time origin.
            await tasks_ready.wait()
            wall_started = time.time()
            loop_started = time.monotonic()
            start_gate.set()
            await asyncio.gather(*tasks)
            await writer.close()

    elapsed = time.monotonic() - loop_started
    cache_ratio_denominator = usage_totals.get("cache_ratio_input_tokens", 0)
    cache_usage = {
        "requests_with_input_usage": usage_totals.get(
            "requests_with_input_usage", 0
        ),
        "requests_with_cached_usage": usage_totals.get(
            "requests_with_cached_usage", 0
        ),
        "replay_input_tokens": usage_totals.get("input_tokens", 0),
        "replay_cached_tokens": usage_totals.get("cached_tokens", 0),
        "replay_cache_hit_ratio": (
            usage_totals.get("cache_ratio_cached_tokens", 0)
            / cache_ratio_denominator
            if cache_ratio_denominator > 0
            else None
        ),
    }
    summary = {
        "schema": "agentic_replay_run/v1",
        "run_id": run_id,
        "started_at_unix_s": wall_started,
        "elapsed_s": elapsed,
        "trees": len(tree_ids),
        "requests": len(nodes),
        "statuses": dict(counters),
        "output_length_fidelity": dict(output_fidelity),
        "output_length_policy": (
            "recorded_limit_ignore_eos"
            if options.fixed_output_length
            else "source_payload"
        ),
        "fixed_output_length_requested": options.fixed_output_length,
        "ignore_eos_requested": (
            request_control_totals.get("requests_with_ignore_eos", 0) > 0
        ),
        "requests_with_ignore_eos": request_control_totals.get(
            "requests_with_ignore_eos", 0
        ),
        "requests_with_output_token_limit": request_control_totals.get(
            "requests_with_output_token_limit", 0
        ),
        "schedule": options.schedule,
        "root_schedule": options.root_schedule,
        "speedup": options.speedup,
        "concurrency": options.concurrency,
        "preloaded_root_requests": root_payload_count,
        "preloaded_root_payload_bytes": root_payload_bytes,
        "root_preload_s": root_preload_s,
        "base_url": options.base_url,
        "model_override": options.model,
        "payload_mode": options.payload_mode,
        "prompt_source": (
            "recorded" if options.payload_mode == "recorded" else "live_chain"
        ),
        "live_output_policy": (
            "metrics_only"
            if options.payload_mode == "recorded"
            else "next_prompt"
        ),
        "chain_mismatch_policy": options.chain_mismatch_policy,
        "live_chain_fidelity": dict(chain_fidelity)
        if options.payload_mode == "live-chain"
        else None,
        "cache_usage": cache_usage,
        "results_jsonl": str(output_path),
    }
    with summary_path.open("x", encoding="utf-8") as summary_handle:
        summary_handle.write(json_dumps(summary, pretty=True) + "\n")
    summary["summary_path"] = str(summary_path)
    return summary
