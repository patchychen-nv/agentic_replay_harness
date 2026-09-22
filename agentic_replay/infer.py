from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .database import connect, initialize
from .util import json_dumps, parse_json_object


@dataclass
class AgentAccumulator:
    tree_id: str
    thread_id: str
    agent_name: str | None
    parent_agent_name: str | None
    session_id: str | None
    first_request_id: str
    last_request_id: str
    first_send_ts: float
    last_send_ts: float
    request_ids: list[str] = field(default_factory=list)
    relation_status: str = "unresolved"


@dataclass(frozen=True)
class IndexedItem:
    ordinal: int
    item_type: str
    item_id: str | None
    call_id: str | None
    role: str | None
    name: str | None
    canonical_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class PromptChainCandidate:
    source_request_id: str
    target_request_id: str
    source_input_items: int
    target_input_items: int
    common_prefix_items: int
    delta_start: int
    response_item_count: int
    matched_response_items: int
    matched_item_ids: int
    matched_call_ids: int
    tool_result_items: int
    source_tool_call_ids: int
    source_send_ts: float
    delay_s: float | None
    timestamp_conflict: bool


def _decode_nested_json(value: Any) -> Any:
    current = value
    for _ in range(3):
        if not isinstance(current, str):
            break
        try:
            current = json.loads(current)
        except json.JSONDecodeError:
            break
    return current


def _target_names(value: Any) -> set[str]:
    value = _decode_nested_json(value)
    found: set[str] = set()

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {"agent_name", "task_name", "target", "sender"} and isinstance(
                    child, str
                ):
                    found.add(child)
                elif key in {"agents", "agent_updates", "updates", "results"}:
                    walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return found


def _is_timeout(value: Any) -> bool:
    value = _decode_nested_json(value)
    return isinstance(value, dict) and bool(value.get("timed_out"))


def _canonical_target(
    tree_agents: dict[str, AgentAccumulator],
    source_agent: str | None,
    target: str,
) -> str | None:
    if target in tree_agents:
        return target
    if target.startswith("/"):
        return None
    if source_agent:
        direct = f"{source_agent.rstrip('/')}/{target}"
        if direct in tree_agents:
            return direct
    suffix_matches = [name for name in tree_agents if name.endswith(f"/{target}")]
    return suffix_matches[0] if len(suffix_matches) == 1 else None


def _insert_diagnostic(
    connection: sqlite3.Connection,
    severity: str,
    code: str,
    subject_id: str | None,
    detail: dict[str, Any],
) -> None:
    connection.execute(
        "INSERT INTO diagnostics(severity, code, subject_id, detail_json) VALUES(?, ?, ?, ?)",
        (severity, code, subject_id, json_dumps(detail)),
    )


def _insert_edge(
    connection: sqlite3.Connection,
    source: str,
    target: str,
    kind: str,
    delay_s: float | None,
    evidence_method: str,
    confidence: str,
    evidence: dict[str, Any],
) -> None:
    connection.execute(
        """
        INSERT OR REPLACE INTO dependency_edges(
          source_request_id, target_request_id, edge_kind, delay_s,
          evidence_method, confidence, evidence_json
        ) VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (
            source,
            target,
            kind,
            delay_s,
            evidence_method,
            confidence,
            json_dumps(evidence),
        ),
    )


def _load_lineage_items(
    connection: sqlite3.Connection, table: str
) -> dict[str, list[IndexedItem]]:
    if table not in {"request_input_items", "response_output_items"}:
        raise ValueError(f"unsupported lineage item table: {table}")
    by_request: dict[str, list[IndexedItem]] = defaultdict(list)
    for row in connection.execute(
        f"""
        SELECT request_id, ordinal, item_type, item_id, call_id, role, name,
               canonical_sha256, content_sha256
        FROM {table}
        ORDER BY request_id, ordinal
        """
    ):
        by_request[str(row["request_id"])].append(
            IndexedItem(
                ordinal=int(row["ordinal"]),
                item_type=str(row["item_type"]),
                item_id=str(row["item_id"]) if row["item_id"] is not None else None,
                call_id=str(row["call_id"])
                if row["call_id"] is not None
                else None,
                role=str(row["role"]) if row["role"] is not None else None,
                name=str(row["name"]) if row["name"] is not None else None,
                canonical_sha256=str(row["canonical_sha256"]),
                content_sha256=str(row["content_sha256"]),
            )
        )
    return by_request


def _common_input_prefix(left: list[IndexedItem], right: list[IndexedItem]) -> int:
    count = 0
    for left_item, right_item in zip(left, right):
        if left_item.canonical_sha256 != right_item.canonical_sha256:
            break
        count += 1
    return count


def _match_response_item(
    output: IndexedItem, target: IndexedItem
) -> tuple[bool, bool, bool]:
    if output.item_type != target.item_type:
        return False, False, False
    if output.item_id is not None and target.item_id is not None:
        if output.item_id != target.item_id:
            return False, False, False
        item_id_match = True
    else:
        item_id_match = False
    if output.call_id is not None and target.call_id is not None:
        if output.call_id != target.call_id:
            return False, False, False
        call_id_match = True
    else:
        call_id_match = False
    # IDs prove object identity; the normalized content hash proves that the
    # target consumed the same prompt-relevant value rather than merely
    # reusing an identifier.
    if output.content_sha256 != target.content_sha256:
        return False, False, False
    return True, item_id_match, call_id_match


def _prompt_chain_candidate(
    source: sqlite3.Row,
    target: sqlite3.Row,
    request_items: dict[str, list[IndexedItem]],
    response_items: dict[str, list[IndexedItem]],
    finish_ts_by_request: dict[str, float | None],
) -> PromptChainCandidate | None:
    source_id = str(source["trace_id"])
    target_id = str(target["trace_id"])
    source_input = request_items.get(source_id, [])
    target_input = request_items.get(target_id, [])
    output = response_items.get(source_id, [])
    if not source_input or not target_input or not output:
        return None
    prefix = _common_input_prefix(source_input, target_input)
    if prefix != len(source_input):
        return None
    delta_start = len(source_input)
    if delta_start + len(output) > len(target_input):
        return None

    matched_item_ids = 0
    matched_call_ids = 0
    for ordinal, output_item in enumerate(output):
        matched, item_id_match, call_id_match = _match_response_item(
            output_item, target_input[delta_start + ordinal]
        )
        if not matched:
            return None
        matched_item_ids += int(item_id_match)
        matched_call_ids += int(call_id_match)

    source_call_ids = {item.call_id for item in output if item.call_id is not None}
    suffix = target_input[delta_start + len(output) :]
    tool_result_items = sum(
        1
        for item in suffix
        if item.item_type.endswith("_output") and item.call_id in source_call_ids
    )
    finish_ts = finish_ts_by_request.get(source_id)
    delay_s = (
        float(target["send_ts"]) - finish_ts if finish_ts is not None else None
    )
    return PromptChainCandidate(
        source_request_id=source_id,
        target_request_id=target_id,
        source_input_items=len(source_input),
        target_input_items=len(target_input),
        common_prefix_items=prefix,
        delta_start=delta_start,
        response_item_count=len(output),
        matched_response_items=len(output),
        matched_item_ids=matched_item_ids,
        matched_call_ids=matched_call_ids,
        tool_result_items=tool_result_items,
        source_tool_call_ids=len(source_call_ids),
        source_send_ts=float(source["send_ts"]),
        delay_s=delay_s,
        timestamp_conflict=delay_s is not None and delay_s < -1e-6,
    )


def _infer_prompt_chains(connection: sqlite3.Connection) -> int:
    request_items = _load_lineage_items(connection, "request_input_items")
    response_items = _load_lineage_items(connection, "response_output_items")
    if not request_items or not response_items:
        return 0
    finish_ts_by_request = {
        str(row["trace_id"]): (
            float(row["finish_ts"]) if row["finish_ts"] is not None else None
        )
        for row in connection.execute("SELECT trace_id, finish_ts FROM responses")
    }

    previous_key: tuple[str, str, str] | None = None
    prior: list[sqlite3.Row] = []
    inferred = 0
    query = """
      SELECT trace_id, tree_id, thread_id, turn_id, send_ts, recorded_at
      FROM requests
      WHERE thread_id IS NOT NULL AND send_ts IS NOT NULL
      ORDER BY tree_id, thread_id, turn_id, send_ts, trace_id
    """
    for target in connection.execute(query):
        key = (
            str(target["tree_id"]),
            str(target["thread_id"]),
            str(target["turn_id"] or ""),
        )
        if key != previous_key:
            previous_key = key
            prior = []
        target_id = str(target["trace_id"])
        if target_id not in request_items:
            prior.append(target)
            continue
        candidates = [
            candidate
            for source in prior
            if (
                candidate := _prompt_chain_candidate(
                    source,
                    target,
                    request_items,
                    response_items,
                    finish_ts_by_request,
                )
            )
            is not None
        ]
        if candidates:
            # The longest consumed request prefix identifies the nearest
            # context frontier.  Explicit identities then break structural
            # ties; timestamp is intentionally only the final tie-breaker.
            chosen = max(
                candidates,
                key=lambda item: (
                    item.source_input_items,
                    item.matched_item_ids + item.matched_call_ids,
                    item.response_item_count,
                    item.source_send_ts,
                ),
            )
            identity_matches = chosen.matched_item_ids + chosen.matched_call_ids
            all_tool_results = (
                chosen.source_tool_call_ids == 0
                or chosen.tool_result_items >= chosen.source_tool_call_ids
            )
            confidence = "high" if identity_matches > 0 and all_tool_results else "medium"
            connection.execute(
                """
                INSERT OR REPLACE INTO prompt_chains(
                  target_request_id, source_request_id, relation_mode,
                  evidence_method, confidence, source_input_items,
                  target_input_items, common_prefix_items, delta_start,
                  response_item_count, matched_response_items,
                  matched_item_ids, matched_call_ids, tool_result_items,
                  delay_s, evidence_json
                ) VALUES(?, ?, 'append', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chosen.target_request_id,
                    chosen.source_request_id,
                    "request_prefix_response_frontier_ids_call_ids",
                    confidence,
                    chosen.source_input_items,
                    chosen.target_input_items,
                    chosen.common_prefix_items,
                    chosen.delta_start,
                    chosen.response_item_count,
                    chosen.matched_response_items,
                    chosen.matched_item_ids,
                    chosen.matched_call_ids,
                    chosen.tool_result_items,
                    chosen.delay_s,
                    json_dumps(
                        {
                            "candidate_count": len(candidates),
                            "tree_id": key[0],
                            "thread_id": key[1],
                            "turn_id": key[2],
                            "source_tool_call_ids": chosen.source_tool_call_ids,
                            "all_tool_results_observed": all_tool_results,
                            "timestamp_conflict": chosen.timestamp_conflict,
                            "target_recorded_at": target["recorded_at"],
                        }
                    ),
                ),
            )
            inferred += 1
            if chosen.timestamp_conflict:
                _insert_diagnostic(
                    connection,
                    "warning",
                    "prompt_chain_timestamp_conflict",
                    chosen.target_request_id,
                    {
                        "source_request_id": chosen.source_request_id,
                        "delay_s": chosen.delay_s,
                        "payload_edge_retained": True,
                    },
                )
        prior.append(target)
    return inferred


def _build_sequences(
    connection: sqlite3.Connection,
) -> tuple[
    dict[tuple[str, str], AgentAccumulator],
    dict[str, sqlite3.Row],
    dict[str, list[str]],
]:
    response_by_request = {
        str(row["trace_id"]): row
        for row in connection.execute(
            "SELECT trace_id, response_status, finish_ts, finished_at, output_tokens FROM responses"
        )
    }
    tool_names: dict[str, list[str]] = defaultdict(list)
    for row in connection.execute(
        "SELECT trace_id, name FROM response_tool_calls ORDER BY trace_id, ordinal"
    ):
        tool_names[str(row["trace_id"])].append(str(row["name"] or "unknown"))

    agents: dict[tuple[str, str], AgentAccumulator] = {}
    request_send_times: dict[str, float] = {}
    previous_key: tuple[str, str, str] | None = None
    previous_row: sqlite3.Row | None = None
    sequence_index = 0
    query = """
      SELECT trace_id, tree_id, thread_id, turn_id, agent_name, parent_agent_name,
             session_id, recorded_at, send_ts
      FROM requests
      WHERE thread_id IS NOT NULL AND send_ts IS NOT NULL
      ORDER BY tree_id, thread_id, turn_id, send_ts, trace_id
    """
    for row in connection.execute(query):
        tree_id = str(row["tree_id"])
        thread_id = str(row["thread_id"])
        agent_key = (tree_id, thread_id)
        sequence_key = (tree_id, thread_id, str(row["turn_id"] or ""))
        if sequence_key != previous_key:
            sequence_index = 0
            previous_row = None
            previous_key = sequence_key
        if agent_key not in agents:
            agents[agent_key] = AgentAccumulator(
                tree_id=tree_id,
                thread_id=thread_id,
                agent_name=row["agent_name"],
                parent_agent_name=row["parent_agent_name"],
                session_id=row["session_id"],
                first_request_id=str(row["trace_id"]),
                last_request_id=str(row["trace_id"]),
                first_send_ts=float(row["send_ts"]),
                last_send_ts=float(row["send_ts"]),
            )
        agent = agents[agent_key]
        request_id = str(row["trace_id"])
        request_send_times[request_id] = float(row["send_ts"])
        agent.request_ids.append(request_id)
        if float(row["send_ts"]) < agent.first_send_ts:
            agent.first_request_id = request_id
            agent.first_send_ts = float(row["send_ts"])
        if float(row["send_ts"]) >= agent.last_send_ts:
            agent.last_request_id = request_id
            agent.last_send_ts = float(row["send_ts"])
        if row["agent_name"] and agent.agent_name != row["agent_name"]:
            _insert_diagnostic(
                connection,
                "warning",
                "thread_agent_name_changed",
                request_id,
                {"first": agent.agent_name, "current": row["agent_name"]},
            )
        connection.execute(
            "UPDATE requests SET sequence_index=?, previous_request_id=? WHERE trace_id=?",
            (
                sequence_index,
                str(previous_row["trace_id"]) if previous_row is not None else None,
                request_id,
            ),
        )

        if previous_row is not None:
            source = str(previous_row["trace_id"])
            previous_response = response_by_request.get(source)
            finish_ts = (
                float(previous_response["finish_ts"])
                if previous_response is not None
                and previous_response["finish_ts"] is not None
                else None
            )
            delay = float(row["send_ts"]) - finish_ts if finish_ts is not None else None
            if delay is not None and delay < -1e-6:
                _insert_diagnostic(
                    connection,
                    "error",
                    "turn_requests_overlap",
                    request_id,
                    {
                        "previous_request_id": source,
                        "overlap_s": -delay,
                        "edge_omitted": True,
                    },
                )
            else:
                _insert_edge(
                    connection,
                    source,
                    request_id,
                    "thread_sequence",
                    max(delay, 0.0) if delay is not None else None,
                    "same_explicit_thread_id_server_order",
                    "high" if finish_ts is not None else "medium",
                    {
                        "thread_id": thread_id,
                        "turn_id": row["turn_id"],
                        "source_finished_at": previous_response["finished_at"]
                        if previous_response is not None
                        else None,
                        "target_recorded_at": row["recorded_at"],
                    },
                )
            names = tool_names.get(source, [])
            if delay is not None and delay >= 0:
                if len(names) == 1:
                    attribution = "single_tool_plus_client_overhead"
                elif len(names) > 1:
                    attribution = "aggregate_multi_tool_wait"
                else:
                    attribution = "client_or_unobserved_external_wait"
                connection.execute(
                    """
                    INSERT OR REPLACE INTO external_gaps(
                      source_request_id, target_request_id, duration_s,
                      tool_call_count, tool_names_json, attribution
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source,
                        request_id,
                        delay,
                        len(names),
                        json_dumps(names),
                        attribution,
                    ),
                )
        previous_row = row
        sequence_index += 1

    for agent in agents.values():
        agent.request_ids.sort(key=request_send_times.__getitem__)

    return agents, response_by_request, tool_names


def _store_agents(
    connection: sqlite3.Connection,
    agents: dict[tuple[str, str], AgentAccumulator],
) -> None:
    for agent in agents.values():
        connection.execute(
            """
            INSERT INTO agents(
              tree_id, thread_id, agent_name, parent_agent_name, session_id,
              first_request_id, last_request_id, first_send_ts, last_send_ts,
              request_count, relation_status
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent.tree_id,
                agent.thread_id,
                agent.agent_name,
                agent.parent_agent_name,
                agent.session_id,
                agent.first_request_id,
                agent.last_request_id,
                agent.first_send_ts,
                agent.last_send_ts,
                len(agent.request_ids),
                agent.relation_status,
            ),
        )


def _control_calls(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            """
            SELECT c.call_id, c.trace_id, c.name, c.arguments_json,
                   q.tree_id, q.agent_name AS source_agent_name,
                   q.thread_id AS source_thread_id,
                   s.finish_ts, s.finished_at,
                   r.observed_in_request_id, r.observed_ts, r.output_json,
                   r.output_truncated
            FROM response_tool_calls c
            JOIN requests q ON q.trace_id=c.trace_id
            LEFT JOIN responses s ON s.trace_id=c.trace_id
            LEFT JOIN control_tool_results r ON r.call_id=c.call_id
            WHERE c.is_control=1
            ORDER BY q.tree_id, s.finish_ts, c.ordinal
            """
        )
    )


def _infer_agent_edges(
    connection: sqlite3.Connection,
    agents: dict[tuple[str, str], AgentAccumulator],
    response_by_request: dict[str, sqlite3.Row],
) -> None:
    by_tree_name: dict[str, dict[str, AgentAccumulator]] = defaultdict(dict)
    for agent in agents.values():
        if agent.agent_name:
            by_tree_name[agent.tree_id][agent.agent_name] = agent

    controls = _control_calls(connection)
    spawns: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in controls:
        if row["name"] == "spawn_agent" and row["source_agent_name"]:
            spawns[(str(row["tree_id"]), str(row["source_agent_name"]))].append(row)

    for agent in agents.values():
        if agent.parent_agent_name is None:
            agent.relation_status = "root"
            continue
        tree_agents = by_tree_name[agent.tree_id]
        parent = tree_agents.get(agent.parent_agent_name)
        if parent is None:
            agent.relation_status = "missing_parent_agent"
            _insert_diagnostic(
                connection,
                "error",
                "missing_parent_agent",
                agent.first_request_id,
                {
                    "tree_id": agent.tree_id,
                    "agent_name": agent.agent_name,
                    "expected_parent": agent.parent_agent_name,
                },
            )
            continue

        exact: list[sqlite3.Row] = []
        fallback: list[sqlite3.Row] = []
        for spawn in spawns.get((agent.tree_id, agent.parent_agent_name), []):
            if spawn["finish_ts"] is None or float(spawn["finish_ts"]) > agent.first_send_ts + 2.0:
                continue
            fallback.append(spawn)
            arguments = parse_json_object(spawn["arguments_json"])
            task_name = arguments.get("task_name")
            expected = (
                f"{agent.parent_agent_name.rstrip('/')}/{task_name}"
                if isinstance(task_name, str)
                else None
            )
            if expected == agent.agent_name:
                exact.append(spawn)
        candidates = exact or fallback
        if not candidates:
            agent.relation_status = "parent_path_only"
            _insert_diagnostic(
                connection,
                "warning",
                "spawn_call_not_found",
                agent.first_request_id,
                {
                    "tree_id": agent.tree_id,
                    "agent_name": agent.agent_name,
                    "parent_agent_name": agent.parent_agent_name,
                },
            )
            continue
        chosen = max(candidates, key=lambda row: float(row["finish_ts"]))
        delay = max(agent.first_send_ts - float(chosen["finish_ts"]), 0.0)
        confidence = "high" if exact else "medium"
        method = (
            "spawn_agent_task_name_plus_agent_path"
            if exact
            else "nearest_parent_spawn_before_child"
        )
        _insert_edge(
            connection,
            str(chosen["trace_id"]),
            agent.first_request_id,
            "spawn",
            delay,
            method,
            confidence,
            {
                "call_id": chosen["call_id"],
                "arguments": parse_json_object(chosen["arguments_json"]),
                "agent_name": agent.agent_name,
                "candidate_count": len(candidates),
            },
        )
        agent.relation_status = "spawn_matched" if exact else "spawn_inferred"

    # Persist relation status after matching.
    for agent in agents.values():
        connection.execute(
            "UPDATE agents SET relation_status=? WHERE tree_id=? AND thread_id=?",
            (agent.relation_status, agent.tree_id, agent.thread_id),
        )

    for control in controls:
        name = str(control["name"])
        if name not in {"wait_agent", "interrupt_agent", "followup_task"}:
            continue
        tree_id = str(control["tree_id"])
        tree_agents = by_tree_name.get(tree_id, {})
        source_agent = control["source_agent_name"]
        result_request = control["observed_in_request_id"]

        if name == "followup_task":
            arguments = parse_json_object(control["arguments_json"])
            target = arguments.get("target")
            if not isinstance(target, str) or control["finish_ts"] is None:
                continue
            canonical = _canonical_target(tree_agents, source_agent, target)
            target_agent = tree_agents.get(canonical) if canonical else None
            if target_agent is None:
                _insert_diagnostic(
                    connection,
                    "warning",
                    "followup_target_unresolved",
                    str(control["trace_id"]),
                    {"target": target, "tree_id": tree_id},
                )
                continue
            next_requests = [
                request_id
                for request_id in target_agent.request_ids
                if (
                    connection.execute(
                        "SELECT send_ts FROM requests WHERE trace_id=?", (request_id,)
                    ).fetchone()[0]
                    > float(control["finish_ts"])
                )
            ]
            if next_requests:
                target_request = next_requests[0]
                target_ts = connection.execute(
                    "SELECT send_ts FROM requests WHERE trace_id=?", (target_request,)
                ).fetchone()[0]
                _insert_edge(
                    connection,
                    str(control["trace_id"]),
                    target_request,
                    "followup_resume",
                    max(float(target_ts) - float(control["finish_ts"]), 0.0),
                    "followup_task_target_plus_next_target_request",
                    "high",
                    {"call_id": control["call_id"], "target": canonical},
                )
            continue

        if not result_request or control["output_json"] is None:
            _insert_diagnostic(
                connection,
                "info",
                "control_result_not_observed",
                str(control["trace_id"]),
                {"call_id": control["call_id"], "tool_name": name},
            )
            continue
        if name == "wait_agent" and _is_timeout(control["output_json"]):
            continue

        targets = _target_names(control["output_json"])
        if name == "interrupt_agent":
            arguments = parse_json_object(control["arguments_json"])
            if isinstance(arguments.get("target"), str):
                targets.add(arguments["target"])
        resolved: list[AgentAccumulator] = []
        for target in targets:
            canonical = _canonical_target(tree_agents, source_agent, target)
            candidate = tree_agents.get(canonical) if canonical else None
            if candidate and candidate.agent_name != source_agent and candidate not in resolved:
                resolved.append(candidate)

        result_send_row = connection.execute(
            "SELECT send_ts FROM requests WHERE trace_id=?", (result_request,)
        ).fetchone()
        result_send = float(result_send_row[0]) if result_send_row else None
        if not resolved and name == "wait_agent" and result_send is not None:
            start = float(control["finish_ts"] or float("-inf"))
            temporal = []
            for candidate in tree_agents.values():
                if candidate.agent_name == source_agent:
                    continue
                finishes = [
                    float(response_by_request[request_id]["finish_ts"])
                    for request_id in candidate.request_ids
                    if request_id in response_by_request
                    and response_by_request[request_id]["finish_ts"] is not None
                    and start
                    <= float(response_by_request[request_id]["finish_ts"])
                    <= result_send
                ]
                if finishes:
                    temporal.append(candidate)
            if len(temporal) == 1:
                resolved = temporal

        for target_agent in resolved:
            completed_candidates = [
                (float(response_by_request[request_id]["finish_ts"]), request_id)
                for request_id in target_agent.request_ids
                if request_id in response_by_request
                and response_by_request[request_id]["finish_ts"] is not None
                and (
                    result_send is None
                    or float(response_by_request[request_id]["finish_ts"])
                    <= result_send + 1e-6
                )
            ]
            if not completed_candidates:
                continue
            terminal_finish, terminal_request_id = max(completed_candidates)
            delay = (
                max(result_send - terminal_finish, 0.0)
                if result_send is not None and terminal_finish is not None
                else None
            )
            _insert_edge(
                connection,
                terminal_request_id,
                str(result_request),
                "join" if name == "wait_agent" else "interrupt_join",
                delay,
                f"{name}_result_agent_identity",
                "high" if targets else "medium",
                {
                    "call_id": control["call_id"],
                    "agent_name": target_agent.agent_name,
                    "result_truncated": bool(control["output_truncated"]),
                },
            )

    # agent_message is the strongest join evidence in the Responses schema:
    # the recipient's recorded payload explicitly contains a message authored
    # by another canonical agent. FINAL_ANSWER marks child completion.
    for message in connection.execute(
        """
        SELECT m.*, q.tree_id, q.agent_name AS target_agent_name, q.send_ts
        FROM agent_messages m
        JOIN requests q ON q.trace_id=m.surfaced_in_request_id
        ORDER BY m.surfaced_ts, m.message_id
        """
    ):
        tree_id = str(message["tree_id"])
        author_name = message["author_agent_name"]
        if not author_name:
            continue
        tree_agents = by_tree_name.get(tree_id, {})
        author = tree_agents.get(str(author_name))
        if author is None:
            _insert_diagnostic(
                connection,
                "warning",
                "agent_message_author_unresolved",
                str(message["message_id"]),
                {"tree_id": tree_id, "author": author_name},
            )
            continue
        target_send = float(message["send_ts"])
        candidates: list[tuple[float, str]] = []
        for request_id in author.request_ids:
            response = response_by_request.get(request_id)
            if response is None or response["finish_ts"] is None:
                continue
            finish_ts = float(response["finish_ts"])
            if finish_ts <= target_send + 1e-6:
                candidates.append((finish_ts, request_id))
        if not candidates:
            _insert_diagnostic(
                connection,
                "warning",
                "agent_message_source_request_unresolved",
                str(message["message_id"]),
                {
                    "tree_id": tree_id,
                    "author": author_name,
                    "surfaced_in_request_id": message["surfaced_in_request_id"],
                },
            )
            continue
        finish_ts, source_request_id = max(candidates)
        message_type = str(message["message_type"] or "MESSAGE")
        edge_kind = (
            "agent_final_join"
            if message_type == "FINAL_ANSWER"
            else "agent_message_delivery"
        )
        _insert_edge(
            connection,
            source_request_id,
            str(message["surfaced_in_request_id"]),
            edge_kind,
            max(target_send - finish_ts, 0.0),
            "explicit_agent_message_author_recipient",
            "high" if message_type == "FINAL_ANSWER" else "medium",
            {
                "message_id": message["message_id"],
                "message_type": message_type,
                "author": author_name,
                "recipient": message["recipient_agent_name"],
                "target_agent_name": message["target_agent_name"],
                "content_sha256": message["content_sha256"],
            },
        )


def _build_trees(connection: sqlite3.Connection) -> None:
    tree_ids = [
        str(row[0])
        for row in connection.execute(
            "SELECT DISTINCT tree_id FROM requests WHERE tree_id IS NOT NULL ORDER BY tree_id"
        )
    ]
    for tree_id in tree_ids:
        stats = connection.execute(
            """
            SELECT COUNT(*) AS request_count,
                   SUM(replay_eligible) AS eligible_count,
                   MIN(send_ts) AS start_ts,
                   MAX(COALESCE(s.finish_ts, q.send_ts)) AS end_ts,
                   MIN(recorded_at) AS started_at,
                   MIN(session_id) AS session_id
            FROM requests q LEFT JOIN responses s ON s.trace_id=q.trace_id
            WHERE tree_id=?
            """,
            (tree_id,),
        ).fetchone()
        agent_stats = connection.execute(
            """
            SELECT COUNT(*) AS agent_count,
                   SUM(CASE WHEN relation_status IN ('missing_parent_agent', 'parent_path_only', 'unresolved')
                            THEN 1 ELSE 0 END) AS unresolved
            FROM agents WHERE tree_id=?
            """,
            (tree_id,),
        ).fetchone()
        root = connection.execute(
            """
            SELECT trace_id FROM requests
            WHERE tree_id=?
            ORDER BY CASE WHEN agent_name='/root' THEN 0 ELSE 1 END, send_ts, trace_id
            LIMIT 1
            """,
            (tree_id,),
        ).fetchone()
        request_count = int(stats["request_count"] or 0)
        eligible_count = int(stats["eligible_count"] or 0)
        agent_count = int(agent_stats["agent_count"] or 0)
        unresolved = int(agent_stats["unresolved"] or 0)
        replayable = int(
            request_count > 0
            and eligible_count == request_count
            and unresolved == 0
        )
        if eligible_count == request_count:
            capture_status = "complete"
        elif eligible_count == 0:
            capture_status = "unreplayable"
        else:
            capture_status = "partial"
        connection.execute(
            """
            INSERT OR REPLACE INTO trees(
              tree_id, session_id, root_request_id, started_at, start_ts,
              end_ts, request_count, eligible_request_count, agent_count,
              unresolved_agent_count, capture_status, replayable
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tree_id,
                stats["session_id"],
                root["trace_id"] if root else None,
                stats["started_at"],
                stats["start_ts"],
                stats["end_ts"],
                request_count,
                eligible_count,
                agent_count,
                unresolved,
                capture_status,
                replayable,
            ),
        )


def infer_graph(database_path: Path) -> dict[str, int]:
    connection = connect(database_path)
    # ``initialize`` is also the non-destructive schema-upgrade path for
    # indexes created before prompt-lineage tables were introduced.
    initialize(connection)
    connection.executescript(
        """
        DELETE FROM dependency_edges;
        DELETE FROM external_gaps;
        DELETE FROM prompt_chains;
        DELETE FROM agents;
        DELETE FROM trees;
        DELETE FROM diagnostics;
        UPDATE requests SET sequence_index=NULL, previous_request_id=NULL;
        """
    )
    agents, response_by_request, _ = _build_sequences(connection)
    prompt_chains = _infer_prompt_chains(connection)
    _store_agents(connection, agents)
    _infer_agent_edges(connection, agents, response_by_request)
    _build_trees(connection)
    connection.commit()
    summary = {
        "trees": int(connection.execute("SELECT COUNT(*) FROM trees").fetchone()[0]),
        "replayable_trees": int(
            connection.execute("SELECT COUNT(*) FROM trees WHERE replayable=1").fetchone()[0]
        ),
        "agents": int(connection.execute("SELECT COUNT(*) FROM agents").fetchone()[0]),
        "edges": int(
            connection.execute("SELECT COUNT(*) FROM dependency_edges").fetchone()[0]
        ),
        "prompt_chains": prompt_chains,
        "diagnostics": int(
            connection.execute("SELECT COUNT(*) FROM diagnostics").fetchone()[0]
        ),
    }
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('graph_summary', ?)",
        (json_dumps(summary),),
    )
    connection.commit()
    connection.close()
    return summary
