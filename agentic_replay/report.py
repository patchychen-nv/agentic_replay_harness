from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .database import connect
from .util import json_dumps


def _count_by(connection: Any, query: str) -> dict[str, int]:
    return {
        str(row[0] if row[0] is not None else "null"): int(row[1])
        for row in connection.execute(query)
    }


def inspect_index(database_path: Path) -> dict[str, Any]:
    connection = connect(database_path, readonly=True)
    scalar = lambda query: int(connection.execute(query).fetchone()[0] or 0)
    summary: dict[str, Any] = {
        "database": str(database_path.absolute()),
        "requests": scalar("SELECT COUNT(*) FROM requests"),
        "responses": scalar("SELECT COUNT(*) FROM responses"),
        "paired_requests": scalar(
            "SELECT COUNT(*) FROM requests q JOIN responses s ON s.trace_id=q.trace_id"
        ),
        "orphan_requests": scalar(
            "SELECT COUNT(*) FROM requests q WHERE NOT EXISTS "
            "(SELECT 1 FROM responses s WHERE s.trace_id=q.trace_id)"
        ),
        "orphan_responses": scalar(
            "SELECT COUNT(*) FROM responses s WHERE NOT EXISTS "
            "(SELECT 1 FROM requests q WHERE q.trace_id=s.trace_id)"
        ),
        "replay_eligible_requests": scalar(
            "SELECT COUNT(*) FROM requests WHERE replay_eligible=1"
        ),
        "request_routes": _count_by(
            connection, "SELECT route, COUNT(*) FROM requests GROUP BY route ORDER BY COUNT(*) DESC"
        ),
        "response_statuses": _count_by(
            connection,
            "SELECT response_status, COUNT(*) FROM responses GROUP BY response_status "
            "ORDER BY COUNT(*) DESC",
        ),
        "eligibility_reasons": _count_by(
            connection,
            "SELECT eligibility_reason, COUNT(*) FROM requests GROUP BY eligibility_reason "
            "ORDER BY COUNT(*) DESC",
        ),
        "metadata_coverage": {
            "session_id": scalar("SELECT COUNT(*) FROM requests WHERE session_id IS NOT NULL"),
            "thread_id": scalar("SELECT COUNT(*) FROM requests WHERE thread_id IS NOT NULL"),
            "turn_id": scalar("SELECT COUNT(*) FROM requests WHERE turn_id IS NOT NULL"),
            "root_turn_id": scalar(
                "SELECT COUNT(*) FROM requests WHERE root_turn_id IS NOT NULL"
            ),
            "agent_name": scalar("SELECT COUNT(*) FROM requests WHERE agent_name IS NOT NULL"),
        },
        "identity_counts": {
            "sessions": scalar(
                "SELECT COUNT(DISTINCT session_id) FROM requests "
                "WHERE session_id IS NOT NULL AND session_id != '_no_session'"
            ),
            "sentinel_sessions": scalar(
                "SELECT COUNT(DISTINCT session_id) FROM requests "
                "WHERE session_id = '_no_session'"
            ),
            "threads": scalar(
                "SELECT COUNT(DISTINCT thread_id) FROM requests WHERE thread_id IS NOT NULL"
            ),
            "root_turns": scalar(
                "SELECT COUNT(DISTINCT root_turn_id) FROM requests "
                "WHERE root_turn_id IS NOT NULL"
            ),
        },
        "trees": scalar("SELECT COUNT(*) FROM trees"),
        "tree_statuses": _count_by(
            connection,
            "SELECT capture_status, COUNT(*) FROM trees GROUP BY capture_status "
            "ORDER BY COUNT(*) DESC",
        ),
        "replayable_trees": scalar("SELECT COUNT(*) FROM trees WHERE replayable=1"),
        "agents": scalar("SELECT COUNT(*) FROM agents"),
        "root_agents": scalar("SELECT COUNT(*) FROM agents WHERE parent_agent_name IS NULL"),
        "subagents": scalar("SELECT COUNT(*) FROM agents WHERE parent_agent_name IS NOT NULL"),
        "agent_relation_statuses": _count_by(
            connection,
            "SELECT relation_status, COUNT(*) FROM agents GROUP BY relation_status "
            "ORDER BY COUNT(*) DESC",
        ),
        "dependency_edges": _count_by(
            connection,
            "SELECT edge_kind, COUNT(*) FROM dependency_edges GROUP BY edge_kind "
            "ORDER BY COUNT(*) DESC",
        ),
        "lineage_items": {
            "request_input_items": scalar("SELECT COUNT(*) FROM request_input_items"),
            "response_output_items": scalar("SELECT COUNT(*) FROM response_output_items"),
        },
        "prompt_chains": scalar("SELECT COUNT(*) FROM prompt_chains"),
        "prompt_chain_modes": _count_by(
            connection,
            "SELECT relation_mode, COUNT(*) FROM prompt_chains GROUP BY relation_mode "
            "ORDER BY COUNT(*) DESC",
        ),
        "prompt_chain_confidence": _count_by(
            connection,
            "SELECT confidence, COUNT(*) FROM prompt_chains GROUP BY confidence "
            "ORDER BY COUNT(*) DESC",
        ),
        "edge_confidence": _count_by(
            connection,
            "SELECT confidence, COUNT(*) FROM dependency_edges GROUP BY confidence "
            "ORDER BY COUNT(*) DESC",
        ),
        "external_gap_attribution": _count_by(
            connection,
            "SELECT attribution, COUNT(*) FROM external_gaps GROUP BY attribution "
            "ORDER BY COUNT(*) DESC",
        ),
        "control_tool_calls": _count_by(
            connection,
            "SELECT name, COUNT(*) FROM response_tool_calls WHERE is_control=1 "
            "GROUP BY name ORDER BY COUNT(*) DESC",
        ),
        "agent_messages": scalar("SELECT COUNT(*) FROM agent_messages"),
        "agent_message_types": _count_by(
            connection,
            "SELECT message_type, COUNT(*) FROM agent_messages GROUP BY message_type "
            "ORDER BY COUNT(*) DESC",
        ),
        "perf_rows": scalar("SELECT COUNT(*) FROM perf_metrics"),
        "requests_with_ttft": scalar(
            """
            SELECT COUNT(*) FROM requests q
            JOIN responses s ON s.trace_id=q.trace_id
            JOIN perf_metrics p ON p.disagg_request_id=s.disagg_request_id
            WHERE p.ttft_s IS NOT NULL
            """
        ),
        "diagnostics": _count_by(
            connection,
            "SELECT severity || ':' || code, COUNT(*) FROM diagnostics "
            "GROUP BY severity, code ORDER BY COUNT(*) DESC",
        ),
    }
    top_agent_names = list(
        connection.execute(
            """
            SELECT agent_name, COUNT(*) AS count
            FROM agents WHERE agent_name IS NOT NULL
            GROUP BY agent_name ORDER BY count DESC, agent_name LIMIT 25
            """
        )
    )
    summary["top_agent_names"] = [
        {"agent_name": row[0], "count": int(row[1])} for row in top_agent_names
    ]
    connection.close()
    return summary


def export_graph(database_path: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = connect(database_path, readonly=True)
    source_by_id = {
        int(row["source_id"]): row
        for row in connection.execute("SELECT * FROM sources")
    }
    dependencies: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in connection.execute(
        "SELECT * FROM dependency_edges ORDER BY target_request_id, edge_kind, source_request_id"
    ):
        dependencies[str(row["target_request_id"])].append(
            {
                "request_id": row["source_request_id"],
                "kind": row["edge_kind"],
                "delay_s": row["delay_s"],
                "confidence": row["confidence"],
                "evidence_method": row["evidence_method"],
                "evidence": json.loads(row["evidence_json"] or "{}"),
            }
        )

    prompt_chains: dict[str, dict[str, Any]] = {}
    for row in connection.execute(
        "SELECT * FROM prompt_chains ORDER BY target_request_id"
    ):
        prompt_chains[str(row["target_request_id"])] = {
            "request_id": row["source_request_id"],
            "relation_mode": row["relation_mode"],
            "confidence": row["confidence"],
            "evidence_method": row["evidence_method"],
            "source_input_items": row["source_input_items"],
            "delta_start": row["delta_start"],
            "response_item_count": row["response_item_count"],
            "tool_result_items": row["tool_result_items"],
            "evidence": json.loads(row["evidence_json"] or "{}"),
        }

    agents_by_tree: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in connection.execute(
        "SELECT * FROM agents ORDER BY tree_id, first_send_ts, agent_name"
    ):
        agents_by_tree[str(row["tree_id"])].append(
            {
                "thread_id": row["thread_id"],
                "agent_name": row["agent_name"],
                "parent_agent_name": row["parent_agent_name"],
                "session_id": row["session_id"],
                "first_request_id": row["first_request_id"],
                "last_request_id": row["last_request_id"],
                "request_count": row["request_count"],
                "relation_status": row["relation_status"],
            }
        )

    requests_by_tree: dict[str, list[dict[str, Any]]] = defaultdict(list)
    request_query = """
      SELECT q.*, s.response_status, s.finished_at, s.finish_ts,
             s.input_tokens, s.output_tokens, s.cached_tokens, s.finish_reason,
             s.disagg_request_id, p.ttft_s
      FROM requests q
      LEFT JOIN responses s ON s.trace_id=q.trace_id
      LEFT JOIN perf_metrics p ON p.disagg_request_id=s.disagg_request_id
      ORDER BY q.tree_id, q.send_ts, q.trace_id
    """
    for row in connection.execute(request_query):
        source = source_by_id[int(row["source_id"])]
        request_id = str(row["trace_id"])
        requests_by_tree[str(row["tree_id"])].append(
            {
                "request_id": request_id,
                "route": row["route"],
                "session_id": row["session_id"],
                "thread_id": row["thread_id"],
                "turn_id": row["turn_id"],
                "root_turn_id": row["root_turn_id"],
                "agent_name": row["agent_name"],
                "sequence_index": row["sequence_index"],
                "previous_request_id": row["previous_request_id"],
                "recorded_at": row["recorded_at"],
                "finished_at": row["finished_at"],
                "model": row["model"],
                "response_status": row["response_status"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "cached_tokens": row["cached_tokens"],
                "ttft_s": row["ttft_s"],
                "replay_eligible": bool(row["replay_eligible"]),
                "eligibility_reason": row["eligibility_reason"],
                "depends_on": dependencies.get(request_id, []),
                "chain_parent": prompt_chains.get(request_id),
                "payload_ref": {
                    "path": source["relative_path"],
                    "absolute_path": source["absolute_path"],
                    "byte_offset": row["line_offset"],
                    "byte_length": row["line_length"],
                    "source_size_bytes": source["size_bytes"],
                    "source_mtime_ns": source["mtime_ns"],
                    "json_pointer": "/body",
                },
            }
        )

    trees_path = output_dir / "replay_trees.jsonl"
    with trees_path.open("w", encoding="utf-8") as handle:
        for tree in connection.execute("SELECT * FROM trees ORDER BY start_ts, tree_id"):
            tree_id = str(tree["tree_id"])
            value = {
                "schema": "agentic_replay_tree/v1",
                "tree_id": tree_id,
                "session_id": tree["session_id"],
                "root_request_id": tree["root_request_id"],
                "started_at": tree["started_at"],
                "capture_status": tree["capture_status"],
                "replayable": bool(tree["replayable"]),
                "agents": agents_by_tree.get(tree_id, []),
                "requests": requests_by_tree.get(tree_id, []),
            }
            handle.write(json_dumps(value) + "\n")

    gaps_path = output_dir / "external_gaps.jsonl"
    with gaps_path.open("w", encoding="utf-8") as handle:
        for row in connection.execute(
            "SELECT * FROM external_gaps ORDER BY source_request_id, target_request_id"
        ):
            handle.write(
                json_dumps(
                    {
                        "source_request_id": row["source_request_id"],
                        "target_request_id": row["target_request_id"],
                        "duration_s": row["duration_s"],
                        "tool_call_count": row["tool_call_count"],
                        "tool_names": json.loads(row["tool_names_json"]),
                        "attribution": row["attribution"],
                        "semantics": (
                            "Aggregate response-finish to next-request-arrival gap; "
                            "not an individual tool duration."
                        ),
                    }
                )
                + "\n"
            )

    diagnostics_path = output_dir / "diagnostics.jsonl"
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        for row in connection.execute("SELECT * FROM diagnostics ORDER BY diagnostic_id"):
            handle.write(
                json_dumps(
                    {
                        "severity": row["severity"],
                        "code": row["code"],
                        "subject_id": row["subject_id"],
                        "detail": json.loads(row["detail_json"] or "{}"),
                    }
                )
                + "\n"
            )

    manifest_path = output_dir / "manifest.json"
    manifest = inspect_index(database_path)
    manifest.update(
        {
            "schema": "agentic_replay_manifest/v1",
            "payload_storage": "source_jsonl_byte_range",
            "timing_semantics": {
                "send": "TRT-LLM server recorded_at (wall clock)",
                "finish": "TRT-LLM server finished_at (wall clock)",
                "external_gap": (
                    "next recorded_at - previous finished_at; includes tool execution, "
                    "scheduler, transport, and client overhead"
                ),
                "ttft": "disagg steady-clock first_token - arrival; audit only",
            },
            "files": {
                "trees": trees_path.name,
                "external_gaps": gaps_path.name,
                "diagnostics": diagnostics_path.name,
            },
        }
    )
    manifest_path.write_text(json_dumps(manifest, pretty=True) + "\n", encoding="utf-8")
    connection.close()
    return {
        "manifest": str(manifest_path),
        "trees": str(trees_path),
        "external_gaps": str(gaps_path),
        "diagnostics": str(diagnostics_path),
    }
