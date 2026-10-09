"""Regression tests for logical agent ancestry without a captured spawn edge."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_replay.database import connect
from agentic_replay.infer import infer_graph
from agentic_replay.ingest import build_index


def _request(
    trace_id: str,
    recorded_at: str,
    *,
    tree_id: str,
    thread_id: str,
    agent_name: str,
    input_items: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    session_id = "session-1"
    metadata = {
        "session_id": session_id,
        "thread_id": thread_id,
        "turn_id": f"turn-{thread_id}",
        "root_turn_id": tree_id,
        "agent_name": agent_name,
    }
    return {
        "event": "request",
        "trace_id": trace_id,
        "route": "/v1/responses",
        "session": session_id,
        "status": "accepted",
        "recorded_at": recorded_at,
        "body": {
            "model": "recorded-model",
            "stream": True,
            "input": input_items
            if input_items is not None
            else [{"type": "message", "role": "user", "content": "fixture"}],
            "client_metadata": {
                **metadata,
                "x-codex-turn-metadata": json.dumps(metadata),
            },
        },
    }


def _response(
    trace_id: str,
    finished_at: str,
    *,
    spawn_call_id: str | None = None,
) -> dict[str, object]:
    events: list[dict[str, object]] = []
    if spawn_call_id is not None:
        events.append(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": spawn_call_id,
                    "name": "spawn_agent",
                    "arguments": json.dumps({"task_name": "child"}),
                },
            }
        )
    events.append(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 2},
                "output": [],
            },
        }
    )
    body = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        for event in events
    )
    return {
        "event": "response",
        "trace_id": trace_id,
        "session": "session-1",
        "status": "completed",
        "finished_at": finished_at,
        "response": {"kind": "sse_text", "body": body},
    }


def _historical_spawn(*, output_path: str | None) -> list[dict[str, object]]:
    items: list[dict[str, object]] = [
        {
            "type": "function_call",
            "call_id": "historical-spawn",
            "name": "spawn_agent",
            "arguments": json.dumps({"task_name": "child"}),
        }
    ]
    if output_path is not None:
        items.append(
            {
                "type": "function_call_output",
                "call_id": "historical-spawn",
                "output": json.dumps({"task_name": output_path}),
            }
        )
    return items


def _build(
    root: Path,
    requests: list[dict[str, object]],
    responses: list[dict[str, object]],
) -> Path:
    hour = root / "attempt-001" / "request_trace" / "2026-01-01T00"
    hour.mkdir(parents=True)
    (hour / "requests-1.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in requests),
        encoding="utf-8",
    )
    (hour / "responses-1.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in responses),
        encoding="utf-8",
    )
    database = root / "replay.sqlite3"
    build_index(root / "attempt-001", database, include_perf=False, progress_every=0)
    infer_graph(database)
    return database


class LogicalAgentRelationsTest(unittest.TestCase):
    def test_history_proves_parent_but_not_executable_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = _build(
                Path(temporary),
                [
                    _request(
                        "child-1",
                        "2026-01-01T00:00:01+00:00",
                        tree_id="tree-1",
                        thread_id="thread-child",
                        agent_name="/root/child",
                    ),
                    _request(
                        "parent-1",
                        "2026-01-01T00:00:02+00:00",
                        tree_id="tree-1",
                        thread_id="thread-parent",
                        agent_name="/root",
                        input_items=_historical_spawn(output_path="/root/child"),
                    ),
                ],
                [
                    _response("child-1", "2026-01-01T00:00:01.500000+00:00"),
                    _response("parent-1", "2026-01-01T00:00:03+00:00"),
                ],
            )
            connection = connect(database)
            try:
                child = connection.execute(
                    "SELECT relation_status FROM agents WHERE tree_id='tree-1' "
                    "AND agent_name='/root/child'"
                ).fetchone()
                self.assertEqual(child[0], "spawn_history_only")
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM dependency_edges WHERE edge_kind='spawn' "
                        "AND target_request_id='child-1'"
                    ).fetchone()[0],
                    0,
                )
                tree = connection.execute(
                    "SELECT capture_status, replayable, unresolved_agent_count "
                    "FROM trees WHERE tree_id='tree-1'"
                ).fetchone()
                self.assertEqual(tuple(tree), ("complete", 0, 1))
                diagnostic = connection.execute(
                    "SELECT detail_json FROM diagnostics "
                    "WHERE code='spawn_history_logical_relation' "
                    "AND subject_id='child-1'"
                ).fetchone()
                self.assertIsNotNone(diagnostic)
                self.assertIs(json.loads(diagnostic[0])["executable_edge_created"], False)
            finally:
                connection.close()

    def test_history_requires_matching_output_path(self) -> None:
        for output_path in (None, "/root/not_child"):
            with self.subTest(output_path=output_path):
                with tempfile.TemporaryDirectory() as temporary:
                    database = _build(
                        Path(temporary),
                        [
                            _request(
                                "child-1",
                                "2026-01-01T00:00:01+00:00",
                                tree_id="tree-1",
                                thread_id="thread-child",
                                agent_name="/root/child",
                            ),
                            _request(
                                "parent-1",
                                "2026-01-01T00:00:02+00:00",
                                tree_id="tree-1",
                                thread_id="thread-parent",
                                agent_name="/root",
                                input_items=_historical_spawn(output_path=output_path),
                            ),
                        ],
                        [
                            _response("child-1", "2026-01-01T00:00:01.500000+00:00"),
                            _response("parent-1", "2026-01-01T00:00:03+00:00"),
                        ],
                    )
                    connection = connect(database)
                    try:
                        child = connection.execute(
                            "SELECT relation_status FROM agents WHERE tree_id='tree-1' "
                            "AND agent_name='/root/child'"
                        ).fetchone()
                        self.assertEqual(child[0], "parent_path_only")
                        self.assertEqual(
                            connection.execute(
                                "SELECT COUNT(*) FROM dependency_edges "
                                "WHERE edge_kind='spawn'"
                            ).fetchone()[0],
                            0,
                        )
                    finally:
                        connection.close()

    def test_cross_tree_thread_continuation_is_not_a_second_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = _build(
                Path(temporary),
                [
                    _request(
                        "parent-a",
                        "2026-01-01T00:00:01+00:00",
                        tree_id="tree-a",
                        thread_id="thread-parent",
                        agent_name="/root",
                    ),
                    _request(
                        "child-a",
                        "2026-01-01T00:00:02.500000+00:00",
                        tree_id="tree-a",
                        thread_id="thread-child",
                        agent_name="/root/child",
                    ),
                    _request(
                        "child-b",
                        "2026-01-01T00:00:04+00:00",
                        tree_id="tree-b",
                        thread_id="thread-child",
                        agent_name="/root/child",
                    ),
                ],
                [
                    _response(
                        "parent-a",
                        "2026-01-01T00:00:02+00:00",
                        spawn_call_id="recorded-spawn",
                    ),
                    _response("child-a", "2026-01-01T00:00:03+00:00"),
                    _response("child-b", "2026-01-01T00:00:05+00:00"),
                ],
            )
            connection = connect(database)
            try:
                child = connection.execute(
                    "SELECT relation_status FROM agents WHERE tree_id='tree-b' "
                    "AND agent_name='/root/child'"
                ).fetchone()
                self.assertEqual(child[0], "cross_tree_continuation")
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM dependency_edges WHERE edge_kind='spawn' "
                        "AND target_request_id='child-b'"
                    ).fetchone()[0],
                    0,
                )
                tree = connection.execute(
                    "SELECT capture_status, replayable, unresolved_agent_count "
                    "FROM trees WHERE tree_id='tree-b'"
                ).fetchone()
                self.assertEqual(tuple(tree), ("complete", 0, 1))
                diagnostic = connection.execute(
                    "SELECT detail_json FROM diagnostics "
                    "WHERE code='cross_tree_agent_continuation' "
                    "AND subject_id='child-b'"
                ).fetchone()
                self.assertIsNotNone(diagnostic)
                self.assertIs(json.loads(diagnostic[0])["executable_edge_created"], False)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
