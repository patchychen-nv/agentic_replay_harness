from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from agentic_replay.cli import _positive_float, build_parser
from agentic_replay.database import connect
from agentic_replay.infer import infer_graph
from agentic_replay.ingest import build_index, enrich_lineage_items
from agentic_replay.replay import (
    ReplayOptions,
    _ResponseParser,
    load_replay_nodes,
    replay_plan,
    run_replay,
)
from agentic_replay.report import export_graph, inspect_index


def _metadata(
    *, thread_id: str, root_turn_id: str, agent_name: str, session_id: str = "session-1"
) -> dict[str, object]:
    nested = {
        "session_id": session_id,
        "thread_id": thread_id,
        "turn_id": f"turn-{thread_id}",
        "root_turn_id": root_turn_id,
        "agent_name": agent_name,
        "request_kind": "turn",
        "turn_started_at_unix_ms": 1_000,
    }
    return {
        "session_id": session_id,
        "thread_id": thread_id,
        "turn_id": nested["turn_id"],
        "root_turn_id": root_turn_id,
        "x-codex-turn-metadata": json.dumps(nested),
    }


def _request(
    trace_id: str,
    timestamp: str,
    *,
    thread_id: str,
    agent_name: str,
    input_items: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "event": "request",
        "trace_id": trace_id,
        "route": "/v1/responses",
        "session": "session-1",
        "status": "accepted",
        "recorded_at": timestamp,
        "server_arrival_time": 10.0,
        "headers": [],
        "body": {
            "model": "recorded-model",
            "stream": True,
            "input": input_items
            or [{"type": "message", "role": "user", "content": "fixture"}],
            "client_metadata": _metadata(
                thread_id=thread_id,
                root_turn_id="root-turn-1",
                agent_name=agent_name,
            ),
        },
    }


def _sse_event(value: dict[str, object]) -> str:
    return f"event: {value.get('type', 'message')}\ndata: {json.dumps(value)}\n\n"


def _response(
    trace_id: str,
    timestamp: str,
    *,
    output_tokens: int,
    tool: tuple[str, str, dict[str, object]] | None = None,
    disagg_id: int,
) -> dict[str, object]:
    body = ""
    if tool:
        call_id, name, arguments = tool
        body += _sse_event(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        )
    body += _sse_event(
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": output_tokens,
                    "input_tokens_details": {"cached_tokens": 5},
                },
                "output": [],
            },
        }
    )
    return {
        "event": "response",
        "trace_id": trace_id,
        "session": "session-1",
        "status": "completed",
        "finished_at": timestamp,
        "disagg_request_id": disagg_id,
        "response": {"kind": "sse_text", "body": body},
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class HarnessFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.attempt = self.root / "attempt-001"
        trace_hour = self.attempt / "request_trace" / "2026-01-01T00"
        spawn_result = {
            "type": "function_call_output",
            "call_id": "spawn-call",
            "output": json.dumps(
                {"agent_id": "agent-1", "canonical_task_name": "/root/child"}
            ),
        }
        wait_result = {
            "type": "function_call_output",
            "call_id": "wait-call",
            "output": json.dumps(
                {
                    "agent_updates": [
                        {"agent_name": "/root/child", "status": "completed"}
                    ]
                }
            ),
        }
        final_message = {
            "type": "agent_message",
            "id": "message-1",
            "author": "/root/child",
            "recipient": "/root",
            "content": "Message Type: FINAL_ANSWER\nTask complete.",
        }
        requests = [
            _request(
                "root-1",
                "2026-01-01T00:00:01+00:00",
                thread_id="thread-root",
                agent_name="/root",
            ),
            _request(
                "root-2",
                "2026-01-01T00:00:02.100000+00:00",
                thread_id="thread-root",
                agent_name="/root",
                input_items=[spawn_result],
            ),
            _request(
                "child-1",
                "2026-01-01T00:00:02.200000+00:00",
                thread_id="thread-child",
                agent_name="/root/child",
            ),
            _request(
                "root-3",
                "2026-01-01T00:00:03.200000+00:00",
                thread_id="thread-root",
                agent_name="/root",
                input_items=[spawn_result, wait_result, final_message],
            ),
        ]
        responses = [
            _response(
                "root-1",
                "2026-01-01T00:00:02+00:00",
                output_tokens=2,
                tool=("spawn-call", "spawn_agent", {"task_name": "child"}),
                disagg_id=1,
            ),
            _response(
                "root-2",
                "2026-01-01T00:00:02.500000+00:00",
                output_tokens=2,
                tool=("wait-call", "wait_agent", {"timeout_ms": 1000}),
                disagg_id=2,
            ),
            _response(
                "child-1",
                "2026-01-01T00:00:03+00:00",
                output_tokens=3,
                disagg_id=3,
            ),
            _response(
                "root-3",
                "2026-01-01T00:00:04+00:00",
                output_tokens=4,
                disagg_id=4,
            ),
        ]
        _write_jsonl(trace_hour / "requests-1.jsonl", requests)
        _write_jsonl(trace_hour / "responses-1.jsonl", responses)
        self.database = self.root / "out" / "replay.sqlite3"
        build_index(
            self.attempt,
            self.database,
            include_perf=False,
            progress_every=0,
        )
        infer_graph(self.database)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_graph_and_evidence(self) -> None:
        summary = inspect_index(self.database)
        self.assertEqual(summary["requests"], 4)
        self.assertEqual(summary["paired_requests"], 4)
        self.assertEqual(summary["trees"], 1)
        self.assertEqual(summary["identity_counts"]["sessions"], 1)
        self.assertEqual(summary["identity_counts"]["threads"], 2)
        self.assertEqual(summary["identity_counts"]["root_turns"], 1)
        self.assertEqual(summary["subagents"], 1)
        self.assertEqual(summary["replayable_trees"], 1)
        self.assertEqual(summary["dependency_edges"]["thread_sequence"], 2)
        self.assertEqual(summary["dependency_edges"]["spawn"], 1)
        self.assertEqual(summary["dependency_edges"]["join"], 1)
        self.assertEqual(summary["dependency_edges"]["agent_final_join"], 1)
        self.assertEqual(
            summary["external_gap_attribution"]["single_tool_plus_client_overhead"],
            2,
        )

    def test_export_and_plan(self) -> None:
        exported = export_graph(self.database, self.root / "export")
        manifest = json.loads(Path(exported["manifest"]).read_text())
        self.assertEqual(manifest["schema"], "agentic_replay_manifest/v1")
        line = Path(exported["trees"]).read_text().splitlines()[0]
        tree = json.loads(line)
        self.assertEqual(len(tree["requests"]), 4)
        root3 = next(item for item in tree["requests"] if item["request_id"] == "root-3")
        self.assertEqual(
            {dep["kind"] for dep in root3["depends_on"]},
            {"thread_sequence", "join", "agent_final_join"},
        )

        options = ReplayOptions(base_url="http://127.0.0.1:1")
        nodes, tree_ids = load_replay_nodes(
            self.database, limit_trees=1, options=options
        )
        plan = replay_plan(nodes, tree_ids)
        self.assertEqual(plan["requests"], 4)
        self.assertEqual(plan["output_token_budget"], 11)

        relocated = ReplayOptions(
            base_url="http://127.0.0.1:1", source_root=self.root / "missing-capture"
        )
        with self.assertRaises(FileNotFoundError):
            load_replay_nodes(self.database, limit_trees=1, options=relocated)

    def test_partial_missing_usage_requires_explicit_fallback(self) -> None:
        connection = connect(self.database)
        connection.execute(
            "UPDATE responses SET output_tokens=NULL WHERE trace_id='root-3'"
        )
        connection.execute(
            "UPDATE requests SET replay_eligible=0, eligibility_reason='missing_output_tokens' "
            "WHERE trace_id='root-3'"
        )
        connection.commit()
        connection.close()
        infer_graph(self.database)

        options = ReplayOptions(
            base_url="http://127.0.0.1:1",
            allow_partial=True,
            fallback_output_tokens=7,
        )
        nodes, _ = load_replay_nodes(self.database, limit_trees=1, options=options)
        self.assertEqual(nodes["root-3"].output_tokens, 7)


class LineageInferenceTest(unittest.TestCase):
    def test_payload_frontier_proves_chain_parent_and_can_be_backfilled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "attempt-001"
            trace_hour = attempt / "request_trace" / "2026-01-01T00"
            base = {
                "type": "message",
                "id": "user-message-1",
                "role": "user",
                "content": "fixture",
            }
            call = {
                "type": "function_call",
                "call_id": "call-1",
                "name": "ordinary_tool",
                "arguments": json.dumps({"value": 1}),
            }
            result = {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "tool result",
            }
            requests = [
                _request(
                    "request-1",
                    "2026-01-01T00:00:01+00:00",
                    thread_id="thread-root",
                    agent_name="/root",
                    input_items=[base],
                ),
                _request(
                    "request-2",
                    "2026-01-01T00:00:02.100000+00:00",
                    thread_id="thread-root",
                    agent_name="/root",
                    input_items=[base, call, result],
                ),
            ]
            responses = [
                _response(
                    "request-1",
                    "2026-01-01T00:00:02+00:00",
                    output_tokens=5,
                    tool=("call-1", "ordinary_tool", {"value": 1}),
                    disagg_id=1,
                ),
                _response(
                    "request-2",
                    "2026-01-01T00:00:03+00:00",
                    output_tokens=3,
                    disagg_id=2,
                ),
            ]
            _write_jsonl(trace_hour / "requests-1.jsonl", requests)
            _write_jsonl(trace_hour / "responses-1.jsonl", responses)
            database = root / "replay.sqlite3"
            build_index(attempt, database, include_perf=False, progress_every=0)
            summary = infer_graph(database)
            self.assertEqual(summary["prompt_chains"], 1)

            connection = connect(database)
            chain = connection.execute(
                "SELECT * FROM prompt_chains WHERE target_request_id='request-2'"
            ).fetchone()
            self.assertIsNotNone(chain)
            assert chain is not None
            self.assertEqual(chain["source_request_id"], "request-1")
            self.assertEqual(chain["relation_mode"], "append")
            self.assertEqual(chain["confidence"], "high")
            self.assertEqual(chain["delta_start"], 1)
            self.assertEqual(chain["response_item_count"], 1)
            self.assertEqual(chain["matched_call_ids"], 1)
            self.assertEqual(chain["tool_result_items"], 1)

            # Simulate a v1 database whose raw source locators are intact but
            # whose lineage index has not yet been populated.
            connection.execute("DELETE FROM prompt_chains")
            connection.execute("DELETE FROM request_input_items")
            connection.execute("DELETE FROM response_output_items")
            connection.commit()
            connection.close()
            enrichment = enrich_lineage_items(
                database, tree_ids=["root-turn-1"]
            )
            self.assertEqual(enrichment["requests"], 2)
            self.assertEqual(enrichment["response_output_items"], 1)
            self.assertEqual(infer_graph(database)["prompt_chains"], 1)


class CliValidationTest(unittest.TestCase):
    def test_positive_float_rejects_non_finite_values(self) -> None:
        for value in ("nan", "inf", "-inf", "0", "-1"):
            with self.subTest(value=value), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                _positive_float(value)

    def test_output_length_controls_are_opt_in(self) -> None:
        parser = build_parser()
        default = parser.parse_args(["plan", "capture.sqlite3"])
        fixed = parser.parse_args(
            ["plan", "capture.sqlite3", "--fixed-output-length"]
        )
        compatibility = parser.parse_args(
            ["plan", "capture.sqlite3", "--natural-eos"]
        )

        self.assertFalse(default.fixed_output_length)
        self.assertTrue(fixed.fixed_output_length)
        self.assertFalse(compatibility.fixed_output_length)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "plan",
                        "capture.sqlite3",
                        "--fixed-output-length",
                        "--natural-eos",
                    ]
                )


class ResponseParserTest(unittest.TestCase):
    def test_responses_terminal_statuses_and_json(self) -> None:
        failed = _ResponseParser("/v1/responses")
        failed.feed(
            b'data: {"type":"response.failed","response":'
            b'{"status":"failed","error":{"message":"boom"}}}\n\n'
        )
        failed.finish()
        self.assertEqual(failed.result_status(200), "response_failed")

        missing = _ResponseParser("/v1/responses")
        missing.feed(b'data: {"type":"response.output_text.delta"}\n\n')
        missing.finish()
        self.assertEqual(missing.result_status(200), "missing_terminal_event")

        complete = _ResponseParser("/v1/responses")
        complete.feed(
            b'{"object":"response","status":"completed",'
            b'"usage":{"output_tokens":17}}'
        )
        complete.finish()
        self.assertEqual(complete.result_status(200), "completed")
        self.assertEqual(complete.output_tokens, 17)


class ReplayIntegration(HarnessFixture, unittest.IsolatedAsyncioTestCase):
    async def test_recorded_mode_never_propagates_live_output(self) -> None:
        """The default replay path is a recorded-payload trajectory.

        A response deliberately returns a tool call that is unrelated to the
        captured conversation.  Every later request must nevertheless retain
        its own source ``input`` (including captured tool results) verbatim.
        """
        from aiohttp import web

        recorded_bodies: dict[str, dict[str, object]] = {}
        request_trace = self.attempt / "request_trace"
        for request_file in sorted(request_trace.rglob("requests-*.jsonl")):
            for raw_line in request_file.read_text(encoding="utf-8").splitlines():
                record = json.loads(raw_line)
                recorded_bodies[record["trace_id"]] = record["body"]

        received: dict[str, dict[str, object]] = {}
        request_paths: list[str] = []

        async def handler(request: web.Request) -> web.Response:
            self.assertEqual(request.headers.get("x-api-key"), "test-replay-key")
            source_id = request.headers["X-Source-Trace-ID"]
            payload = await request.json()
            received[source_id] = payload
            request_paths.append(request.path)

            live_item = {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": f"live-item-{source_id}",
                    "call_id": f"live-call-{source_id}",
                    "name": "unrelated_live_tool",
                    "arguments": "{}",
                },
            }
            terminal = {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": int(nodes[source_id].output_tokens or 0) + 1,
                    },
                },
            }
            return web.Response(
                text=_sse_event(live_item) + _sse_event(terminal),
                content_type="text/event-stream",
            )

        app = web.Application()
        app.router.add_post("/v1/responses", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        sockets = site._server.sockets  # type: ignore[union-attr]
        port = sockets[0].getsockname()[1]
        output_path = self.root / "replay-results-recorded.jsonl"
        try:
            options = ReplayOptions(
                base_url=f"http://127.0.0.1:{port}",
                x_api_key="test-replay-key",
                schedule="asap",
                concurrency=4,
            )
            self.assertEqual(options.payload_mode, "recorded")
            nodes, tree_ids = load_replay_nodes(
                self.database, limit_trees=1, options=options
            )
            result = await run_replay(nodes, tree_ids, options, output_path)
        finally:
            await runner.cleanup()

        self.assertEqual(result["payload_mode"], "recorded")
        self.assertEqual(result["prompt_source"], "recorded")
        self.assertEqual(result["live_output_policy"], "metrics_only")
        self.assertEqual(result["output_length_policy"], "source_payload")
        self.assertFalse(result["fixed_output_length_requested"])
        self.assertFalse(result["ignore_eos_requested"])
        self.assertEqual(result["requests_with_ignore_eos"], 0)
        self.assertEqual(result["requests_with_output_token_limit"], 0)
        self.assertEqual(result["output_length_fidelity"], {"mismatched": 4})
        self.assertEqual(set(request_paths), {"/v1/responses"})
        self.assertEqual(set(received), set(recorded_bodies))
        for source_id, payload in received.items():
            with self.subTest(source_id=source_id):
                self.assertEqual(payload["input"], recorded_bodies[source_id]["input"])
                self.assertNotIn("ignore_eos", payload)
                self.assertNotIn("max_output_tokens", payload)
                self.assertNotIn("max_tokens", payload)
                self.assertNotIn(
                    "live-call-", json.dumps(payload["input"], sort_keys=True)
                )

        # Captured tool results remain bound to captured call IDs; live call
        # IDs are metrics only and are never rebound into a later payload.
        self.assertEqual(received["root-2"]["input"][0]["call_id"], "spawn-call")
        root3_call_ids = [
            item.get("call_id")
            for item in received["root-3"]["input"]
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        self.assertEqual(root3_call_ids, ["spawn-call", "wait-call"])

        replay_lines = [
            json.loads(line) for line in output_path.read_text().splitlines()
        ]
        for line in replay_lines:
            source_input = recorded_bodies[line["source_request_id"]]["input"]
            expected_tool_results = sum(
                1
                for item in source_input
                if isinstance(item, dict)
                and str(item.get("type", "")).endswith("_call_output")
            )
            with self.subTest(result_source_id=line["source_request_id"]):
                self.assertEqual(line["prompt_source"], "recorded")
                self.assertEqual(line["live_output_policy"], "metrics_only")
                self.assertFalse(line["requested_ignore_eos"])
                self.assertIsNone(line["requested_output_token_limit"])
                self.assertEqual(line["output_token_delta"], 1)
                self.assertFalse(line["output_length_match"])
                self.assertEqual(line["recorded_input_items"], len(source_input))
                self.assertEqual(line["sent_input_items"], len(source_input))
                self.assertEqual(
                    line["recorded_tool_result_items"], expected_tool_results
                )
                self.assertEqual(
                    line["sent_tool_result_items"], expected_tool_results
                )
                self.assertEqual(
                    line["recorded_input_sha256"], line["sent_input_sha256"]
                )
                self.assertEqual(line["live_output_items"], 1)
                for semantic_field in (
                    "live_output_item_types",
                    "live_output_item_ids",
                    "live_output_call_ids",
                    "live_output_sha256",
                ):
                    self.assertNotIn(semantic_field, line)

    async def test_mock_http_replay_obeys_join(self) -> None:
        from aiohttp import web

        arrivals: list[str] = []

        async def handler(request: web.Request) -> web.Response:
            source_id = request.headers["X-Source-Trace-ID"]
            payload = await request.json()
            self.assertTrue(payload["ignore_eos"])
            arrivals.append(source_id)
            if source_id == "child-1":
                await __import__("asyncio").sleep(0.05)
            event = {
                "type": "response.completed",
                "response": {"usage": {"output_tokens": payload["max_output_tokens"]}},
            }
            return web.Response(
                text=f"data: {json.dumps(event)}\n\n",
                content_type="text/event-stream",
            )

        app = web.Application()
        app.router.add_post("/v1/responses", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        sockets = site._server.sockets  # type: ignore[union-attr]
        port = sockets[0].getsockname()[1]
        try:
            options = ReplayOptions(
                base_url=f"http://127.0.0.1:{port}",
                schedule="asap",
                concurrency=4,
                fixed_output_length=True,
            )
            nodes, tree_ids = load_replay_nodes(
                self.database, limit_trees=1, options=options
            )
            result = await run_replay(
                nodes, tree_ids, options, self.root / "replay-results.jsonl"
            )
            with self.assertRaises(FileExistsError):
                await run_replay(
                    nodes, tree_ids, options, self.root / "replay-results.jsonl"
                )
        finally:
            await runner.cleanup()
        self.assertEqual(result["statuses"], {"completed": 4})
        self.assertEqual(result["output_length_fidelity"], {"matched": 4})
        self.assertTrue(result["fixed_output_length_requested"])
        self.assertTrue(result["ignore_eos_requested"])
        self.assertEqual(result["requests_with_ignore_eos"], 4)
        self.assertEqual(result["requests_with_output_token_limit"], 4)
        self.assertEqual(result["preloaded_root_requests"], 1)
        self.assertGreater(result["preloaded_root_payload_bytes"], 0)
        replay_lines = [
            json.loads(line)
            for line in (self.root / "replay-results.jsonl").read_text().splitlines()
        ]
        self.assertTrue(all(line["output_length_match"] for line in replay_lines))
        self.assertEqual(arrivals[0], "root-1")
        self.assertEqual(arrivals[-1], "root-3")

    async def test_output_length_fidelity_is_independent_of_terminal_status(
        self,
    ) -> None:
        from aiohttp import web

        async def handler(request: web.Request) -> web.Response:
            source_id = request.headers["X-Source-Trace-ID"]
            payload = await request.json()
            output_tokens = payload["max_output_tokens"]

            if source_id == "root-1":
                event = {
                    "type": "response.incomplete",
                    "response": {
                        "status": "incomplete",
                        "usage": {"output_tokens": output_tokens},
                        "incomplete_details": {"reason": "max_output_tokens"},
                    },
                }
            elif source_id == "root-2":
                event = {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {"output_tokens": output_tokens - 1},
                    },
                }
            elif source_id == "child-1":
                event = {
                    "type": "response.completed",
                    "response": {"status": "completed"},
                }
            else:
                event = {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {"output_tokens": output_tokens},
                    },
                }

            return web.Response(
                text=f"data: {json.dumps(event)}\n\n",
                content_type="text/event-stream",
            )

        app = web.Application()
        app.router.add_post("/v1/responses", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        sockets = site._server.sockets  # type: ignore[union-attr]
        port = sockets[0].getsockname()[1]
        output_path = self.root / "replay-results-fidelity.jsonl"
        try:
            options = ReplayOptions(
                base_url=f"http://127.0.0.1:{port}",
                schedule="asap",
                concurrency=4,
                fixed_output_length=True,
            )
            nodes, tree_ids = load_replay_nodes(
                self.database, limit_trees=1, options=options
            )
            result = await run_replay(nodes, tree_ids, options, output_path)
        finally:
            await runner.cleanup()

        self.assertEqual(
            result["statuses"], {"response_incomplete": 1, "completed": 3}
        )
        self.assertEqual(
            result["output_length_fidelity"],
            {"matched": 2, "mismatched": 1, "missing_usage": 1},
        )
        self.assertNotIn("not_completed", result["output_length_fidelity"])

        replay_lines = {
            line["source_request_id"]: line
            for line in (
                json.loads(raw_line) for raw_line in output_path.read_text().splitlines()
            )
        }
        incomplete = replay_lines["root-1"]
        self.assertEqual(incomplete["status"], "response_incomplete")
        self.assertTrue(incomplete["output_length_match"])
        self.assertEqual(incomplete["output_token_delta"], 0)

        mismatched = replay_lines["root-2"]
        self.assertFalse(mismatched["output_length_match"])
        self.assertEqual(mismatched["output_token_delta"], -1)

        missing_usage = replay_lines["child-1"]
        self.assertIsNone(missing_usage["output_length_match"])
        self.assertNotIn("output_token_delta", missing_usage)

        self.assertTrue(replay_lines["root-3"]["output_length_match"])


if __name__ == "__main__":
    unittest.main()
