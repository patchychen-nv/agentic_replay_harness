from __future__ import annotations

import json
import unittest
from pathlib import Path

from agentic_replay.replay import (
    ChainMismatchError,
    ReplayNode,
    ReplayOptions,
    _ResponseParser,
    _prepare_payload,
    _rewrite_live_chain_payload,
)


def _chain_node() -> ReplayNode:
    return ReplayNode(
        request_id="target",
        tree_id="tree",
        route="/v1/responses",
        session_id="session",
        agent_name="/root",
        send_ts=2.0,
        input_tokens=10,
        output_tokens=4,
        cached_tokens=5,
        source_path=Path("unused"),
        source_size_bytes=0,
        source_mtime_ns=0,
        line_offset=0,
        line_length=0,
        chain_parent_request_id="parent",
        chain_relation_mode="append",
        chain_confidence="high",
        chain_delta_start=1,
        chain_response_item_count=2,
        chain_tool_result_items=1,
    )


class ResponseItemCaptureTest(unittest.TestCase):
    def test_stream_done_item_wins_over_terminal_snapshot(self) -> None:
        parser = _ResponseParser("/v1/responses")
        done = {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "live-stream-id",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "live"}],
            },
        }
        terminal = {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "regenerated-terminal-id",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "terminal"}],
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 7},
                },
            },
        }
        parser.feed(
            (f"data: {json.dumps(done)}\n\n" f"data: {json.dumps(terminal)}\n\n").encode()
        )
        parser.finish()

        self.assertEqual(parser.output_items[0]["id"], "live-stream-id")
        self.assertEqual(parser.output_tokens, 3)
        self.assertEqual(parser.input_tokens, 10)
        self.assertEqual(parser.cached_tokens, 7)
        self.assertEqual(parser.result_status(200), "completed")

    def test_metrics_only_parser_counts_without_retaining_live_items(self) -> None:
        parser = _ResponseParser("/v1/responses", capture_output_items=False)
        done = {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "id": "must-not-be-retained",
                "call_id": "must-not-be-retained",
                "name": "live_tool",
                "arguments": "{}",
            },
        }

        parser.feed(f"data: {json.dumps(done)}\n\n".encode())
        parser.finish()

        self.assertEqual(parser.output_item_count, 1)
        self.assertEqual(parser.output_items, [])


class PayloadPreparationTest(unittest.TestCase):
    def test_default_preserves_source_output_controls(self) -> None:
        node = _chain_node()
        body = {"max_output_tokens": 99, "ignore_eos": False}

        prepared = _prepare_payload(
            body,
            node,
            ReplayOptions(base_url="http://example.invalid"),
        )

        self.assertEqual(prepared, {"max_output_tokens": 99, "ignore_eos": False})

    def test_output_limit_uses_effective_endpoint(self) -> None:
        node = _chain_node()
        node.route = "/v1/chat/completions"
        body = {"max_tokens": 99}
        options = ReplayOptions(
            base_url="http://example.invalid",
            endpoint="/v1/responses",
            fixed_output_length=True,
        )

        prepared = _prepare_payload(body, node, options)

        self.assertEqual(prepared["max_output_tokens"], node.output_tokens)
        self.assertNotIn("max_tokens", prepared)
        self.assertTrue(prepared["ignore_eos"])


class LiveChainRewriteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recorded_body = {
            "input": [
                {"type": "message", "role": "user", "content": "recorded-root"},
                {
                    "type": "reasoning",
                    "id": "recorded-reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "old"}],
                },
                {
                    "type": "custom_tool_call",
                    "id": "recorded-tool-item",
                    "call_id": "recorded-call",
                    "name": "exec",
                    "input": "old arguments",
                },
                {
                    "type": "custom_tool_call_output",
                    "id": "recorded-result",
                    "call_id": "recorded-call",
                    "output": [{"type": "input_text", "text": "recorded result"}],
                },
            ]
        }
        self.parent_sent_input = [
            {"type": "message", "role": "user", "content": "live-root"},
            {
                "type": "message",
                "id": "earlier-live-response",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "earlier"}],
            },
        ]
        self.parent_live_items = [
            {
                "type": "reasoning",
                "id": "live-reasoning",
                "status": "completed",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "new"}],
                "encrypted_content": None,
            },
            {
                "type": "message",
                "id": "extra-live-message",
                "role": "assistant",
                "status": "completed",
                "phase": None,
                "content": [
                    {
                        "type": "output_text",
                        "text": "extra",
                        "annotations": [],
                        "logprobs": None,
                    }
                ],
            },
            {
                "type": "custom_tool_call",
                "id": "live-tool-item",
                "call_id": "live-call",
                "name": "exec",
                "input": "new arguments",
                "namespace": "functions",
                "caller": None,
                "async_": None,
            },
        ]

    def test_uses_cumulative_live_prefix_and_rebinds_tool_result(self) -> None:
        rewritten, metrics = _rewrite_live_chain_payload(
            self.recorded_body,
            _chain_node(),
            self.parent_sent_input,
            self.parent_live_items,
        )
        items = rewritten["input"]

        self.assertEqual(items[:2], self.parent_sent_input)
        self.assertEqual([item["id"] for item in items[2:5]], [
            "live-reasoning",
            "extra-live-message",
            "live-tool-item",
        ])
        self.assertNotIn("status", items[2])
        self.assertNotIn("phase", items[3])
        self.assertNotIn("annotations", items[3]["content"][0])
        self.assertEqual(items[4]["namespace"], "functions")
        self.assertEqual(items[5]["call_id"], "live-call")
        self.assertEqual(items[5]["output"][0]["text"], "recorded result")
        self.assertEqual(metrics["chain_status"], "applied")
        self.assertEqual(metrics["chain_tool_results_rebound"], 1)
        self.assertEqual(metrics["chain_tool_argument_mismatches"], 1)
        self.assertEqual(metrics["chain_live_input_items"], 6)

    def test_tool_identity_mismatch_is_strict(self) -> None:
        self.parent_live_items[-1]["name"] = "different_tool"
        with self.assertRaises(ChainMismatchError) as raised:
            _rewrite_live_chain_payload(
                self.recorded_body,
                _chain_node(),
                self.parent_sent_input,
                self.parent_live_items,
            )
        self.assertIn("name differs", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
