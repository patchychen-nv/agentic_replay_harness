from __future__ import annotations

import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from agentic_replay.cli import build_parser
from agentic_replay.replay import ReplayNode, ReplayOptions, run_replay


class ConcurrencyValidationTest(unittest.TestCase):
    def test_cli_accepts_unlimited_and_positive_concurrency(self) -> None:
        parser = build_parser()
        for command in ("plan", "replay"):
            args = [command, "capture.sqlite3"]
            if command == "replay":
                args += ["--output", "results.jsonl"]
            for value in (0, 1, 128):
                with self.subTest(command=command, concurrency=value):
                    parsed = parser.parse_args(args + ["--concurrency", str(value)])
                    self.assertEqual(parsed.concurrency, value)
                    self.assertFalse(parsed.fixed_output_length)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(args + ["--concurrency", "-1"])

    def test_api_rejects_negative_concurrency(self) -> None:
        with self.assertRaisesRegex(ValueError, "concurrency must be non-negative"):
            ReplayOptions(base_url="http://unused.invalid", concurrency=-1)


class ConcurrencyIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_unlimited_and_bounded_http_concurrency_preserve_payloads(self) -> None:
        from aiohttp import web

        # More than both aiohttp's default connection limit (100) and the
        # harness default request limit (128), so either hidden cap fails.
        for concurrency, request_count in ((0, 132), (2, 6)):
            with self.subTest(concurrency=concurrency), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "requests.jsonl"
                locators: list[tuple[str, int, int]] = []
                recorded: dict[str, dict[str, object]] = {}
                with source.open("wb") as handle:
                    for index in range(request_count):
                        request_id = f"request-{index}"
                        body = {
                            "model": "recorded-model",
                            "input": [{"role": "user", "content": request_id}],
                            "max_output_tokens": 37,
                            "ignore_eos": False,
                        }
                        recorded[request_id] = body
                        data = (json.dumps({"trace_id": request_id, "body": body}) + "\n").encode()
                        locators.append((request_id, handle.tell(), len(data)))
                        handle.write(data)
                stat = source.stat()
                nodes = {
                    request_id: ReplayNode(
                        request_id=request_id,
                        tree_id="tree",
                        route="/v1/responses",
                        session_id=None,
                        agent_name="/root",
                        send_ts=0.0,
                        input_tokens=10,
                        output_tokens=7,
                        cached_tokens=None,
                        source_path=source,
                        source_size_bytes=stat.st_size,
                        source_mtime_ns=stat.st_mtime_ns,
                        line_offset=offset,
                        line_length=length,
                    )
                    for request_id, offset, length in locators
                }
                expected_active = concurrency or request_count
                all_slots_arrived = asyncio.Event()
                release = asyncio.Event()
                active = 0
                peak_active = 0
                received: dict[str, dict[str, object]] = {}

                async def handler(request: web.Request) -> web.Response:
                    nonlocal active, peak_active
                    source_id = request.headers["X-Source-Trace-ID"]
                    received[source_id] = await request.json()
                    active += 1
                    peak_active = max(peak_active, active)
                    if active == expected_active:
                        all_slots_arrived.set()
                    try:
                        await release.wait()
                        return web.json_response(
                            {"status": "completed", "usage": {"input_tokens": 10, "output_tokens": 7}}
                        )
                    finally:
                        active -= 1

                app = web.Application()
                app.router.add_post("/v1/responses", handler)
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, "127.0.0.1", 0, backlog=256)
                await site.start()
                port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
                options = ReplayOptions(
                    base_url=f"http://127.0.0.1:{port}",
                    schedule="asap",
                    concurrency=concurrency,
                    timeout_s=15,
                )
                task = asyncio.create_task(
                    run_replay(nodes, ["tree"], options, root / "results.jsonl")
                )
                try:
                    await asyncio.wait_for(all_slots_arrived.wait(), timeout=10)
                    # Requests remain blocked at the server during this check.
                    # A bounded client must not open additional request slots.
                    await asyncio.sleep(0.05)
                    self.assertEqual(active, expected_active)
                    release.set()
                    result = await asyncio.wait_for(task, timeout=10)
                finally:
                    release.set()
                    if not task.done():
                        task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await task
                    await runner.cleanup()

                self.assertEqual(peak_active, expected_active)
                self.assertEqual(result["concurrency"], concurrency)
                self.assertEqual(result["statuses"], {"completed": request_count})
                self.assertEqual(result["output_length_policy"], "source_payload")
                self.assertFalse(result["ignore_eos_requested"])
                self.assertEqual(received, recorded)
