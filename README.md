# Agentic Replay Harness

Recovers the agent request dependency DAG (replay trees) from TensorRT-LLM
`request_trace` logs and replays the same workload, dependency by dependency, against
any OpenAI-compatible endpoint (normally `/v1/responses`).

Default policy: every request is sent with its recorded input, including conversation
history and tool results. Live output is measured but never fed into later requests.

## Install

Python >= 3.10; the only dependency is `aiohttp`.

```bash
cd /path/to/agentic_replay_harness
python3 -m pip install -e .
python3 -m unittest discover -v
```

## Input

One completed, no-longer-written TRT-LLM attempt directory:

```text
attempt-001/
├── request_trace/<UTC-hour>/requests-*.jsonl     # required
├── request_trace/<UTC-hour>/responses-*.jsonl    # required
└── perf_metrics/perf_metrics-disagg-*.jsonl      # optional, provides recorded TTFT
```

SQLite does not copy payloads. It stores each payload's source path, byte offset,
length, file size and mtime. Keep the source directory readable and unchanged after
build; if it must move, pass the new location with `--source-root`.

## Build replay trees

```bash
ATTEMPT_DIR=/abs/path/to/attempt-001
ARTIFACT_DIR=artifacts/<capture_tag>
DATABASE=$ARTIFACT_DIR/replay.sqlite3

python3 -m agentic_replay build "$ATTEMPT_DIR" \
  --output-dir "$ARTIFACT_DIR" --progress-every 10000 \
  | tee "$ARTIFACT_DIR/build-summary.json"

python3 -m agentic_replay inspect "$DATABASE" | tee "$ARTIFACT_DIR/inspect.json"
```

Outputs: `replay.sqlite3`, `manifest.json`, `replay_trees.jsonl`,
`external_gaps.jsonl`, `diagnostics.jsonl`. An existing database is never
overwritten; use `--rebuild` to recreate it.

List small replayable trees for a canary:

```bash
sqlite3 -header -column "$DATABASE" '
  SELECT tree_id, eligible_request_count AS requests, agent_count AS agents
  FROM trees WHERE replayable = 1
  ORDER BY eligible_request_count, start_ts LIMIT 20;'
```

## Plan

`plan` is read-only and sends no HTTP. It selects requests, checks source file
size/mtime against the index, requires positive recorded OSL, verifies the DAG is
acyclic, and applies two local token guards (not sent to the server, nothing is
truncated).

```bash
MAX_INPUT_TOKENS=1048575
MAX_SEQ_LEN=1048576

python3 -m agentic_replay plan "$DATABASE" \
  --payload-mode recorded --schedule causal --root-schedule recorded --speedup 1 \
  --max-input-tokens "$MAX_INPUT_TOKENS" --max-seq-len "$MAX_SEQ_LEN" \
  | tee "$ARTIFACT_DIR/plan.json"
```

Without `--tree-id` / `--limit-trees`, every tree with `replayable = 1` is selected.
`plan` and `replay` must use identical selection and scheduling flags.

## Replay

```bash
REPLAY_BASE_URL=http://host:port
REPLAY_X_API_KEY='<x-api-key>'   # drop --x-api-key if unused; bearer auth: --api-key / OPENAI_API_KEY
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)

# Canary: one small tree first
python3 -m agentic_replay replay "$DATABASE" \
  --base-url "$REPLAY_BASE_URL" --x-api-key "$REPLAY_X_API_KEY" \
  --tree-id "<tree_id>" \
  --payload-mode recorded --schedule causal --root-schedule recorded --speedup 1 \
  --concurrency 4 --timeout 3600 \
  --max-input-tokens "$MAX_INPUT_TOKENS" --max-seq-len "$MAX_SEQ_LEN" \
  --output "$ARTIFACT_DIR/replay-results-canary-$RUN_TAG.jsonl"

# Full run: drop --tree-id; run inside tmux
python3 -m agentic_replay replay "$DATABASE" \
  --base-url "$REPLAY_BASE_URL" --x-api-key "$REPLAY_X_API_KEY" \
  --payload-mode recorded --schedule causal --root-schedule recorded --speedup 1 \
  --concurrency 128 --timeout 3600 \
  --max-input-tokens "$MAX_INPUT_TOKENS" --max-seq-len "$MAX_SEQ_LEN" \
  --output "$ARTIFACT_DIR/replay-results-full-$RUN_TAG.jsonl" \
  2>&1 | tee "$ARTIFACT_DIR/replay-console-$RUN_TAG.log"
```

Result rows are flushed in completion order; `<output>.summary.json` is written only
when the whole run finishes. Runs cannot be resumed; restart with a new `--output`.

Common flags:

| Flag | Meaning |
|---|---|
| `--schedule causal` | Wait for the replayed predecessor to finish, then add the recorded edge delay (default) |
| `--schedule fixed` | Recorded global arrival timeline plus dependency gating |
| `--schedule asap` | Dependencies only; all recorded waits removed |
| `--root-schedule recorded\|simultaneous` | Keep recorded offsets between tree roots, or release them together (causal only) |
| `--speedup N` | Divide recorded waits and arrival offsets by N |
| `--concurrency N` | Limit concurrent HTTP requests (default 128); `0` removes both client request and connection limits, while retaining dependency and timing gates |
| `--fixed-output-length` | Set `max_output_tokens` to recorded OSL and add `ignore_eos`; by default source output controls are untouched |
| `--payload-mode live-chain` | Experimental: replace the parent-response slice in the next request with live output |
| `--allow-partial` | Also replay eligible requests from incomplete trees; changes the workload |

## First principles of tree construction

**1. Identity comes from client metadata; nothing is guessed.** Each request's
`client_metadata` decides its tree, agent and thread:

```text
client_metadata
├─ root_turn_id ──▶ tree_id        (falls back to turn_id)
├─ session_id
├─ thread_id ─┬──▶ agent = (tree_id, thread_id)
├─ turn_id ───┘     sequence scope = (tree_id, thread_id, turn_id)
└─ agent_name  "/root/worker-1" ──▶ parent path "/root"
```

**2. An edge exists only when there is evidence.** Every edge stores
`evidence_method`, `confidence` and an evidence JSON.

```text
tree_id = root_turn_id
│
├─ agent /root  (thread T0)
│   r1 ─seq─▶ r2 ─────────────────seq──────────────▶ r5 ─seq─▶ r6
│              │ output has spawn_agent("worker-1")      ▲ input has wait_agent result
│              │ spawn                          join     │
│              ▼                                         │
└─ agent /root/worker-1  (thread T1)                     │
    r3 ─seq─▶ r4 ──────────── agent_final_join ──────────┘
```

| Edge | Evidence |
|---|---|
| `thread_sequence` | Adjacent requests in the same `(tree, thread, turn)`, and the previous response finished before the next request was sent. Overlapping requests produce a diagnostic, not a forced serial edge |
| `spawn` | `spawn_agent(task_name)` tool call in the parent's response ──▶ the child agent's first request |
| `join` / `interrupt_join` | `wait_agent` / `interrupt_agent` result appears in the parent's next request input |
| `followup_resume` | `followup_task` result ──▶ the resumed agent's next request |
| `agent_final_join` | Structured agent message of type `FINAL_ANSWER`: the author's last completed response before the message was sent ──▶ the request whose input surfaces the message |
| `agent_message_delivery` | Same, for message types `MESSAGE` / `NEW_TASK` |

Agent identity evidence is separate from an executable dependency. If no
same-tree spawn response is available, inference also checks the first captured
parent request's recorded `body.input` for a matching `spawn_agent` call **and**
result. `spawn_history_only` proves the logical parent/child path, but does not
create a `spawn` edge: that prompt may have arrived after the child began, and
the original spawn response timestamp is unavailable. A matching earlier
segment of the same `(session_id, thread_id, agent_name)` is marked
`cross_tree_continuation` when `root_turn_id` splits a thread; it is not treated
as a new spawn. Both statuses remain excluded from strict default replay until
an executable cross-tree/untimed scheduling policy is chosen. Their evidence
is recorded in `diagnostics`.

**3. Waits are aggregate external gaps, not per-tool latency.** The logs contain no
tool start/end events, so the only recoverable quantity is:

```text
edge_delay = next_request.recorded_at − previous_response.finished_at
           = tool execution + client scheduling + serialization + transport
```

`causal` scheduling reproduces this delay at replay time. Several tool calls inside
one gap cannot be split apart. `recorded_at` is the server-side request trace's
arrival timestamp, not a client-side send timestamp. If the previous response
is missing, the difference between two request arrivals also includes the
unknown LLM duration and cannot be attributed to tool execution.

**4. A tree is replayable iff its graph is closed.**

```text
replayable  ⇔  every request is eligible (paired completed response, has thread_id)
            ∧  every agent has an executable same-tree parent relation
```

`capture_status` separately reports whether all, some, or none of the requests
are individually eligible (`complete`, `partial`, `unreplayable`). A `complete`
tree can still be `replayable=0` because its agent relation is logical-only.
Strict default replay selects only `replayable=1` trees. This means the graph
is closed, not that every edge is high confidence; filter on each edge's
`confidence` for sensitivity analysis.
