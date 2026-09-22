from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_VERSION = 2


SCHEMA = r"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id INTEGER PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    absolute_path TEXT NOT NULL,
    kind TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS responses (
    trace_id TEXT PRIMARY KEY,
    response_status TEXT,
    finished_at TEXT,
    finish_ts REAL,
    disagg_request_id INTEGER,
    termination_json TEXT,
    response_kind TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_tokens INTEGER,
    finish_reason TEXT,
    output_types_json TEXT,
    source_id INTEGER,
    line_offset INTEGER,
    line_length INTEGER,
    parse_error TEXT,
    FOREIGN KEY(source_id) REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS response_tool_calls (
    call_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    item_type TEXT,
    name TEXT,
    arguments_json TEXT,
    is_control INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS response_tool_calls_trace_idx
ON response_tool_calls(trace_id, ordinal);

CREATE TABLE IF NOT EXISTS requests (
    trace_id TEXT PRIMARY KEY,
    route TEXT,
    request_status TEXT,
    recorded_at TEXT,
    send_ts REAL,
    server_arrival_time REAL,
    trace_session_id TEXT,
    session_id TEXT,
    thread_id TEXT,
    turn_id TEXT,
    root_turn_id TEXT,
    tree_id TEXT,
    agent_name TEXT,
    parent_agent_name TEXT,
    request_kind TEXT,
    context_window_id TEXT,
    window_id TEXT,
    turn_started_at_unix_ms INTEGER,
    model TEXT,
    stream INTEGER,
    prompt_cache_key TEXT,
    input_item_count INTEGER,
    source_id INTEGER NOT NULL,
    line_offset INTEGER NOT NULL,
    line_length INTEGER NOT NULL,
    sequence_index INTEGER,
    previous_request_id TEXT,
    replay_eligible INTEGER NOT NULL DEFAULT 0,
    eligibility_reason TEXT,
    FOREIGN KEY(source_id) REFERENCES sources(source_id)
);

CREATE INDEX IF NOT EXISTS requests_tree_thread_time_idx
ON requests(tree_id, thread_id, send_ts, trace_id);
CREATE INDEX IF NOT EXISTS requests_agent_idx
ON requests(tree_id, agent_name, send_ts);
CREATE INDEX IF NOT EXISTS requests_session_idx
ON requests(session_id, send_ts);

-- Compact, queryable fingerprints of the ordered request context.  The raw
-- payload remains in its source JSONL and is addressed by requests.source_id
-- plus line_offset/line_length; duplicating it here would make large indexes
-- prohibitively expensive.
CREATE TABLE IF NOT EXISTS request_input_items (
    request_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    item_type TEXT NOT NULL,
    item_id TEXT,
    call_id TEXT,
    role TEXT,
    name TEXT,
    canonical_sha256 TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    PRIMARY KEY(request_id, ordinal)
);

CREATE INDEX IF NOT EXISTS request_input_items_item_id_idx
ON request_input_items(item_id);
CREATE INDEX IF NOT EXISTS request_input_items_call_id_idx
ON request_input_items(call_id);

-- These rows are sourced from streaming response.output_item.done events,
-- not response.completed.response.output.  In the captured Responses traces
-- the latter can contain regenerated IDs that the client never propagates.
CREATE TABLE IF NOT EXISTS response_output_items (
    request_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    item_type TEXT NOT NULL,
    item_id TEXT,
    call_id TEXT,
    role TEXT,
    name TEXT,
    canonical_sha256 TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    PRIMARY KEY(request_id, ordinal)
);

CREATE INDEX IF NOT EXISTS response_output_items_item_id_idx
ON response_output_items(item_id);
CREATE INDEX IF NOT EXISTS response_output_items_call_id_idx
ON response_output_items(call_id);

-- Prompt/dataflow lineage is deliberately separate from dependency_edges.
-- The latter gates execution; this table says whose live response may replace
-- a recorded response slice when constructing the target prompt.
CREATE TABLE IF NOT EXISTS prompt_chains (
    target_request_id TEXT PRIMARY KEY,
    source_request_id TEXT NOT NULL,
    relation_mode TEXT NOT NULL,
    evidence_method TEXT NOT NULL,
    confidence TEXT NOT NULL,
    source_input_items INTEGER NOT NULL,
    target_input_items INTEGER NOT NULL,
    common_prefix_items INTEGER NOT NULL,
    delta_start INTEGER NOT NULL,
    response_item_count INTEGER NOT NULL,
    matched_response_items INTEGER NOT NULL,
    matched_item_ids INTEGER NOT NULL,
    matched_call_ids INTEGER NOT NULL,
    tool_result_items INTEGER NOT NULL,
    delay_s REAL,
    evidence_json TEXT
);

CREATE INDEX IF NOT EXISTS prompt_chains_source_idx
ON prompt_chains(source_request_id);

CREATE TABLE IF NOT EXISTS control_tool_results (
    call_id TEXT PRIMARY KEY,
    observed_in_request_id TEXT NOT NULL,
    observed_at TEXT,
    observed_ts REAL,
    output_json TEXT,
    output_truncated INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS agent_messages (
    message_id TEXT PRIMARY KEY,
    surfaced_in_request_id TEXT NOT NULL,
    surfaced_at TEXT,
    surfaced_ts REAL,
    author_agent_name TEXT,
    recipient_agent_name TEXT,
    message_type TEXT,
    content_sha256 TEXT,
    item_json TEXT
);

CREATE INDEX IF NOT EXISTS agent_messages_surface_idx
ON agent_messages(surfaced_in_request_id);

CREATE TABLE IF NOT EXISTS perf_metrics (
    disagg_request_id INTEGER PRIMARY KEY,
    status TEXT,
    arrival_time REAL,
    first_token_time REAL,
    ttft_s REAL,
    source_id INTEGER,
    line_offset INTEGER,
    line_length INTEGER,
    FOREIGN KEY(source_id) REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS agents (
    tree_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    agent_name TEXT,
    parent_agent_name TEXT,
    session_id TEXT,
    first_request_id TEXT,
    last_request_id TEXT,
    first_send_ts REAL,
    last_send_ts REAL,
    request_count INTEGER NOT NULL,
    relation_status TEXT,
    PRIMARY KEY(tree_id, thread_id)
);

CREATE INDEX IF NOT EXISTS agents_tree_name_idx
ON agents(tree_id, agent_name);

CREATE TABLE IF NOT EXISTS trees (
    tree_id TEXT PRIMARY KEY,
    session_id TEXT,
    root_request_id TEXT,
    started_at TEXT,
    start_ts REAL,
    end_ts REAL,
    request_count INTEGER NOT NULL,
    eligible_request_count INTEGER NOT NULL,
    agent_count INTEGER NOT NULL,
    unresolved_agent_count INTEGER NOT NULL,
    capture_status TEXT NOT NULL,
    replayable INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS dependency_edges (
    source_request_id TEXT NOT NULL,
    target_request_id TEXT NOT NULL,
    edge_kind TEXT NOT NULL,
    delay_s REAL,
    evidence_method TEXT NOT NULL,
    confidence TEXT NOT NULL,
    evidence_json TEXT,
    PRIMARY KEY(source_request_id, target_request_id, edge_kind)
);

CREATE INDEX IF NOT EXISTS dependency_edges_target_idx
ON dependency_edges(target_request_id);

CREATE TABLE IF NOT EXISTS external_gaps (
    source_request_id TEXT NOT NULL,
    target_request_id TEXT NOT NULL,
    duration_s REAL,
    tool_call_count INTEGER NOT NULL,
    tool_names_json TEXT NOT NULL,
    attribution TEXT NOT NULL,
    PRIMARY KEY(source_request_id, target_request_id)
);

CREATE TABLE IF NOT EXISTS diagnostics (
    diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    severity TEXT NOT NULL,
    code TEXT NOT NULL,
    subject_id TEXT,
    detail_json TEXT
);
"""


def connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    if not readonly:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = MEMORY")
    return connection


def initialize(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    connection.commit()
