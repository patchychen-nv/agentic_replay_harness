from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from .database import connect
from .infer import infer_graph
from .ingest import build_index, enrich_agent_messages, enrich_lineage_items
from .replay import ReplayOptions, load_replay_nodes, replay_plan, run_replay
from .report import export_graph, inspect_index
from .util import json_dumps


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _print(value: Any) -> None:
    print(json_dumps(value, pretty=True))


def _compact_plan(plan: dict[str, Any], *, max_tree_ids: int = 20) -> dict[str, Any]:
    """Keep human-facing CLI output bounded for full-capture plans."""
    tree_ids = plan.get("tree_ids")
    if not isinstance(tree_ids, list) or len(tree_ids) <= max_tree_ids:
        return plan
    compact = dict(plan)
    compact["tree_ids"] = tree_ids[:max_tree_ids]
    compact["tree_ids_truncated"] = True
    compact["tree_ids_omitted"] = len(tree_ids) - max_tree_ids
    return compact


def _checkpoint(database: Path) -> None:
    connection = connect(database)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()


def _cmd_build(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    database = args.database.absolute() if args.database else output_dir / "replay.sqlite3"
    ingest_summary = build_index(
        args.attempt_dir,
        database,
        include_perf=not args.no_perf,
        progress_every=args.progress_every,
        rebuild=args.rebuild,
    )
    graph_summary = infer_graph(database)
    exported = None if args.no_export else export_graph(database, output_dir)
    _checkpoint(database)
    _print(
        {
            "database": str(database),
            "ingest": ingest_summary,
            "graph": graph_summary,
            "inspection": inspect_index(database),
            "exported": exported,
        }
    )


def _cmd_infer(args: argparse.Namespace) -> None:
    result = infer_graph(args.database)
    _checkpoint(args.database)
    _print(result)


def _cmd_enrich_messages(args: argparse.Namespace) -> None:
    enrichment = enrich_agent_messages(
        args.database, progress_every=args.progress_every
    )
    graph = infer_graph(args.database)
    _checkpoint(args.database)
    _print({"agent_messages": enrichment, "graph": graph})


def _cmd_enrich_lineage(args: argparse.Namespace) -> None:
    enrichment = enrich_lineage_items(args.database, tree_ids=args.tree_id)
    graph = infer_graph(args.database)
    _checkpoint(args.database)
    _print({"lineage_items": enrichment, "graph": graph})


def _cmd_inspect(args: argparse.Namespace) -> None:
    _print(inspect_index(args.database))


def _cmd_export(args: argparse.Namespace) -> None:
    _print(export_graph(args.database, args.output_dir))


def _replay_options(args: argparse.Namespace) -> ReplayOptions:
    endpoint = args.endpoint
    if endpoint is not None and not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    return ReplayOptions(
        base_url=args.base_url,
        api_key=args.api_key,
        x_api_key=args.x_api_key,
        model=args.model,
        endpoint=endpoint,
        schedule=args.schedule,
        root_schedule=args.root_schedule,
        speedup=args.speedup,
        concurrency=args.concurrency,
        timeout_s=args.timeout,
        fixed_output_length=args.fixed_output_length,
        fallback_output_tokens=args.fallback_output_tokens,
        max_seq_len=args.max_seq_len,
        max_input_tokens=args.max_input_tokens,
        source_root=args.source_root,
        allow_source_changes=args.allow_source_changes,
        allow_partial=args.allow_partial,
        payload_mode=args.payload_mode,
        chain_mismatch_policy=args.chain_mismatch_policy,
        dry_run=args.dry_run,
    )


def _cmd_plan(args: argparse.Namespace) -> None:
    options = _replay_options(args)
    nodes, tree_ids = load_replay_nodes(
        args.database,
        tree_ids=args.tree_id,
        limit_trees=args.limit_trees,
        options=options,
    )
    _print(_compact_plan(replay_plan(nodes, tree_ids)))


def _cmd_replay(args: argparse.Namespace) -> None:
    options = _replay_options(args)
    nodes, tree_ids = load_replay_nodes(
        args.database,
        tree_ids=args.tree_id,
        limit_trees=args.limit_trees,
        options=options,
    )
    plan = replay_plan(nodes, tree_ids)
    if args.dry_run:
        _print({"plan": _compact_plan(plan), "result": {"dry_run": True}})
        return
    result = asyncio.run(run_replay(nodes, tree_ids, options, args.output))
    _print({"plan": _compact_plan(plan), "result": result})


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tree-id",
        action="append",
        help="Replay this tree ID; repeat for multiple trees. Defaults to replayable trees.",
    )
    parser.add_argument(
        "--limit-trees",
        type=_positive_int,
        help="Limit the number of selected trees.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Use only eligible nodes from incomplete or relation-unresolved trees.",
    )


def _add_replay_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("database", type=_path)
    _add_selection_arguments(parser)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="Bearer token; defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--x-api-key",
        help="Send this value in the x-api-key request header.",
    )
    parser.add_argument("--model", help="Override every recorded model name.")
    parser.add_argument("--endpoint", help="Override every recorded HTTP route.")
    parser.add_argument(
        "--schedule",
        choices=("causal", "fixed", "asap"),
        default="causal",
        help=(
            "causal: replay completion + recorded edge delay; fixed: captured global arrival "
            "timeline with dependency gating; asap: dependencies only"
        ),
    )
    parser.add_argument(
        "--root-schedule",
        choices=("simultaneous", "recorded"),
        default="simultaneous",
        help="Release root trees together or preserve their captured offsets in causal mode.",
    )
    parser.add_argument(
        "--speedup",
        type=_positive_float,
        default=1.0,
        help="Divide recorded waits by this factor.",
    )
    parser.add_argument(
        "--concurrency",
        type=_nonnegative_int,
        default=128,
        help="Maximum concurrent HTTP requests; 0 removes client concurrency limits.",
    )
    parser.add_argument("--timeout", type=_positive_float, default=3600.0)
    output_length = parser.add_mutually_exclusive_group()
    output_length.add_argument(
        "--fixed-output-length",
        action="store_true",
        dest="fixed_output_length",
        help=(
            "Opt in to recorded-OSL request controls: send the recorded output-token "
            "count as the request limit and set ignore_eos=true. By default the source "
            "payload is left unchanged."
        ),
    )
    output_length.add_argument(
        "--natural-eos",
        action="store_false",
        dest="fixed_output_length",
        help=(
            "Compatibility spelling for the default behavior: leave the source payload's "
            "output controls unchanged."
        ),
    )
    parser.set_defaults(fixed_output_length=False)
    parser.add_argument(
        "--fallback-output-tokens",
        type=_positive_int,
        help="Fallback only for explicitly allowed partial nodes without usage.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=_positive_int,
        help=(
            "Local preflight guard for recorded input+output tokens; not sent to the "
            "server and not a live output limit."
        ),
    )
    parser.add_argument(
        "--max-input-tokens",
        type=_positive_int,
        help="Local preflight guard for recorded input tokens; not sent to the server.",
    )
    parser.add_argument(
        "--source-root",
        type=_path,
        help="Resolve indexed relative payload paths below a relocated attempt directory.",
    )
    parser.add_argument("--allow-source-changes", action="store_true")
    parser.add_argument(
        "--payload-mode",
        choices=("recorded", "live-chain"),
        default="recorded",
        help=(
            "recorded (default): resend each captured input/tool-result payload and use "
            "live output only for metrics; live-chain (experimental): replace each indexed "
            "parent-response slice with the replayed /v1/responses output and rebind "
            "captured tool results"
        ),
    )
    parser.add_argument(
        "--chain-mismatch-policy",
        choices=("error", "recorded"),
        default="error",
        help=(
            "On a live-chain structure/tool mismatch, stop that chain node (error) or "
            "explicitly fall back to its captured payload (recorded)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-replay",
        description=(
            "Index TensorRT-LLM request traces into an evidence-bearing agent DAG and "
            "replay it against an OpenAI-compatible server."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Stream raw logs into SQLite and infer the DAG.")
    build.add_argument("attempt_dir", type=_path)
    build.add_argument("--output-dir", type=_path, required=True)
    build.add_argument("--database", type=_path)
    build.add_argument("--no-perf", action="store_true")
    build.add_argument("--no-export", action="store_true")
    build.add_argument("--rebuild", action="store_true")
    build.add_argument("--progress-every", type=int, default=1000)
    build.set_defaults(handler=_cmd_build)

    infer = subparsers.add_parser("infer", help="Recompute agents and dependency edges.")
    infer.add_argument("database", type=_path)
    infer.set_defaults(handler=_cmd_infer)

    enrich = subparsers.add_parser(
        "enrich-messages",
        help="Rescan payloads for structured agent_message items, then re-infer the DAG.",
    )
    enrich.add_argument("database", type=_path)
    enrich.add_argument("--progress-every", type=int, default=10_000)
    enrich.set_defaults(handler=_cmd_enrich_messages)

    enrich_lineage = subparsers.add_parser(
        "enrich-lineage",
        help=(
            "Read selected records by byte locator, index input/output item IDs, "
            "and infer prompt-chain parents."
        ),
    )
    enrich_lineage.add_argument("database", type=_path)
    enrich_lineage.add_argument(
        "--tree-id",
        action="append",
        help="Only enrich this tree ID; repeat for multiple trees. Defaults to all trees.",
    )
    enrich_lineage.set_defaults(handler=_cmd_enrich_lineage)

    inspect = subparsers.add_parser("inspect", help="Print capture and graph quality statistics.")
    inspect.add_argument("database", type=_path)
    inspect.set_defaults(handler=_cmd_inspect)

    export = subparsers.add_parser("export", help="Export portable metadata JSONL and manifest.")
    export.add_argument("database", type=_path)
    export.add_argument("--output-dir", type=_path, required=True)
    export.set_defaults(handler=_cmd_export)

    plan = subparsers.add_parser("plan", help="Validate and summarize a replay selection.")
    _add_replay_arguments(plan)
    plan.set_defaults(handler=_cmd_plan)

    replay = subparsers.add_parser("replay", help="Execute a dependency-aware replay.")
    _add_replay_arguments(replay)
    replay.add_argument("--output", type=_path, required=True)
    replay.set_defaults(handler=_cmd_replay)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "progress_every", 0) < 0:
        parser.error("--progress-every must be non-negative")
    if (
        getattr(args, "fallback_output_tokens", None) is not None
        and not getattr(args, "allow_partial", False)
    ):
        parser.error("--fallback-output-tokens requires --allow-partial")
    args.handler(args)
