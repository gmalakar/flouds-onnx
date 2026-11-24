# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
"""
flouds-onnx CLI — entry-point: flouds-onnx <command> [options]

Commands:
  pull    <model_name>  [--for fe] [--task T] [--optimize] [--force]
                        [--trust-remote-code] [--use-external-data-format]
                        [--use-fallback-if-failed] [--hf-token T]
  list
  info    <model_name>  [--for fe]
  remove  <model_name>  [--for fe]
  reload  <model_name>  [--for fe]
  stats
  serve   [--host H] [--port P]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def _client() -> Any:
    from flouds_onnx.api.client import get_default_client

    return get_default_client()


def _version() -> str:
    from flouds_onnx import __version__

    return __version__


def cmd_pull(args: argparse.Namespace) -> int:
    print(f"Pulling '{args.model_name}' (model_for={args.model_for}, optimize={args.optimize}) ...")
    try:
        from flouds_onnx.config.export_options import ExportOptions

        # Every export option the parser defines, by the field name ExportOptions uses.
        # Empty strings from the command line mean "not given".
        given = {name: getattr(args, name) for name in ExportOptions.model_fields if hasattr(args, name)}
        options = ExportOptions.build(**{name: (value if value != "" else None) for name, value in given.items()})
        if options.hf_token is None:
            # Prefer the environment: a token on the command line is visible in the
            # process list and saved in shell history.
            options = options.model_copy(update={"hf_token": os.environ.get("HUGGINGFACE_HUB_TOKEN") or None})
        manifest = _client().pull(args.model_name, model_for=args.model_for, force=args.force, **options.as_kwargs())
        print(f"OK  {manifest.model_name}  [{manifest.model_for}]  pulled_at={manifest.pulled_at}")
        print(f"    strategy : {manifest.session_strategy}")
        print(f"    onnx     : {', '.join(manifest.onnx_files) or 'none'}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    manifests = _client().list()
    if not manifests:
        print("No models in local store.")
        return 0
    fmt = "{:<50}  {:<10}  {:<25}  {}"
    print(fmt.format("MODEL", "FOR", "PULLED AT", "STRATEGY"))
    print("-" * 105)
    for m in manifests:
        print(fmt.format(m.model_name[:48], m.model_for, m.pulled_at[:24], m.session_strategy))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    manifest = _client().get_manifest(args.model_name, args.model_for)
    if manifest is None:
        print(f"Model '{args.model_name}' not found.", file=sys.stderr)
        return 1
    print(json.dumps(manifest.model_dump(mode="json"), indent=2))
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    client = _client()
    present = client.get_manifest(args.model_name, args.model_for) is not None
    removed = client.remove(args.model_name, args.model_for)
    if removed:
        print(f"Removed '{args.model_name}'")
        return 0
    if present:
        print(f"ERROR: '{args.model_name}' could not be fully deleted; a file may still be in use.", file=sys.stderr)
    else:
        print(f"'{args.model_name}' was not in the local store.", file=sys.stderr)
    return 1


def cmd_reload(args: argparse.Namespace) -> int:
    try:
        loaded = _client().reload(args.model_name, args.model_for)
        print(f"Reloaded '{loaded.model_name}' — strategy={loaded.session_strategy.value}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def cmd_stats(args: argparse.Namespace) -> int:
    print(json.dumps(_client().cache_stats(), indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        from flouds_onnx.api.server import run_server

        print(f"Starting flouds-onnx server on {args.host}:{args.port} ...")
        run_server(host=args.host, port=args.port)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flouds-onnx", description="Ollama-style ONNX model store and runtime")
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # pull
    p = sub.add_parser("pull", help="Pull (export) a model to the local store")
    p.add_argument("model_name")
    p.add_argument("--for", dest="model_for", default="fe", help="fe|s2s|sc|ranker|llm (default: fe)")
    p.add_argument("--task", default="", help="Optimum export task")
    p.add_argument("--optimize", action="store_true", default=True)
    p.add_argument("--no-optimize", dest="optimize", action="store_false")
    p.add_argument("--optimization-level", type=int, default=99)
    p.add_argument("--opset-version", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--framework", default="")
    p.add_argument("--library", default="")
    p.add_argument("--normalize-embeddings", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--use-external-data-format", action="store_true")
    p.add_argument("--use-subprocess", action="store_true")
    p.add_argument("--use-fallback-if-failed", action="store_true")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--skip-validator", action="store_true")
    p.add_argument("--revision", default="", help="Branch, tag or commit to export; a commit hash gives a reproducible export")
    p.add_argument("--hf-token", default="", help="HuggingFace API token. Prefer setting HUGGINGFACE_HUB_TOKEN instead.")
    p.set_defaults(func=cmd_pull)

    # list
    p = sub.add_parser("list", help="List all locally stored models")
    p.set_defaults(func=cmd_list)

    # info
    p = sub.add_parser("info", help="Show manifest for a model")
    p.add_argument("model_name")
    p.add_argument("--for", dest="model_for", default="fe")
    p.set_defaults(func=cmd_info)

    # remove
    p = sub.add_parser("remove", help="Delete a model from the local store")
    p.add_argument("model_name")
    p.add_argument("--for", dest="model_for", default="fe")
    p.set_defaults(func=cmd_remove)

    # reload
    p = sub.add_parser("reload", help="Evict and re-load a model session from disk")
    p.add_argument("model_name")
    p.add_argument("--for", dest="model_for", default="fe")
    p.set_defaults(func=cmd_reload)

    # stats
    p = sub.add_parser("stats", help="Show session cache statistics")
    p.set_defaults(func=cmd_stats)

    # serve
    p = sub.add_parser("serve", help="Start the HTTP server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=19720)
    p.set_defaults(func=cmd_serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from flouds_onnx.exceptions import FloudsOnnxError

    try:
        code = args.func(args)
    except FloudsOnnxError as exc:
        # For example an invalid model name, or a name that clashes with a stored model.
        print(f"ERROR: {exc}", file=sys.stderr)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
