# Changelog

All notable changes to `flouds-onnx` will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [2.1.0] - 2026-11-26

### Security

- The opt-in Hugging Face Hub tokenizer fallback (`allow_hub_tokenizer_fallback`) downloads each repository at a pinned commit, so a later push to `sentence-transformers/all-MiniLM-L6-v2` or `sentence-transformers/sentence-t5-base` cannot change what is loaded. Its bandit B615 exemption is removed.

### Changed

- transformers 5 is supported: `transformers>=4.44.0,<6` (was `<4.58.0`). A base or `[server]` install now gets transformers 5, which fixes the advisories pip-audit reports for transformers 4.57.6 (PYSEC-2025-217, PYSEC-2026-2288, -2289, -2290, -3929 and -4174). Tested with transformers 5.24.1 on Python 3.14 and 3.11. Token IDs and ONNX outputs for all-MiniLM-L6-v2, ms-marco-MiniLM-L-6-v2 and t5-small match transformers 4.57.6.
- The `export` and `seq2seq` extras still install transformers 4.57.6. They depend on optimum-onnx, whose latest release (0.1.0) requires `transformers<4.58.0`, so they still carry those advisories.

## [2.0.1] - 2026-11-26

### Changed

- Python 3.11, 3.12, 3.13 and 3.14 are supported (`requires-python = ">=3.11,<3.15"`). Development, CI checks that run once and the Flouds services use 3.14.
- NumPy 2 is supported: `numpy>=1.26.4,<3` (was `<1.27`). Tested with NumPy 2.5.5 on Python 3.14 and 2.4.6 on Python 3.11.
- Development tools: black 25.12.0, isort 6.1.0, flake8 7.5.0, mypy 1.20.2, bandit 1.9.4, pre-commit 4.7.0, pre-commit-hooks 6.0.0.

### Fixed

- `FloudsOnnxError` and its subclasses keep `detail` and their attributes through `pickle` and `copy` (the keyword-only `detail` was lost).
- The local pre-commit hooks start under any `python` on PATH (`python -E`; pyright runs through `tools/run_in_venv.py`), and the pre-push check waits for the checks on Windows.

### Requires

- The `export` extra requires `flouds-model-exporter>=2.0.2,<3`: 2.0.2 patches optimum-onnx for Python 3.14 (`functools.partial` binding), which seq2seq export needs, and supports Python 3.11 to 3.14.

## [2.0.0] - 2026-11-24

### Renamed

- **Breaking:** the project is renamed from `floudsonnx` to `flouds-onnx`. Install it as `flouds-onnx`, import it as `flouds_onnx`, run the command as `flouds-onnx`, and set `FLOUDS_ONNX_*` environment variables instead of `FLOUDSONNX_*`; the old variables are no longer read. The health check reports `"service": "flouds-onnx"`. Class names and the store location are unchanged. The `floudsonnx` package stays on PyPI at 1.0.0.

### Security

- A model name of `..` or `.`, or a `model_for` value containing path separators, could address directories outside the model store. `remove()` and `pull(force=True)` could delete them, including the whole store. Names and types are now validated and every store path is confined to the store.

- The HTTP server now controls access on every route except `/health`. With `FLOUDS_ONNX_API_TOKEN` set it requires `Authorization: Bearer <token>`; without one it serves only requests from the same machine, checks the `Host` header against DNS rebinding, and refuses to start on a non-loopback address. Requests from web pages on other origins are refused.
- `trust_remote_code` and `hf_token` are no longer accepted in HTTP requests. Remote code would run on the server for any caller, and the token would appear in request bodies and logs.
- HTTP error responses no longer include filesystem paths or third-party error output; the detail is written to the server log.
- A model from one organisation is no longer served in place of another with the same short name (`org-a/bert-base` and `org-b/bert-base` share a folder). Asking for the second raises `ModelNameConflictError`.
- **Breaking:** the `export` extra requires `flouds-model-exporter>=2.0.1,<3`. Earlier exporter versions could load a model with `trust_remote_code=True` after the caller passed `False`, and saved the Hugging Face token to disk as the machine's default credential; 2.0.0 does neither.
- Tokenizers are loaded from local files only. The automatic download of a replacement tokenizer from the Hugging Face Hub is off unless `allow_hub_tokenizer_fallback` is set.

### Fixed

- The HTTP server ran exports and model loads on the event loop, so one export blocked every other request, including `/health`, for its whole duration. Blocking work now runs in a worker thread.
- A model that is not in the store is reported as `404` instead of `500`.
- Evicting, unloading or checking one model no longer also matches models whose folder name starts with the same characters (for example `all-MiniLM-L6` and `all-MiniLM-L6-v2`).
- A store root not named `models` (for example `onnx_path="/data/store"`) works: the exported files are moved into the store. Before, they were written beside it and the model could never be loaded.
- A pull that produces no ONNX files fails with `ExportError` instead of writing a manifest for an unusable model.
- A pull that fails before anything is written (for example because the exporter is not installed) no longer leaves an empty model folder behind.
- The manifest records the export options that were used. It previously recorded fixed defaults (`optimize: true`, `opset_version: 17`) whatever was requested.
- `manifest.json` is written atomically, so an interrupted write cannot leave a truncated file.
- `remove()` returns `False` when the directory could not be deleted (for example a model file still open on Windows) instead of reporting success.
- Reloading or removing a model now makes every thread reload its tokenizer; before, only the calling thread did and the others kept the old one.
- Exports no longer change the process environment: the store root is passed to the exporter as an argument instead of through `ONNX_PATH`. Exports in one process still take turns, because the exporter changes process-wide state (stderr, thread-count variables) while it runs.
- Errors from the exporter keep their meaning: a model that needs `trust_remote_code` raises `RemoteCodeRequiredError` (HTTP `422`), another process exporting the same model raises `ExportInProgressError` (HTTP `409`), and an invalid name raises `InvalidModelReferenceError` (HTTP `400`). Before, all of them became a generic `ExportError`, and an export already in progress ended the process with `SystemExit`. A failed export's `detail` is the exporter's last error instead of only "All export attempts failed".
- The configured execution provider is now applied to seq2seq models, which previously always used the default.
- Reading the store (`list`, `exists`, `get_manifest`) no longer creates the store directory.
- `flouds-onnx remove` exits with code 1 when nothing was removed, and an invalid model name prints an error instead of a traceback.

### Changed

- **Breaking:** `model_for` must be 1-32 letters, digits, `-` or `_`, starting with a letter or digit. Custom types that fit are still accepted.
- **Breaking:** model names that are empty, consist only of dots, or end with a dot are rejected.
- **Breaking (HTTP):** unknown request fields are rejected with `422`, and `POST /api/v1/models/create` no longer returns `model_dir`.
- **Breaking (HTTP):** `flouds-onnx serve --host 0.0.0.0` (or any non-loopback address) needs `FLOUDS_ONNX_API_TOKEN`.
- The server's reported version comes from the package version instead of a hard-coded string.
- Added `FloudsOnnxClient.get_manifest()`; the CLI and server no longer reach into private attributes for it.
- Invalid names and types raise the new `InvalidModelReferenceError`, which is both a `FloudsOnnxError` and a `ValueError`. The HTTP server returns 400 for it.
- **Breaking:** the Hub tokenizer fallback is opt-in (see Security).
- **Breaking:** the manifest's `export_options` fields are optional and mean "as used"; `None` means the exporter's default applied. Manifests carry `schema_version` (2); older ones load as version 1.
- The process-wide default client, the CLI and the HTTP server read `FLOUDS_ONNX_*` environment variables through `FloudsOnnxSettings.from_env()`.
- Example scripts are named `examples/example_*.py`; the manual smoke scripts moved from the repository root to `scripts/`.
- CI runs coverage with an 85% floor, Bandit, pip-audit and a build-and-install check, and installs the server extra so the HTTP tests run. A release tag no longer publishes unless lint, type checks and unit tests pass.
- Export options are defined once, in `flouds_onnx.ExportOptions`. `pull()` and `create_model()` accept an `options=` object, the same keyword arguments as before, or both. The registry, exporter bridge, HTTP request model and CLI all read from it instead of each listing the options.
- **Breaking:** `ExporterBridge.export()` takes its options as keywords or an `options=` object; `task` is the only option that can still be passed positionally.
- **Breaking:** removed code that was never used: the decoder session cache with its `decoder_cache_max` setting and `FLOUDS_ONNX_DECODER_CACHE_MAX` variable, the `decoder` entry in cache statistics, `SessionPool.evict_all()`, `TokenizerCache.clear()`, and the `CacheError` and `ModelAlreadyExistsError` exceptions.
- **Breaking for contributors:** `pyproject.toml` is the only place dependencies and the mypy and pyright settings are defined. `requirements-*.txt`, `mypy.ini`, `pyrightconfig.json` and `tools/check_dependency_sync.py` are gone. Install with `pip install -e ".[dev,export,seq2seq,server]"`.

### Added

- Settings: `strict_provider`, `intra_op_num_threads`, `inter_op_num_threads`, `tokenizer_cache_max`, `allow_hub_tokenizer_fallback`, and `FloudsOnnxSettings.from_env()`.
- `GET /health/ready`, which returns 503 when the model store is unusable; `/health` now includes the version.
- `flouds-onnx --version`. `flouds-onnx pull` reads `HUGGINGFACE_HUB_TOKEN` when `--hf-token` is not given.
- `ModelNameConflictError`, `ServerConfigurationError`.
- Dependabot configuration for Python packages and GitHub Actions.
- `ExportOptions` and `SafeExportOptions` (the subset accepted over HTTP).
- `revision` export option (`--revision` on the CLI, `revision` in HTTP pull requests): the branch, tag or commit to export. It is recorded in the manifest as `hf_revision`, and a later pull that asks for a different revision raises `RevisionMismatchError` (HTTP `409`) instead of returning the stored one; pass `force=True` to replace it.
- `RemoteCodeRequiredError`, `ExportInProgressError` and `RevisionMismatchError`. `ModelNameConflictError` is now reported as HTTP `409` instead of `500`.

## [1.0.0] - 2026-06-14
### Added

- `FloudsOnnxClient`, the primary Python API facade.
- Top-level helpers: `create_model()`, `load_model()`, `pull()`, `list_models()`, and `remove_model()`.
- Local ONNX model store with per-model `manifest.json` files.
- `ModelLoader`, `ModelRegistry`, `SessionPool`, and `TokenizerCache`.
- `ExporterBridge`, a lazy integration with `flouds-model-exporter` that sets `ONNX_PATH` internally during export.
- Automatic default task resolution from `model_for`.
- Runtime strategies for `onnxruntime.InferenceSession` and Optimum `ORTModelForSeq2SeqLM`.
- `LoadedModel` wrapper with `.run()` for encoder-style models and `.seq2seq_model` for seq2seq models.
- CLI commands: `pull`, `list`, `info`, `remove`, `reload`, `stats`, and `serve`.
- Explicit CLI export options including `--library`, `--normalize-embeddings`, `--device`, `--framework`, `--use-subprocess`, `--merge`, and `--skip-validator`.
- Optional FastAPI HTTP server with `floudsonnx[server]`.
- Example scripts and manual smoke tests for feature extraction, seq2seq, ranker, and LLM-style models.
- Unit and integration test coverage plus CI, pre-commit, dependency sync, version bump, and release tooling.

### Changed

- Export configuration is passed explicitly through Python/API/CLI calls instead of `FLOUDSONNX_*` environment variables.
- Hugging Face token environment usage is standardized on `HUGGINGFACE_HUB_TOKEN`.
- `FloudsOnnxSettings` is an explicit Pydantic model for store paths, ORT provider, and cache sizes.
- Feature-extraction, classification, and ranker examples pass `library="transformers"` explicitly to avoid `sentence-transformers` auto-detection.

### Dependencies

- Core: `onnxruntime>=1.20.1`, `transformers>=4.44.0,<4.58.0`, `pydantic>=2.0`, `numpy>=1.26.4,<1.27`.
- `export`: `flouds-model-exporter>=1.0.1`.
- `seq2seq`: `optimum[onnxruntime]>=1.22.0`.
- `server`: `fastapi>=0.110.0`, `uvicorn[standard]>=0.29.0`.

## [0.1.0] - 2026-06-11

- Initial development release.
