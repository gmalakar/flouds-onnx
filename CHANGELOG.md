# Changelog

All notable changes to `floudsonnx` will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-11

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
