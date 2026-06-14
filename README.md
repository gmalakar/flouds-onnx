# floudsonnx

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

`floudsonnx` is a small ONNX model store and runtime for Python. It can pull
Hugging Face models into a local ONNX store, cache runtime sessions, and load
models as either `onnxruntime.InferenceSession` or Optimum
`ORTModelForSeq2SeqLM` objects.

The package is designed for applications that want an Ollama-like local model
store, but for ONNX artifacts.

## Important: Hugging Face Models Must Be Converted To ONNX

`floudsonnx` runs ONNX models. Hugging Face models must be exported to ONNX
before they can be loaded as runtime sessions.

Install the `export` extra when you want `floudsonnx` to convert Hugging Face
models automatically:

```bash
pip install "floudsonnx[export]"
```

That extra installs `flouds-model-exporter`, which is required for converting
Hugging Face models to ONNX. Without it, `floudsonnx` can still load models
that already exist in the local ONNX store, but it cannot pull/export new
Hugging Face models.

For private or gated Hugging Face models, provide a token with the standard
Hugging Face Hub variable:

```bash
export HUGGINGFACE_HUB_TOKEN="hf_xxx_your_token"
```

`HUGGINGFACE_HUB_TOKEN` is read by the Hugging Face/exporter stack during
pull/export.

You can also pass a token per call:

```python
from floudsonnx import pull

pull("org/private-model", model_for="fe", hf_token="hf_xxx_your_token")
```

Never commit real Hugging Face tokens to source control.

## Features

- Local model store under `~/.flouds/models` by default.
- Auto-export from Hugging Face through the optional `flouds-model-exporter`
  integration.
- Runtime loading for feature extraction, sequence classification, ranker,
  seq2seq, and LLM-style model categories.
- Thread-safe session caching for ONNX Runtime and seq2seq models.
- Tokenizer loading and caching.
- Python API, CLI, and optional FastAPI server.
- Manifest files for locally stored models.

## Installation

`floudsonnx` supports Python `3.11` and `3.12`.

```bash
pip install floudsonnx
```

The core install loads models that are already exported to ONNX. Install extras
for export, seq2seq, or server support:

```bash
pip install "floudsonnx[export]"          # auto-export via flouds-model-exporter
pip install "floudsonnx[seq2seq]"         # Optimum ORTModelForSeq2SeqLM support
pip install "floudsonnx[server]"          # FastAPI + Uvicorn HTTP server
pip install "floudsonnx[all]"             # export + seq2seq + server
```

Core dependencies:

- `onnxruntime`
- `transformers`
- `pydantic`
- `numpy`

Optional extras:

- `export`: `flouds-model-exporter>=1.0.1`
- `seq2seq`: `optimum[onnxruntime]`
- `server`: `fastapi`, `uvicorn[standard]`

For feature-extraction, classification, and ranker exports, prefer
`library="transformers"` to avoid `sentence-transformers` auto-detection in the
exporter.

### Store Location And `ONNX_PATH`

`floudsonnx` stores models under `~/.flouds/models` by default. Override the
store root with `FloudsOnnxSettings`:

```python
from floudsonnx import FloudsOnnxClient, FloudsOnnxSettings

client = FloudsOnnxClient(
    FloudsOnnxSettings(onnx_path="/path/to/onnx/models")
)
```

When `floudsonnx` calls `flouds-model-exporter`, it sets the raw `ONNX_PATH`
environment variable internally so exported files land in the selected
`floudsonnx` store.

If you run `flouds-model-exporter` directly, use its native `ONNX_PATH`
variable:

```bash
export ONNX_PATH="/path/to/onnx/models"
flouds-export export --model-name t5-small --model-for s2s --task seq2seq-lm
```

## Quickstart

This example pulls a feature-extraction model, loads it, tokenizes input text,
builds the ONNX Runtime input feed from the session's actual inputs, and runs
inference.

```python
import numpy as np

from floudsonnx import create_model

model = create_model(
    "sentence-transformers/all-MiniLM-L6-v2",
    model_for="fe",
    task="feature-extraction",
    library="transformers",
    normalize_embeddings=True,
)

encoded = model.tokenizer(
    ["Hello world"],
    return_tensors="np",
    padding=True,
    truncation=True,
    max_length=64,
)

session_inputs = {item.name for item in model.session.get_inputs()}
feed = {name: encoded[name].astype(np.int64) for name in session_inputs if name in encoded}

# Some exported encoder models require token_type_ids even when the tokenizer
# does not return them.
for missing_name in session_inputs - set(feed):
    feed[missing_name] = np.zeros_like(next(iter(feed.values())))

outputs = model.run(None, feed)
print(outputs[0].shape)
```

The `create_model()` helper pulls/exports the model if needed, then loads it.
Use `load_model()` when the model is already present on disk and you do not
want auto-export.

## Seq2Seq Example

Seq2seq loading requires the `seq2seq` extra. Pulling from Hugging Face also
requires the `export` extra.

```python
from floudsonnx import create_model

model = create_model("t5-small", model_for="s2s", task="seq2seq-lm")

encoded = model.tokenizer(
    ["summarize: The quick brown fox jumps over the lazy dog."],
    return_tensors="pt",
    truncation=True,
    max_length=64,
)

tokens = model.seq2seq_model.generate(
    input_ids=encoded["input_ids"],
    max_new_tokens=32,
)

print(model.tokenizer.decode(tokens[0], skip_special_tokens=True))
```

## Model Types

| `model_for` | Default export task | Runtime strategy |
|---|---|---|
| `fe` | `feature-extraction` | `onnxruntime.InferenceSession` |
| `sc` | `text-classification` | `onnxruntime.InferenceSession` |
| `ranker` | `text-classification` | `onnxruntime.InferenceSession` |
| `s2s` | `seq2seq-lm` | `ORTModelForSeq2SeqLM` |
| `llm` | `text-generation-with-past` | `onnxruntime.InferenceSession` by default, or `ORTModelForSeq2SeqLM` when configured with `use_seq2seqlm=True` |

## Python API

Top-level convenience functions:

```python
from floudsonnx import (
    create_model,
    list_models,
    load_model,
    pull,
    remove_model,
)
```

Available top-level functions:

| Function | Description |
|---|---|
| `create_model(model_name, model_for="fe", **kwargs)` | Pull/export if needed, then load the model. |
| `load_model(model_name, model_for="fe")` | Load an existing local model without auto-export. |
| `pull(model_name, model_for="fe", **kwargs)` | Export/store the model and return its manifest. |
| `list_models()` | Return local `ModelManifest` objects. |
| `remove_model(model_name, model_for="fe")` | Delete the local model and evict cached sessions. |

For explicit settings and cache control, use `FloudsOnnxClient`:

```python
from floudsonnx import FloudsOnnxClient, FloudsOnnxSettings

settings = FloudsOnnxSettings(
    home_dir="/data/flouds",
    session_provider="CPUExecutionProvider",
)
client = FloudsOnnxClient(settings)

manifest = client.pull(
    "BAAI/bge-base-en-v1.5",
    model_for="fe",
    task="feature-extraction",
    library="transformers",
    normalize_embeddings=True,
)

model = client.load_model("BAAI/bge-base-en-v1.5", model_for="fe")
print(model.session_strategy)

client.unload("BAAI/bge-base-en-v1.5", model_for="fe")
client.remove("BAAI/bge-base-en-v1.5", model_for="fe")
```

`FloudsOnnxClient` methods:

- `pull(...)`
- `list()`
- `remove(model_name, model_for="fe")`
- `create_model(...)`
- `load_model(model_name, model_for="fe")`
- `reload(model_name, model_for="fe")`
- `unload(model_name, model_for="fe")`
- `is_loaded(model_name, model_for="fe")`
- `cache_stats()`

## LoadedModel

`create_model()` and `load_model()` return a `LoadedModel` with:

- `model_name`
- `model_for`
- `model_dir`
- `config`
- `tokenizer`
- `session_strategy`
- `session` for `onnxruntime.InferenceSession` models
- `seq2seq_model` for `ORTModelForSeq2SeqLM` models
- `is_seq2seq`
- `run(output_names, input_feed, run_options=None)`

For encoder/classification/ranker models, call `model.run(...)`.

For seq2seq models, call `model.seq2seq_model.generate(...)`.

## CLI

```bash
floudsonnx --help
```

Commands:

```bash
# Pull/export a model to the local store
floudsonnx pull sentence-transformers/all-MiniLM-L6-v2 --for fe --task feature-extraction

# Disable ONNX optimization during pull
floudsonnx pull t5-small --for s2s --task seq2seq-lm --no-optimize

# List locally stored models
floudsonnx list

# Show manifest JSON
floudsonnx info sentence-transformers/all-MiniLM-L6-v2 --for fe

# Remove from local store
floudsonnx remove sentence-transformers/all-MiniLM-L6-v2 --for fe

# Evict and reload from disk
floudsonnx reload sentence-transformers/all-MiniLM-L6-v2 --for fe

# Show session cache stats
floudsonnx stats

# Start the optional HTTP server
floudsonnx serve --host 127.0.0.1 --port 19720
```

Current `pull` CLI options:

- `--for`
- `--task`
- `--optimize`
- `--no-optimize`
- `--optimization-level`
- `--opset-version`
- `--device`
- `--framework`
- `--library`
- `--normalize-embeddings`
- `--force`
- `--trust-remote-code`
- `--use-external-data-format`
- `--use-subprocess`
- `--use-fallback-if-failed`
- `--merge`
- `--skip-validator`
- `--hf-token`

Example:

```bash
floudsonnx pull sentence-transformers/all-MiniLM-L6-v2 \
  --for fe \
  --task feature-extraction \
  --library transformers \
  --normalize-embeddings
```

## Optional HTTP Server

Install the server extra:

```bash
pip install "floudsonnx[server]"
```

Start the server:

```bash
floudsonnx serve --host 127.0.0.1 --port 19720
```

Routes:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check. |
| `GET` | `/api/v1/models` | List local model manifests. |
| `GET` | `/api/v1/models/{name}?model_for=fe` | Get one manifest. |
| `POST` | `/api/v1/models/pull` | Pull/export a model. |
| `POST` | `/api/v1/models/load` | Load a local model into cache. |
| `POST` | `/api/v1/models/reload` | Evict and reload a model. |
| `POST` | `/api/v1/models/unload` | Evict a model from memory. |
| `DELETE` | `/api/v1/models/{name}?model_for=fe` | Remove a model from disk. |
| `GET` | `/api/v1/stats` | Return cache statistics. |

Pull request body:

```json
{
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "model_for": "fe",
  "task": "feature-extraction",
  "force": false,
  "optimize": false,
  "trust_remote_code": false,
  "use_external_data_format": false,
  "use_fallback_if_failed": false,
  "hf_token": null
}
```

Load/reload/unload request body:

```json
{
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "model_for": "fe"
}
```

## Configuration

Configure `floudsonnx` with `FloudsOnnxSettings` when creating a client:

```python
from floudsonnx import FloudsOnnxClient, FloudsOnnxSettings

client = FloudsOnnxClient(
    FloudsOnnxSettings(
        home_dir="/data/flouds",
        onnx_path="/data/flouds/models",
        session_provider="CPUExecutionProvider",
        encoder_cache_max=5,
        decoder_cache_max=5,
        seq2seq_cache_max=3,
    )
)
```

Additional variables:

| Environment variable | Used by | Description |
|---|---|---|
| `HUGGINGFACE_HUB_TOKEN` | Hugging Face Hub / `flouds-model-exporter` | Standard token variable for private or gated Hugging Face models. |
| `ONNX_PATH` | `flouds-model-exporter` | Native exporter output root. `floudsonnx` sets this internally during export; configure the `floudsonnx` store with `FloudsOnnxSettings`. |

## Local Store Layout

Default layout:

```text
~/.flouds/
`-- models/
    |-- fe/
    |   `-- all-MiniLM-L6-v2/
    |       |-- model.onnx
    |       |-- model_optimized.onnx
    |       |-- tokenizer.json
    |       `-- manifest.json
    |-- s2s/
    |-- sc/
    |-- ranker/
    `-- llm/
```

Each model directory contains a `manifest.json` with the model name, model
type, export options, selected runtime strategy, discovered ONNX files, and
model config.

## Development

```bash
pip install -r requirements-dev.txt
pip install -e ".[export,seq2seq,server]"
pre-commit install
pytest tests/unit -v
```

Useful checks:

```bash
python tools/check_dependency_sync.py
black --check src tests tools
isort --check-only src tests tools
flake8 src tests tools
mypy src/floudsonnx
pyright src/floudsonnx
python -m build
twine check dist/*
```

Integration tests download and export real models:

```bash
pytest tests/integration -m integration -v
```

## Release

Releases are tag-driven through GitHub Actions. See
[`docs/RELEASE_PROCESS.md`](docs/RELEASE_PROCESS.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
