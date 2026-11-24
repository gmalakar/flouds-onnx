# flouds-onnx

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://github.com/gmalakar/flouds-onnx/blob/main/LICENSE)

`flouds-onnx` is a small ONNX model store and runtime for Python. It can pull
Hugging Face models into a local ONNX store, cache runtime sessions, and load
models as either `onnxruntime.InferenceSession` or Optimum
`ORTModelForSeq2SeqLM` objects.

The package is designed for applications that want an Ollama-like local model
store, but for ONNX artifacts.

> **Renamed from `floudsonnx`.** From 2.0.0 the project is published as
> `flouds-onnx`. The old `floudsonnx` package stays on PyPI at 1.0.0 and receives no
> further releases. See [Moving from floudsonnx](#moving-from-floudsonnx).

## Important: Hugging Face Models Must Be Converted To ONNX

`flouds-onnx` runs ONNX models. Hugging Face models must be exported to ONNX
before they can be loaded as runtime sessions.

Install the `export` extra when you want `flouds-onnx` to convert Hugging Face
models automatically:

```bash
pip install "flouds-onnx[export]"
```

That extra installs `flouds-model-exporter`, which is required for converting
Hugging Face models to ONNX. Without it, `flouds-onnx` can still load models
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
from flouds_onnx import pull

pull("org/private-model", model_for="fe", hf_token="hf_xxx_your_token")
```

Never commit real Hugging Face tokens to source control.

## Moving from floudsonnx

Everything that carried the old name changed in 2.0.0:

| What | 1.x (`floudsonnx`) | 2.x (`flouds-onnx`) |
|---|---|---|
| Install | `pip install floudsonnx[export]` | `pip install "flouds-onnx[export]"` |
| Import | `import floudsonnx` | `import flouds_onnx` |
| Command | `floudsonnx pull ...` | `flouds-onnx pull ...` |
| Environment variables | `FLOUDSONNX_HOME_DIR`, `FLOUDSONNX_API_TOKEN`, ... | `FLOUDS_ONNX_HOME_DIR`, `FLOUDS_ONNX_API_TOKEN`, ... |
| Health check `service` field | `floudsonnx` | `flouds-onnx` |
| Repository | `github.com/gmalakar/floudsonnx` | `github.com/gmalakar/flouds-onnx` |

Class names (`FloudsOnnxClient`, `FloudsOnnxSettings`, `FloudsOnnxError`) and the
store location (`~/.flouds/models`) are unchanged, so models already in the store
are found without re-exporting. Uninstall the old package first
(`pip uninstall floudsonnx`); the two can be installed side by side but must not be
mixed. The old `FLOUDSONNX_*` variables are no longer read.

The 2.0.0 release has further breaking changes; see `CHANGELOG.md`.

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

`flouds-onnx` supports Python `3.11` and `3.12`.

```bash
pip install flouds-onnx
```

The core install loads models that are already exported to ONNX. Install extras
for export, seq2seq, or server support:

```bash
pip install "flouds-onnx[export]"          # auto-export via flouds-model-exporter
pip install "flouds-onnx[seq2seq]"         # Optimum ORTModelForSeq2SeqLM support
pip install "flouds-onnx[server]"          # FastAPI + Uvicorn HTTP server
pip install "flouds-onnx[all]"             # export + seq2seq + server
```

Core dependencies:

- `onnxruntime`
- `transformers`
- `pydantic`
- `numpy`

Optional extras:

- `export`: `flouds-model-exporter>=2.0.1,<3`
- `seq2seq`: `optimum[onnxruntime]`
- `server`: `fastapi`, `uvicorn[standard]`

For feature-extraction, classification, and ranker exports, prefer
`library="transformers"` to avoid `sentence-transformers` auto-detection in the
exporter.

### Store Location And `ONNX_PATH`

`flouds-onnx` stores models under `~/.flouds/models` by default. Override the
store root with `FloudsOnnxSettings`:

```python
from flouds_onnx import FloudsOnnxClient, FloudsOnnxSettings

client = FloudsOnnxClient(
    FloudsOnnxSettings(onnx_path="/path/to/onnx/models")
)
```

The store folder can have any name. `flouds-onnx` passes the store root to
`flouds-model-exporter` as an argument (the `ONNX_PATH` variable is not used or
changed), and afterwards makes sure the exported files are in the selected store.
A pull that produces no ONNX files fails with `ExportError` instead of recording a
model that cannot be loaded. Exports in one process run one at a time.

Pass `revision` to export a specific branch, tag or commit; a commit hash gives a
reproducible export. The revision is recorded in the manifest as `hf_revision`.
Asking later for a different revision raises `RevisionMismatchError` rather than
returning the stored export; pull with `force=True` to replace it.

```python
client.pull("sentence-transformers/all-MiniLM-L6-v2", revision="<commit hash from the model page>")
```

A model that needs `trust_remote_code` raises `RemoteCodeRequiredError` unless you
pass `trust_remote_code=True`, which runs code from the model repository; review it
first. The HTTP server never accepts that option, so such models must be pulled
with the Python API or the CLI.

Model names and model types are validated before they are used as folder names.
A name such as `..`, or a `model_for` containing a path separator, raises
`InvalidModelReferenceError`.

The store keeps one folder per short model name, so `org-a/bert-base` and
`org-b/bert-base` would share `bert-base`. The second one is not served in place
of the first: asking for it raises `ModelNameConflictError`. Remove the stored
model, or pull with `force=True` to replace it.

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

from flouds_onnx import create_model

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
from flouds_onnx import create_model

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
from flouds_onnx import (
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
from flouds_onnx import FloudsOnnxClient, FloudsOnnxSettings

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

### Export options

Export options can be passed as keyword arguments, as shown above, or as one
`ExportOptions` object that you build once and reuse:

```python
from flouds_onnx import ExportOptions, pull

options = ExportOptions(task="feature-extraction", library="transformers", normalize_embeddings=True)
pull("sentence-transformers/all-MiniLM-L6-v2", model_for="fe", options=options)
pull("BAAI/bge-small-en-v1.5", model_for="fe", options=options, optimize=True)  # keywords win
```

`ExportOptions` is the single definition of every option; the CLI flags and the
HTTP request body carry the same names. An option left as `None` uses its default.

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
flouds-onnx --help
```

Commands:

```bash
# Pull/export a model to the local store
flouds-onnx pull sentence-transformers/all-MiniLM-L6-v2 --for fe --task feature-extraction

# Disable ONNX optimization during pull
flouds-onnx pull t5-small --for s2s --task seq2seq-lm --no-optimize

# List locally stored models
flouds-onnx list

# Show manifest JSON
flouds-onnx info sentence-transformers/all-MiniLM-L6-v2 --for fe

# Remove from local store (exit code 1 if it was not there or could not be deleted)
flouds-onnx remove sentence-transformers/all-MiniLM-L6-v2 --for fe

# Print the installed version
flouds-onnx --version

# Evict and reload from disk
flouds-onnx reload sentence-transformers/all-MiniLM-L6-v2 --for fe

# Show session cache stats
flouds-onnx stats

# Start the optional HTTP server
flouds-onnx serve --host 127.0.0.1 --port 19720
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
flouds-onnx pull sentence-transformers/all-MiniLM-L6-v2 \
  --for fe \
  --task feature-extraction \
  --library transformers \
  --normalize-embeddings
```

## Optional HTTP Server

Install the server extra:

```bash
pip install "flouds-onnx[server]"
```

Start the server:

```bash
flouds-onnx serve --host 127.0.0.1 --port 19720
```

### Access control

The server can download and export models, load them into memory and delete them from
disk, so every route except `/health` is protected.

| Mode | When | Who is served |
|---|---|---|
| Local | `FLOUDS_ONNX_API_TOKEN` is not set | Only requests from this machine: the connection must be from a loopback address and the `Host` header must be `localhost`, `127.0.0.1` or `::1`. |
| Token | `FLOUDS_ONNX_API_TOKEN` is set | Any client that sends `Authorization: Bearer <token>`. |

The server refuses to start on a non-loopback address (for example `--host 0.0.0.0`)
unless a token is set:

```bash
export FLOUDS_ONNX_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
flouds-onnx serve --host 0.0.0.0 --port 19720

curl -H "Authorization: Bearer $FLOUDS_ONNX_API_TOKEN" http://server:19720/api/v1/models
```

Requests made by a web page (those with an `Origin` header) are refused unless the
origin is the server itself or is listed, comma-separated, in
`FLOUDS_ONNX_ALLOWED_ORIGINS`. Behind a reverse proxy every request arrives from the
proxy's address, so set a token; local mode cannot tell proxied callers apart.

The server does not use TLS. Put it behind a proxy that does if the token travels over
a network.

Two export options are not available over HTTP:

- `trust_remote_code` runs Python from the model repository on the server. Use the CLI
  or the Python API for models that need it.
- `hf_token`: the server uses `HUGGINGFACE_HUB_TOKEN` from its own environment.

A request that includes either, or any other unknown field, gets a `422` response.

Routes:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness: the process is up. No token needed. |
| `GET` | `/health/ready` | Readiness: the model store is usable; `503` if not. No token needed. |
| `GET` | `/api/v1/models` | List local model manifests. |
| `GET` | `/api/v1/models/{name}?model_for=fe` | Get one manifest. |
| `POST` | `/api/v1/models/pull` | Pull/export a model. |
| `POST` | `/api/v1/models/create` | Pull/export if needed, then load the model into cache. |
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
  "optimization_level": 99,
  "opset_version": null,
  "device": "cpu",
  "framework": null,
  "library": "transformers",
  "normalize_embeddings": true,
  "use_external_data_format": false,
  "use_subprocess": false,
  "use_fallback_if_failed": false,
  "merge": false,
  "skip_validator": false,
  "revision": null
}
```

Use the same body with `POST /api/v1/models/create` when you want the remote
equivalent of Python `create_model()`.

Example:

```bash
curl -X POST http://127.0.0.1:19720/api/v1/models/create \
  -H "Content-Type: application/json" \
  -d '{"model_name":"sentence-transformers/all-MiniLM-L6-v2","model_for":"fe","task":"feature-extraction","library":"transformers","normalize_embeddings":true}'
```

Errors:

| Status | Meaning |
|---|---|
| `400` | The model name or `model_for` is not valid. |
| `401` | A token is configured and the request did not send it. |
| `403` | No token is configured and the request is not local, or the web origin is not allowed. |
| `404` | The model is not in the local store. |
| `409` | Another process is exporting the same model, a different model already uses the store folder, or the stored export is from a different revision. |
| `422` | The request body has a missing, wrong or unknown field, or the model needs `trust_remote_code` (pull it with the Python API or the CLI). |
| `500` | The operation failed. The response names the error type only; details are in the server log. |
| `501` | An optional package (`flouds-onnx[export]` or `flouds-onnx[seq2seq]`) is not installed on the server. |

Load/reload/unload request body:

```json
{
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "model_for": "fe"
}
```

## Configuration

Configure `flouds-onnx` with `FloudsOnnxSettings` when creating a client:

```python
from flouds_onnx import FloudsOnnxClient, FloudsOnnxSettings

client = FloudsOnnxClient(
    FloudsOnnxSettings(
        home_dir="/data/flouds",
        onnx_path="/data/flouds/models",
        session_provider="CPUExecutionProvider",
        strict_provider=False,
        intra_op_num_threads=4,
        encoder_cache_max=5,
        seq2seq_cache_max=3,
    )
)
```

| Setting | Default | Environment variable | Description |
|---|---|---|---|
| `home_dir` | `~/.flouds` | `FLOUDS_ONNX_HOME_DIR` | Base directory; models go in `<home_dir>/models` unless `onnx_path` is set. |
| `onnx_path` | unset | `FLOUDS_ONNX_ONNX_PATH` | Model store root, any folder name. |
| `session_provider` | `CPUExecutionProvider` | `FLOUDS_ONNX_SESSION_PROVIDER` | ONNX Runtime execution provider, used for encoder and seq2seq models. |
| `strict_provider` | `False` | `FLOUDS_ONNX_STRICT_PROVIDER` | Fail when the provider is unavailable instead of falling back to CPU. Set this in GPU deployments. |
| `intra_op_num_threads` | ONNX Runtime default | `FLOUDS_ONNX_INTRA_OP_THREADS` | Threads used inside one operator, per session. |
| `inter_op_num_threads` | ONNX Runtime default | `FLOUDS_ONNX_INTER_OP_THREADS` | Threads used across operators, per session. |
| `encoder_cache_max` | `5` | `FLOUDS_ONNX_ENCODER_CACHE_MAX` | Encoder sessions kept in memory. |
| `seq2seq_cache_max` | `3` | `FLOUDS_ONNX_SEQ2SEQ_CACHE_MAX` | Seq2seq models kept in memory. |
| `tokenizer_cache_max` | `16` | `FLOUDS_ONNX_TOKENIZER_CACHE_MAX` | Tokenizers kept per thread. |
| `allow_hub_tokenizer_fallback` | `False` | `FLOUDS_ONNX_ALLOW_HUB_TOKENIZER_FALLBACK` | Let a failed local tokenizer load download a known replacement from the Hugging Face Hub. |

The environment variables are read by `FloudsOnnxSettings.from_env()`. The
top-level helpers (`create_model()`, `pull()`, ...), the CLI and the HTTP server
use a default client built that way. A client you create yourself uses exactly
the settings you pass:

```python
client = FloudsOnnxClient(FloudsOnnxSettings.from_env(seq2seq_cache_max=1))
```

### Performance notes

- **Threads.** By default every ONNX Runtime session uses one thread per
  physical core. With several models loaded in one process, or several worker
  processes on one machine, the sessions compete for the same cores. Set
  `intra_op_num_threads` to roughly `cores / concurrent sessions`.
- **Provider.** If the requested provider is missing, the default is to fall
  back to CPU with a warning, which can be many times slower. Use
  `strict_provider=True` so a GPU deployment without its GPU build fails at
  start-up.
- **Cache limits are counts, not memory.** Three large seq2seq models can use
  more memory than fifty small encoders. Size the limits for your models.
- **Tokenizers are cached per thread**, because fast tokenizers are not safe to
  share between threads. Memory therefore grows with the number of threads
  that serve requests; `tokenizer_cache_max` bounds each thread.

Other variables:

| Environment variable | Used by | Description |
|---|---|---|
| `HUGGINGFACE_HUB_TOKEN` | Hugging Face Hub / `flouds-model-exporter` | Token for private or gated Hugging Face models. Read by the CLI and the HTTP server. |
| `FLOUDS_ONNX_API_TOKEN` | HTTP server | Bearer token required on every route except the health checks. |
| `FLOUDS_ONNX_ALLOWED_ORIGINS` | HTTP server | Comma-separated web origins allowed to call the server. |
| `ONNX_PATH` | `flouds-model-exporter` | Native exporter output root, used only when you run `flouds-export` yourself. `flouds-onnx` ignores it; configure the store with `FloudsOnnxSettings`. |

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

Each model directory contains a `manifest.json` with a schema version, the
model name, model type, the export options that were actually used, the selected
runtime strategy, discovered ONNX files, and model config. It is written
atomically, after the export has been checked.

## Development

```bash
pip install -e ".[dev,export,seq2seq,server]"
pre-commit install
pytest tests/unit -v --cov=flouds_onnx
```

Install in editable mode (`-e`) so that scripts and the `flouds-onnx` command
use your working copy. A plain `pip install .` leaves a snapshot in
`site-packages` that silently goes stale.

Useful checks:

```bash
black --check src tests
isort --check-only src tests
flake8 src tests
bandit -r src -ll
mypy src/flouds_onnx
pyright src/flouds_onnx
python -m build
twine check dist/*
```

Integration tests download and export real models:

```bash
pytest tests/integration -m integration -v
```

Runnable examples are in [`examples/`](https://github.com/gmalakar/flouds-onnx/tree/main/examples), and manual smoke scripts for
all model types are in [`scripts/`](https://github.com/gmalakar/flouds-onnx/tree/main/scripts).

## Release

Releases are tag-driven through GitHub Actions. See
[`docs/RELEASE_PROCESS.md`](https://github.com/gmalakar/flouds-onnx/blob/main/docs/RELEASE_PROCESS.md).

## License

Apache-2.0. See [`LICENSE`](https://github.com/gmalakar/flouds-onnx/blob/main/LICENSE).
