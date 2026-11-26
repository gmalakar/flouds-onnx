# flouds-onnx: Production Readiness Audit

Reviewed: 2026-11-21, at commit `753e3cb` (Version 1.0.0) plus the uncommitted changes in
the working tree (`README.md`, `api/routers/models.py`, `tests/unit/test_api_models_router.py`).

Scope: all 1,700 lines under `src/flouds_onnx`, the tests, tooling, packaging and both
GitHub workflows. Everything was read, not sampled.

How to read the evidence. Findings marked **(verified)** were reproduced by running the
code in a temporary directory; nothing in the repository or the real model store was
touched. Findings marked **(by reading)** follow from the code but were not run. The
unit suite was run: 89 passed, 1 skipped (the server test, because `fastapi` is not
installed in the local environment). The integration tests, the HTTP server and a real
export were not run.

---

## 1. Executive Summary

flouds-onnx is a small, tidy library that does one thing: keep exported ONNX models in a
local folder and hand back a loaded ONNX Runtime session. As an in-process library used
by trusted code with trusted model names, it works and the design is sound. It is not
safe to expose to input you do not control, and the optional HTTP server should not be
run anywhere reachable by others.

| Area | Score (0-10) |
|---|---|
| Overall | 4 |
| Architecture | 6 |
| Code quality | 6 |
| Security | 3 (library), 1 (HTTP server) |
| Packaging and CI/CD | 6 |
| Production readiness | 3 |
| Docker | not applicable (no image; correct for a library) |

### Status after the 2.0.0 work

The scores above describe version 1.0.0 as reviewed. The findings below were then
worked through on the `release/2.0.0` branch. Revised scores:

| Area | 1.0.0 | 2.0.0 | What moved it |
|---|---|---|---|
| Overall | 4 | 7 | |
| Architecture | 6 | 7 | One validated path function; exporter output verified and relocated; settings from the environment |
| Code quality | 6 | 7 | Truthful return values and manifest; atomic writes; no private-attribute access across layers |
| Security (library) | 3 | 7 | Path confinement, name-collision refusal, local-only tokenizers |
| Security (HTTP server) | 1 | 6 | Token or local-only access, Host and Origin checks, no remote-code or token fields, sanitised errors |
| Packaging and CI/CD | 6 | 8 | Coverage floor (90% measured), Bandit, pip-audit, build check, release gated on tests |
| Production readiness | 3 | 6 | Thread and provider controls, readiness check, exports off the event loop |

Evidence for the revised scores, all run locally on Windows:

- 241 unit tests pass at 90% coverage, in the development environment and in fresh
  Python 3.11 and 3.12 environments with the newest dependencies the declared ranges
  allow (ONNX Runtime 1.32, FastAPI 0.165). Black, flake8, mypy, pyright and Bandit are
  clean in each.
- All 9 end-to-end tests pass (feature extraction and seq2seq: real download, export,
  load, inference, generation) in a fresh environment with the declared exporter
  version (1.0.1) and CPU PyTorch 2.6.0.
- The call sequence the `vectordb_interface` reranker makes was run against a real
  cross-encoder model and ranked the relevant passage first.
- The wheel and source distribution build, pass `twine check`, and install into a clean
  environment; the installed `flouds-onnx` command works from outside the repository.
- The HTTP server was started from that installed wheel and exercised over real HTTP in
  local mode and token mode: access rules, origin and host checks, traversal attempts,
  rejected fields and error codes all behaved as documented.
- A real pull, load, inference and removal succeeded with a store folder not named
  `models`.

Not verified: the updated GitHub workflows have never run on GitHub; nothing was run on
Linux or macOS, with a GPU provider, or under several server processes sharing one
store; and no load or memory test was done.

Status of the top ten:

| # | Issue | Status |
|---|---|---|
| 1 | `model_for` path traversal | Fixed |
| 2 | Model name `..` deletes the store | Fixed |
| 3 | Unauthenticated HTTP server with remote-code export | Fixed |
| 4 | Models from different organisations collide | Mitigated: the second model is refused, not served. The store still has one folder per short name, so the two cannot coexist |
| 5 | Custom `onnx_path` breaks exports | Fixed |
| 6 | Eviction matches by name prefix | Fixed |
| 7 | Per-thread tokenizer cache: stale after reload, unbounded | Fixed (still per thread, by design; now bounded and invalidated everywhere) |
| 8 | Concurrent exports share `ONNX_PATH` | Fixed: the store root is now an argument to the exporter and the environment is not touched. Exports in one process still take turns, because the exporter changes process-wide state while it runs |
| 9 | Blocking work on the event loop | Fixed |
| 10 | Silent Hub download for tokenizers | Fixed: opt-in |

Refactoring done after the fixes: the export options are defined once (`ExportOptions`) instead of in five modules; the never-used decoder cache, two exceptions and two methods are removed; and `pyproject.toml` is the only place for dependencies and for the mypy and pyright settings.

Still open, and why the scores are not higher:

- **Store identity.** The revision is recorded when one is requested, and a pull for a
  different revision is refused instead of answered with the stored one. No file hashes
  are checked, so a modified model file is loaded without question (S6), and two
  variants of one model still cannot coexist (A1).
- **No locking between processes.** Two processes pulling the same model into one
  store still race. Run one writer, or one worker.
- **Cache limits are counts, not memory** (A5).
- **`ModelConfig` still carries many fields nothing reads**, and the library still
  stops at a raw session (A6).
- **Server:** no rate or size limits, no request logging or metrics, no TLS. A caller
  with the token can still fill the disk or memory.
- ~~**`numpy<1.27`** forces NumPy 1.x on consumers~~ - fixed 2026-11-26: `numpy>=1.26.4,<3`,
  tested with NumPy 2.5.5 (Python 3.14) and 2.4.6 (Python 3.11).
- **The per-model lock tables only grow** (A5). Negligible unless model names are
  unbounded.

The four things that matter most:

- **A model name or model type can delete files outside the store.** `remove("..")`
  deletes the entire model store, and a crafted `model_for` deletes any directory the
  process can write to. Through the HTTP server this needs no credentials.
- **The HTTP server has no authentication** and exposes export with
  `trust_remote_code`, which runs code downloaded from the Hugging Face Hub.
- **Two different Hub models share one folder.** `org-a/bert-base` and `org-b/bert-base`
  are the same model as far as the store is concerned, so a caller can be given a
  different model from the one they named.
- **The store path setting only works if the folder is called `models`.** With any other
  name the export lands somewhere else and the model can never be loaded.

What is good: clear layering, lazy optional dependencies, a real unit suite, a CI matrix
on two operating systems and two Python versions, and a release workflow that uses PyPI
Trusted Publishing and checks the tag against the version.

---

## 2. Top 10 Critical Issues

| # | Severity | Issue | Where | Evidence |
|---|---|---|---|---|
| 1 | Critical | `model_for` is joined into the store path with no validation; `remove`, `pull(force=True)` and `get` can reach any directory | `store/registry.py:171-172`, `:70`, `:147` | verified |
| 2 | Critical | A model name of `..` or `.` survives the folder sanitiser; `remove("..")` deletes the whole store | `utils/path_guard.py:13`, `:25` | verified |
| 3 | Critical | HTTP server: no authentication, no authorisation, and remote-code export reachable by anyone who can connect | `api/routers/models.py:30`, `:54`, `:81`, `:136` | by reading |
| 4 | High | Model identity is the last path segment only, so models from different organisations collide | `utils/path_guard.py:25` | verified |
| 5 | High | A custom `onnx_path` not named `models` sends the export to a different directory from the manifest | `store/exporter_bridge.py:79` | verified |
| 6 | High | Evicting or checking one model also matches any model whose folder name starts the same | `runtime/session_pool.py:81-83`, `runtime/loader.py:173` | verified |
| 7 | High | Tokenizers are cached per thread: eviction and reload do not reach other threads, and memory grows with thread count | `runtime/tokenizer_cache.py:39` | verified |
| 8 | High | Export changes the process-wide `ONNX_PATH` variable; concurrent exports of different models interfere | `store/exporter_bridge.py:81`, `:121` | by reading |
| 9 | Medium | Every HTTP handler is `async` but calls blocking work; one export freezes the server, including `/health`, for minutes | `api/routers/models.py:43-141` | by reading |
| 10 | Medium | The tokenizer loader silently downloads from the Hub, unpinned, when a local load fails | `runtime/tokenizer_cache.py:30`, `:100` | by reading |

**Update (first fixes, uncommitted):** issues 1, 2 and 6 are fixed. Model names and model types are validated (`utils/path_guard.py`), every store path is built by one function that confines it to the store (`ModelRegistry._model_dir`), and eviction and `is_loaded` match one model directory instead of a name prefix. A bad name or type raises `InvalidModelReferenceError`, which the HTTP server reports as 400. Covered by `tests/unit/test_store_confinement.py`. Everything else in this table is still open; in particular the server still has no authentication (issue 3).

**Second update (server, uncommitted):** issues 3 and 9 are fixed. The server requires a bearer token, or without one serves only local requests and refuses a non-loopback bind; it checks `Host` and `Origin`; `trust_remote_code` and `hf_token` are no longer request fields; blocking work runs off the event loop; errors no longer carry paths. Covered by `tests/unit/test_server_security.py`. Still open for the server: it has no rate or size limits, no request logging or metrics, no TLS, a health check that tests nothing, and it still cannot be pointed at a different store or provider (A4). A caller with the token can still fill the disk or memory by pulling and loading models.

A note on the evidence: the local virtualenv has an old copy of the package installed (0.1.0, not editable), so a script that does not put `src` on the path imports that copy. The reproductions were rerun against the committed 1.0.0 source and gave the same results.

Honourable mentions: the manifest records export options that were not the ones used;
`shutil.rmtree(..., ignore_errors=True)` reports success when deletion failed; the
configured execution provider is ignored for seq2seq models; the CLI's `--optimize`
defaults to on while the API defaults to off.

---

## 3. Architecture Review

### What exists

```
api/client.py        FloudsOnnxClient: facade, plus a process-wide default instance
runtime/loader.py    ModelLoader: orchestrates the three below; returns LoadedModel
store/registry.py    ModelRegistry: folders, manifests, pull/remove
store/exporter_bridge.py   lazy call into flouds-model-exporter
runtime/session_pool.py    LRU caches of ORT sessions and seq2seq models
runtime/tokenizer_cache.py per-thread tokenizer cache
api/server.py + routers/   optional FastAPI management API
cli/main.py          argparse CLI over the client
```

The layering is right and the dependencies point one way. Optional packages (exporter,
optimum, FastAPI) are imported lazily and raise a specific, helpful exception when
missing. That part is better than most libraries of this size.

### Problems

**A1. The store has no notion of a model's identity (High).**
A model is identified by `model_for` plus the last segment of its name
(`path_guard.py:25`). The organisation, the revision and the export options are not part
of the key. Consequences:

- `org-a/bert-base` and `org-b/bert-base` collide (verified: `exists("org-b/bert-base")`
  returns true and the manifest returned is `org-a`'s).
- The same model exported with and without optimisation, or at two revisions, cannot
  coexist.
- `hf_revision` exists in the manifest but is never written, so there is no record of
  what was actually downloaded.

A store that is described as "Ollama-style" needs a real key: full repository ID,
revision and an export-options digest.

**A2. Path building is not centralised (Critical).**
`safe_join` exists and is used for the ONNX file name inside a model directory, which is
the least dangerous place. The model directory itself is built with a plain
`models_root / model_for / folder` (`registry.py:172`). There should be exactly one
function that turns `(model_for, model_name)` into a directory, and it should refuse
anything that does not resolve to a direct grandchild of the store root.

**A3. The exporter contract is implicit (High).**
`ExporterBridge` derives the exporter's root as "three levels above the model
directory" (`exporter_bridge.py:79`) and communicates it by setting an environment
variable. The exporter always writes to `<root>/models/<model_for>/<folder>`. So the
arrangement only works when the store root is literally named `models`. The README's own
example (`onnx_path="/path/to/onnx/models"`) happens to satisfy that; `/data/store` does
not (verified: the manifest is written to `my-onnx-store/s2s/t5-small` while the exporter
is pointed at the parent of `my-onnx-store`). The exporter already accepts an
`onnx_path` argument; pass it explicitly, and after export verify that ONNX files
actually exist in the model directory before writing a manifest.

**A4. One process-wide default client (Medium).**
`get_default_client()` builds a `FloudsOnnxClient()` with default settings. The HTTP
server and the top-level functions always use it, so the server cannot be pointed at a
different store, provider or cache size. The changelog says environment variables were
removed on purpose; that leaves the server with no configuration at all.

**A5. Cache design (High).**
- Session cache keys are `"<dir>#<provider>"` and eviction matches by string prefix on
  `<dir>` (`session_pool.py:81`). `all-MiniLM-L6` is a prefix of `all-MiniLM-L6-v2`
  (verified: evicting the first removes both). Match on `<dir>#`.
- Caches are bounded by count (5 encoders, 3 seq2seq), not by memory. Three large
  seq2seq models can exhaust RAM while "within the limit".
- The lock dictionaries (`registry.py:174`, `session_pool.py:100`) only ever grow.
- Tokenizers are per thread (`tokenizer_cache.py:39`). In a thread-pool server each
  worker thread holds its own copy of every tokenizer, the copies are never bounded, and
  `evict()` only clears the calling thread (verified). After a reload, other threads keep
  using the old tokenizer with the new model.

**A6. The library stops short of inference (Medium, by design).**
`LoadedModel.run` proxies `session.run`. Tokenising, pooling, normalising and decoding
are left to the caller, yet `ModelConfig` carries about sixty fields for exactly those
steps (`pooling_strategy`, `normalize`, `chunk_logic`, `num_beams`, ...) that nothing in
this package reads. Either provide `encode()`, `rank()` and `generate()` helpers that use
the config, or remove the fields. Carrying unused configuration is how callers come to
believe a setting has an effect.

### Over- and under-engineered

Over: three session caches where one (`decoder`) is never written to; a sixty-field
config copied from another service; the server module pattern of wrapping a whole router
in `try/except ImportError`, which also hides any genuine import error inside it.

Under: no model identity, no atomic writes, no locking between processes, no memory
budget, no revision pinning, no integrity check of what was downloaded.

---

## 4. Code Quality Review

Generally clean: consistent formatting, type hints throughout, small functions, a clear
exception hierarchy, headers and docstrings. The issues are specific.

**Q1. Broad exception handling.** `tokenizer_cache.py:82`, `:91`, `:102` catch
`Exception` three times and fall through to the next attempt; a permissions error or a
corrupt file is reported as "could not load tokenizer" with the cause only in a warning
log. `exporter_bridge.py` wraps every exporter failure, including `KeyboardInterrupt`'s
siblings, as `ExportError` with the cause flattened to a string.

**Q2. Failed operations reported as success.** `registry.py:147` deletes with
`ignore_errors=True` and returns `True`. On Windows a loaded ONNX file is held open, so
`remove()` can report success while the files remain, with the manifest possibly already
gone. `cli/main.py:87` returns exit code 0 when the model was not found.

**Q3. Private attributes used across layers.** `routers/models.py:48` and
`cli/main.py:79` reach `client._loader._registry`; `loader.py:172` reaches into the
pool's private caches; `client.py` returns `self._loader._settings`. Add
`get_manifest()` to the client and `is_cached()` to the pool.

**Q4. Inconsistent defaults for the same option.**

| Option | CLI | Python and HTTP API | Recorded in manifest |
|---|---|---|---|
| `optimize` | on (`cli/main.py:130`) | off | on, always |
| `opset_version` | exporter default | exporter default | 17, always |

`ExportOptions(task=task)` (`registry.py:198`) fills the manifest with defaults instead
of the values used (verified). The manifest is the only record of how a model was built
and it is wrong.

**Q5. Side effects in a property.** `FloudsOnnxSettings.models_root`
(`settings.py:45`) creates the directory every time it is read, including from `list()`
and `exists()`. A read-only query should not create folders, and it makes a read-only
store mount fail on a simple listing.

**Q6. Non-atomic writes.** The manifest is written straight to `manifest.json`
(`manifest.py`). A crash mid-write leaves a truncated file; `pull()` then treats the
model as corrupt and re-exports over the existing directory. Write to a temporary file
and rename. Export into a temporary directory and move it into place on success, so a
failed export never leaves a half-populated model directory.

**Q7. Version in three places.** `pyproject.toml`, `__init__.py` and a hard-coded
`"1.0.0"` in `api/server.py:19`. The bump tool and the release check cover the first two
only. Read the version from package metadata.

**Q8. Repository hygiene.** `test_all_models.py` and `manual_test.py` sit at the
repository root with pytest-style names but are manual scripts. The examples are named
`test_*.py` and edit `sys.path`. `CHANGELOG.md` has an empty `[0.1.0]` heading directly
above `[1.0.0]`. The classifier says "Development Status :: 3 - Alpha" on a 1.0.0
release.

**Q9. Compatibility patch on a third-party object.** `ort_seq2seq.py` sets
`model._supports_cache_class = False` when missing. It is a private attribute of another
library; pin the supported `optimum` range and test it instead.

---

## 5. Security Audit

### Critical

**S1. Arbitrary directory deletion through `model_for` (verified).**
`_model_dir` joins `model_for` unchecked. With `model_for="../.."` and
`model_name="victim"`, `remove()` resolved to a directory outside the store and deleted
it, returning `True`. The same path is used by `pull(force=True)` (`registry.py:70`),
which deletes and then writes, and by `get_manifest`, which reads. Over HTTP:

```
DELETE /api/v1/models/victim?model_for=../..
```

Fix: accept only the five known values (`fe`, `sc`, `ranker`, `s2s`, `llm`), as an enum
at every boundary, and resolve-and-check the final path.

**S2. Store wipe through the model name (verified).**
The sanitiser replaces path separators and reserved characters but not dots.
`safe_model_folder("..")` returns `".."`, so `remove("..", "fe")` deleted `models/` and
everything in it. `"."` deletes the whole `fe` folder. Over HTTP the name arrives through
`{name:path}`, and `org/..` reduces to `..` as well.

Fix: reject names that are empty, `.` or `..`, or that do not match the Hub's
repository-ID pattern (`[A-Za-z0-9][A-Za-z0-9._-]*` per segment, at most two segments).

**S3. The HTTP server is an unauthenticated remote-control interface (by reading).**
There is no authentication, no rate limiting and no request size or time limit. Anyone
who can connect can:

- run code from the Hub: `POST /models/pull` with `trust_remote_code: true` executes the
  model repository's Python on the server;
- fill the disk: pull any number of multi-gigabyte models;
- exhaust memory: `create` large models into the cache;
- delete models, and with S1 and S2 delete anything else;
- read filesystem layout: responses include `model_dir` (`models.py:106`) and raw
  exception text (`models.py:78`).

The default bind address is `127.0.0.1`, which is the only thing standing between this
and the network. That does not protect against other local users, other containers
sharing the network namespace, or a browser on the same machine: there is no CORS or
origin check, so a web page can send `POST` requests to `127.0.0.1:19720`.

Fix: require a token on every route except `/health`; refuse to start on a non-loopback
address without one; remove `trust_remote_code` and `hf_token` from the HTTP request
model entirely; validate `Origin`.

### High

**S4. Model substitution through name collisions (verified).** See A1. If a deployment
pulls `trusted-org/model` and someone later requests `evil-org/model`, they are served
the first. The reverse is the dangerous direction: whoever pulls first decides what
everyone gets for that name.

**S5. Unpinned, silent network download in a "local" code path (by reading).**
`TokenizerCache` falls back to downloading `sentence-transformers/all-MiniLM-L6-v2` or
`sentence-t5-base` from the Hub when a local load fails (`tokenizer_cache.py:100`),
selected by folder name, with no revision. A process that is meant to run offline makes
an outbound request, and gets whatever that repository contains that day. Remove the
fallback; fix the export so the tokenizer is saved correctly.

**S6. No integrity record for model files (by reading).** Nothing records a hash of the
ONNX files or the Hub revision. A modified `model.onnx` or `manifest.json` in the store
is loaded without question. ONNX Runtime will load custom operator libraries only if
asked, which this code does not do, so the direct risk is wrong results and resource
exhaustion, not code execution. Record the revision and a SHA-256 per file in the
manifest and verify on load.

**S7. Secrets on the command line and in requests (by reading).** `--hf-token`
(`cli/main.py:145`) puts the token in shell history and the process list; `hf_token` in
the HTTP body puts it in any request log. Read it from `HUGGINGFACE_HUB_TOKEN` or a file
only.

### Medium

- **Manifest content is trusted.** `ModelConfig` allows extra fields and the file names
  it carries are used to pick the ONNX file. `safe_join` contains that correctly; keep
  it.
- **Error messages leak paths.** Exception text with absolute paths is returned to HTTP
  clients.
- **`**export_kwargs` passes anything through** to the exporter
  (`exporter_bridge.py:108`), so any future exporter option, safe or not, is reachable by
  a Python caller. Acceptable for a library; make it an explicit allowlist.

### OWASP Top 10 mapping (HTTP server)

| Category | Status |
|---|---|
| A01 Broken access control | Fails: no access control at all |
| A03 Injection | Fails: path traversal through name and type |
| A04 Insecure design | Fails: remote code execution is a request field |
| A05 Security misconfiguration | Partial: loopback default is right; no CORS or origin check |
| A06 Vulnerable components | Unknown: no dependency audit in CI |
| A08 Software and data integrity | Fails: no revision or hash for downloaded models |
| A09 Logging and monitoring | Fails: no request or audit logging of destructive actions |
| A10 SSRF | Partial: the server fetches whatever Hub repository a caller names |

---

## 6. Refactoring Recommendations

In priority order.

1. **One path function, strictly validated.** `ModelRegistry._model_dir(model_for, name)`
   takes a `ModelFor` enum and a validated repository ID, resolves the result and asserts
   it is a direct grandchild of the store root. Every caller goes through it.
2. **A real store key.** Folder name from the full repository ID (for example
   `org--name`), with revision recorded; reject or version on collision. Provide a
   migration for existing stores.
3. **Export to a temporary directory, verify, then move into place**, and write the
   manifest atomically with the real export options, the Hub revision and file hashes.
4. **Pass `onnx_path` to the exporter explicitly** and stop touching `os.environ`.
5. **Fix eviction matching** to compare whole keys (`<dir>#`), and add
   `SessionPool.contains(dir)` so `ModelLoader.is_loaded` stops reading private state.
6. **Replace the per-thread tokenizer cache** with a shared, bounded cache. Fast
   tokenizers are safe to share for encoding; if a specific tokenizer is not, lock it.
7. **Server:** token authentication, sync handlers (or `run_in_threadpool`) so exports do
   not block the event loop, typed response models, sanitised errors, and settings read
   from the environment at start-up.
8. **Remove the Hub fallback** in the tokenizer loader.
9. **Decide what the library is.** Either add `encode`, `rank` and `generate` helpers
   that honour `ModelConfig`, or cut `ModelConfig` down to the fields that are used.
10. **Budget the caches by memory**, or at least document that the limits are counts.

---

## 7. Docker Review

There is no Dockerfile, and for a library that is the right choice. Two notes for
consumers that build this into an image:

- NumPy is no longer capped at 1.x (`>=1.26.4,<3` since 2026-11-26).
- The `export` extra pulls PyTorch, Optimum and the exporter. Images that only load
  pre-exported models should install the base package and mount the store read-only,
  which currently fails on first access because of Q5.

---

## 8. Deployment and CI/CD Review

### What is there

- `ci.yml`: Ubuntu and Windows, Python 3.11, 3.12, 3.13 and 3.14; black, isort, flake8, mypy,
  pyright, a dependency-sync check, unit tests. Integration tests run on pushes to
  `main`.
- `release.yml`: checks the tag is semver and matches both version declarations, builds
  with `python -m build`, runs `twine check`, creates a GitHub release and publishes to
  PyPI with Trusted Publishing inside a protected environment.
- Pre-commit: the same checks plus bandit and a private-key detector, with unit tests on
  pre-push.

This is a good base; most projects this size have less.

### Gaps

| Gap | Why it matters | Fix |
|---|---|---|
| Release does not depend on CI | A tag on a commit with failing tests still publishes to PyPI | Run the test job inside the release workflow, or require a passing CI run for the tagged commit |
| No coverage measurement | The server, CLI and failure paths have almost no tests and nothing reports it | `pytest --cov` with a floor |
| Bandit runs only in pre-commit | Contributors who skip hooks skip the security lint | Add it to CI |
| No dependency audit | Known vulnerabilities in `transformers`, `onnxruntime` or FastAPI go unnoticed | `pip-audit` in CI, Dependabot for updates |
| Actions pinned by tag | A moved tag changes what runs with publish rights | Pin `softprops/action-gh-release` and `pypa/gh-action-pypi-publish` by commit SHA |
| Workflow-level `contents: write` | Every job gets write access; only the release job needs it | Default to `contents: read`, grant per job |
| Server extra never installed in CI | The only server test is skipped everywhere | Install `.[server]` in the test job |
| Integration tests only after merge | A break is found on `main`, not in the pull request | Run a small-model subset on pull requests |
| No build check in CI | A packaging error surfaces only at release time | `python -m build` and `twine check` in CI |
| Mixed action versions | `checkout@v4` in CI, `@v5` in release | Align |

### Release process

Trusted Publishing, the environment gate and the version check are all correct. Add:
build once and publish that exact artifact to TestPyPI first; generate the changelog
section from the tag; attach build provenance (`actions/attest-build-provenance`).

---

## 9. Production Readiness Review

Assessed for two uses: as a library inside a service, and as the standalone server.

| Concern | Library | Server |
|---|---|---|
| Input validation | Fails (S1, S2) | Fails |
| Authentication | Not applicable | None |
| Concurrency within a process | Mostly sound; eviction and tokenizer defects | Blocks the event loop |
| Concurrency between processes | None: two processes pulling the same model race on the same directory | Same, with more than one worker |
| Failure handling | Half-written model directories; false success on delete | Every error is a 500 |
| Observability | Standard `logging`, no metrics | No request logs, no metrics, health check tests nothing |
| Resource limits | Count-based caches, no memory or disk budget | No request limits |
| Configuration | Explicit settings object | None |
| Offline operation | Tokenizer fallback reaches the network | Same |
| Upgrade path | No manifest schema version, no store migration | Same |

Specific points:

- **Multiple workers.** Gunicorn or Uvicorn with more than one worker means several
  processes sharing one store with only in-process locks. Two simultaneous pulls of the
  same model write into the same directory. Use a lock file per model directory.
- **Health check.** `/health` returns a constant. It should at least confirm the store
  root is readable.
- **Provider fallback.** Asking for a GPU provider that is unavailable falls back to CPU
  with a warning (`ort_session.py:78`). In production that turns a deployment mistake
  into a silent tenfold slowdown. Make it an error unless fallback is requested. The
  provider setting is also ignored for seq2seq models (`loader.py:202`), which always
  load on Optimum's default.
- **No session options.** Thread counts, graph optimisation level and memory arena
  settings cannot be set, so every session uses all cores. Several models loaded in one
  process will oversubscribe the CPU.
- **Manifest has no schema version.** Any change to `ModelConfig` or the manifest shape
  leaves existing stores to be interpreted by guesswork.

---

## 10. Dependency Review

| Dependency | Declared | Risk | Recommendation |
|---|---|---|---|
| `numpy` | `>=1.26.4,<3` | Low: tested with NumPy 2.5.5 (3.14) and 2.4.6 (3.11) | Fine |
| `onnxruntime` | `>=1.20.1` | Medium: no upper bound on a library with behaviour changes between minors | Add a tested upper bound |
| `transformers` | `>=4.44.0,<4.58.0` | Low: bounded, but the 5.x line is excluded and will need work | Track and test 5.x |
| `pydantic` | `>=2.0` | Low | Fine |
| `flouds-model-exporter` | `>=2.0.2,<3` (extra) | Low: the store root is passed as an argument, the exporter's own errors are translated, and 2.0.2 carries the Python 3.14 optimum fix seq2seq export needs (tested against the local 2.0.2 checkout on Python 3.14 and 3.11 before its release) | Keep the upper bound until 3.0 is tested |
| `optimum[onnxruntime]` | `>=1.22.0` (extra) | Medium: no upper bound, and the code patches a private attribute | Bound it and test the supported range |
| `fastapi`, `uvicorn` | unbounded (extra) | Low | Add upper bounds |
| Dev tools | exact pins (black 25.12.0, isort 6.1.0, flake8 7.5.0, mypy 1.20.2, bandit 1.9.4), updated 2026-11-26 for Python 3.14 | Low | Update on a schedule |

Other observations:

- `requirements-prod.txt` duplicates `pyproject.toml` and a script checks they match.
  One source would remove the need for the check.
- There is no lock file for development, so CI installs whatever is current. A break in
  a transitive dependency will look like a break in this project.
- `requires-python = ">=3.11,<3.15"` (2026-11-26): 3.11 to 3.14, tested on 3.11 and 3.14 locally and
  on all four in CI. Development and the Flouds services use 3.14 (the platform standard).

---

## 11. Roadmap

### Phase 1: make it safe (days)

1. Validate `model_for` as an enum and the model name against a strict pattern;
   resolve-and-check every store path.
2. Fix prefix matching in eviction and `is_loaded`.
3. Remove `trust_remote_code` and `hf_token` from the HTTP request model; add token
   authentication; refuse non-loopback binding without it.
4. Remove the tokenizer Hub fallback.
5. Stop reporting success when deletion fails.
6. Tests for each of the above, including the traversal cases.

### Phase 2: make the store correct (two to three weeks)

1. Store key from the full repository ID and revision; migration for existing stores.
2. Export to a temporary directory, verify, move into place; atomic manifest with real
   options, revision, file hashes and a schema version.
3. Explicit `onnx_path` to the exporter; no environment mutation.
4. Shared, bounded tokenizer cache.
5. Lock file per model for multi-process safety.
6. Coverage, bandit, pip-audit and a build check in CI; release gated on tests.

### Phase 3: make it operable (one to two months)

1. Decide the inference surface: helpers that honour `ModelConfig`, or a smaller config.
2. Memory-aware cache limits and configurable session options.
3. Server configuration from the environment, sync handlers, structured errors, request
   logging, metrics, a meaningful health check.
4. Strict provider selection, applied to seq2seq as well.
5. NumPy 2 support and tested upper bounds on the runtime dependencies.

---

## 12. Target Architecture

```
Caller
  └─ FloudsOnnxClient(settings)            no process-wide singleton required
       ├─ ModelStore                        single owner of the filesystem
       │    ├─ key: (model_for: enum, repo_id, revision, options digest)
       │    ├─ path(key) → validated, resolved, inside root
       │    ├─ pull: export to tmp → verify → hash → atomic move → atomic manifest
       │    └─ lock file per model (cross-process)
       ├─ Exporter (protocol)               explicit arguments, no environment
       ├─ SessionCache                      one cache, keyed by store key, memory budget
       ├─ TokenizerCache                    shared, bounded
       └─ Inference helpers                 encode / rank / generate, driven by config

Optional server (separate, opt-in)
  └─ token auth → sync handlers → client    settings from environment, loopback by default
```

Why: making the store the only code that touches the filesystem turns path safety into
one function to get right and test. A real key removes collisions and makes revisions and
variants possible. Atomic, verified writes mean the store is either correct or untouched.
Taking the environment variable out of the exporter contract removes a whole class of
concurrency and configuration faults.

---

## 13. Quick Wins

Each is an hour or less.

1. Reject `.`, `..` and empty names in `safe_model_folder`.
2. Restrict `model_for` to the five known values in the registry and in the request
   models.
3. Change the eviction prefix to `<dir>#`.
4. Delete the Hub fallback block in `tokenizer_cache.py`.
5. Record the real export options in the manifest.
6. Return the truth from `remove()`: check the directory is gone.
7. Make `cmd_remove` exit non-zero when nothing was removed.
8. Read the server's version from package metadata.
9. Move `models_root.mkdir` out of the property and into `pull`.
10. Add `pip install -e ".[server]"` to the CI test job so the server test runs.
11. Add `needs:` on a test job to the release workflow.
12. Fix the empty `[0.1.0]` heading in the changelog and the "Alpha" classifier.

---

## 14. Things Done Well

- **Layering.** Client, loader, registry, pool and exporter bridge each have one job and
  depend in one direction.
- **Optional dependencies handled properly.** Lazy imports with specific exceptions that
  tell the user which extra to install.
- **`safe_join` is correct**, resolving symlinks and comparing with a separator. It just
  needs to be used for the directory as well as the file.
- **Double-checked locking per key** in the session pool is implemented correctly, so a
  model is loaded once under concurrent requests.
- **Local-only loading by default** (`local_files_only=True`) for both tokenizers and
  seq2seq models, apart from the one fallback.
- **Loopback default** for the server, and the server is an opt-in extra.
- **A real unit suite** (89 tests) that runs in under a minute with no network.
- **CI on two operating systems and two Python versions**, with two type checkers.
- **Release workflow** with Trusted Publishing, an environment gate, and a tag, version
  and `__version__` consistency check.
- **Clear exception hierarchy** exported from the package root.

---

## 15. What I Would Do If I Owned This Codebase

### First 30 days

1. Day 1: ship a patch release with the name and `model_for` validation, the eviction
   fix and honest `remove()`, with tests. These are small and they close the
   file-deletion holes.
2. Week 1: lock down the server: token authentication, no `trust_remote_code` or token
   in requests, sync handlers, sanitised errors. Mark it experimental in the README until
   done.
3. Week 2: remove the tokenizer network fallback; replace the per-thread cache.
4. Weeks 2-3: CI additions: coverage with a floor, bandit, pip-audit, build check, server
   extra installed; gate release on tests.
5. Week 4: explicit exporter arguments, verified export, atomic manifest with the real
   options.

### Days 31-60

1. New store key with repository ID and revision, plus a migration command and a
   manifest schema version.
2. File hashes recorded at pull and verified at load.
3. Cross-process lock per model.
4. NumPy 2 compatibility and tested upper bounds; lock file for development.
5. Tests for the CLI and server, and for failure paths (failed export, corrupt manifest,
   deletion of a loaded model on Windows).

### Days 61-90

1. Decide and build the inference surface, or trim `ModelConfig`.
2. Memory-aware caching and session options.
3. Server operability: configuration, metrics, request logging, a real health check.
4. Strict provider selection across both loaders.
5. Documentation: a threat model for the server, the store layout as a stable contract,
   and an upgrade guide.

The order matters. Nothing after day 1 is worth doing while a model name can delete the
store.
