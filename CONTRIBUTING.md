# Contributing

Thanks for your interest in contributing to flouds-onnx.

## Development setup

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install in editable mode with the development and optional extras
pip install -e ".[dev,export,seq2seq,server]"

# 3. Install pre-commit hooks (REQUIRED — hooks run on every commit)
pre-commit install
pre-commit install --hook-type pre-push

# 4. Verify hooks are working
pre-commit run --all-files
```

Or use the helper script:

```powershell
.\setup_dev.ps1
```

## Local checks

```powershell
pre-commit run --all-files          # format, lint, type-check, bandit
pytest tests/unit/ -q --cov=flouds_onnx --cov-fail-under=85   # unit tests (fast, no network)
pytest -m integration -v            # integration tests (requires HF access)
```

## Pull request checklist

- [ ] `pre-commit run --all-files` passes
- [ ] `pytest tests/unit/ -q` passes
- [ ] New behaviour has tests, and coverage stays at or above 85%
- [ ] Anything that turns a model name or type into a path goes through `ModelRegistry._model_dir`
- [ ] `CHANGELOG.md` updated
- [ ] No `sentence-transformers` added to dependencies
- [ ] Dependency changes are made in `pyproject.toml`, the only place they are listed

## Important rules

- Do **not** install `sentence-transformers` — always pass `library="transformers"` explicitly
- `flouds-model-exporter` must stay in `[export]` optional dep only
- Export options should be passed explicitly through Python/API/CLI calls.
- Use `HUGGINGFACE_HUB_TOKEN` for private or gated Hugging Face models.

## Reporting issues

Include: OS, Python version, exact command, full error output, steps to reproduce.
