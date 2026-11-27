# =============================================================================
# Copyright (c) 2026 Goutam Malakar. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# =============================================================================
from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, List, Optional

from flouds_onnx.config.export_options import ExportOptions
from flouds_onnx.config.model_config import ModelConfig
from flouds_onnx.config.settings import MODEL_FOR_DEFAULT_TASKS, FloudsOnnxSettings
from flouds_onnx.exceptions import (
    ExportDisabledError,
    ExportError,
    InvalidModelReferenceError,
    ManifestError,
    ModelNameConflictError,
    RevisionMismatchError,
)
from flouds_onnx.store.exporter_bridge import ExporterBridge
from flouds_onnx.store.manifest import ExportOptions as ManifestExportOptions
from flouds_onnx.store.manifest import ModelManifest
from flouds_onnx.utils.path_guard import safe_model_folder, safe_model_for

log = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, settings: FloudsOnnxSettings) -> None:
        self._settings = settings
        self._export_locks: dict[str, threading.Lock] = {}
        self._lock_registry = threading.Lock()

    def pull(
        self,
        model_name: str,
        model_for: str = "fe",
        task: Optional[str] = None,
        force: bool = False,
        config: Optional[ModelConfig] = None,
        options: Optional[ExportOptions] = None,
        **option_kwargs: Any,
    ) -> ModelManifest:
        """Make sure the model is in the store, exporting it if it is not.

        Export options come from *options*, from keyword arguments named after
        ``ExportOptions`` fields (``optimize=True``, ``library="transformers"``, ...),
        or both; keywords win. Other keywords are passed to the exporter unchanged.
        """
        folder = safe_model_folder(model_name)
        model_dir = self._model_dir(model_for, folder)
        key = f"{model_for}/{folder}"

        opts = ExportOptions.build(options, task=task, **option_kwargs)
        if opts.task is None:
            # task must not be None — optimum crashes with AttributeError otherwise
            opts = opts.model_copy(update={"task": MODEL_FOR_DEFAULT_TASKS.get(model_for.lower(), "feature-extraction")})

        with self._per_model_lock(key):
            if model_dir.is_dir() and ModelManifest.exists(str(model_dir)):
                if not force:
                    try:
                        manifest = ModelManifest.load(str(model_dir))
                    except ManifestError:
                        log.warning("pull: manifest corrupt for '%s'; re-exporting", key)
                    else:
                        # A different model that happens to share the folder name must
                        # not be handed back as if it were the one asked for.
                        self._check_same_model(model_name, manifest)
                        self._check_same_revision(model_name, opts.revision, manifest)
                        log.debug("pull: '%s' already on disk; skipping export", key)
                        return manifest
                else:
                    self._require_export_allowed(model_name, model_for)
                    log.info("pull: force=True; removing '%s' before re-export", key)
                    shutil.rmtree(str(model_dir), ignore_errors=True)

            # Checked before anything on disk changes: with export off a missing model is
            # an error and nothing is created or removed.
            self._require_export_allowed(model_name, model_for)

            os.makedirs(str(model_dir), exist_ok=True)
            log.info(
                "pull: exporting '%s' (model_for=%s, task=%s, optimize=%s, library=%s, normalize_embeddings=%s)",
                model_name,
                model_for,
                opts.task,
                opts.optimize,
                opts.library,
                opts.normalize_embeddings,
            )

            try:
                ExporterBridge.export(model_name=model_name, model_dir=str(model_dir), model_for=model_for, force=force, **opts.as_kwargs())
                self._collect_export_output(model_for, folder, model_dir)
            except BaseException:
                # Includes ExportInProgressError, raised when another process is already
                # exporting this model. Only empty folders are removed, so that
                # process's files are never touched.
                self._remove_if_empty(model_dir)
                raise

            manifest = self._build_manifest(model_name, model_for, folder, model_dir, ManifestExportOptions(**opts.manifest_record()), config)
            manifest.hf_revision = opts.revision
            if not manifest.onnx_files:
                # Without this check a failed or misdirected export would be recorded as
                # a model that can never be loaded.
                raise ExportError(f"Export of '{model_name}' produced no ONNX files in '{model_dir}'")
            manifest.save(str(model_dir))
            log.info("pull: manifest written for '%s'", key)
            return manifest

    def _require_export_allowed(self, model_name: str, model_for: str) -> None:
        """Raise ExportDisabledError when export is switched off."""
        if not self._settings.allow_export:
            raise ExportDisabledError(
                f"Model '{model_name}' (model_for={model_for}) must be exported, but export is disabled "
                "(allow_export=False / FLOUDS_ONNX_ALLOW_EXPORT=false). Put the exported model in the store, "
                "or enable export and install flouds-onnx[export]."
            )

    def list(self) -> List[ModelManifest]:
        manifests: List[ModelManifest] = []
        models_root = self._settings.models_root
        if not models_root.exists():
            return manifests
        for model_for_dir in models_root.iterdir():
            if not model_for_dir.is_dir():
                continue
            for model_dir in model_for_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                if ModelManifest.exists(str(model_dir)):
                    try:
                        manifests.append(ModelManifest.load(str(model_dir)))
                    except ManifestError as exc:
                        log.warning("list: skipping '%s': %s", model_dir, exc)
        return manifests

    def remove(self, model_name: str, model_for: str = "fe") -> bool:
        """Delete a model's directory. True only if it existed and is now gone."""
        folder = safe_model_folder(model_name)
        model_dir = self._model_dir(model_for, folder)
        key = f"{model_for}/{folder}"
        if not model_dir.exists():
            return False
        with self._per_model_lock(key):
            if ModelManifest.exists(str(model_dir)):
                try:
                    self._check_same_model(model_name, ModelManifest.load(str(model_dir)))
                except ManifestError:
                    pass  # an unreadable manifest must not make the directory undeletable
            shutil.rmtree(str(model_dir), ignore_errors=True)
            if model_dir.exists():
                # Typically Windows refusing to delete an ONNX file that is still open.
                log.warning("remove: could not fully delete '%s'; a file may still be in use", model_dir)
                return False
            log.info("remove: deleted '%s'", model_dir)
            return True

    def get_manifest(self, model_name: str, model_for: str = "fe") -> Optional[ModelManifest]:
        """Manifest for *model_name*, or None if it is not stored.

        Raises ModelNameConflictError if the folder holds a different model of the same
        short name.
        """
        folder = safe_model_folder(model_name)
        model_dir = self._model_dir(model_for, folder)
        if not ModelManifest.exists(str(model_dir)):
            return None
        try:
            manifest = ModelManifest.load(str(model_dir))
        except ManifestError as exc:
            log.warning("get_manifest: cannot read '%s/%s': %s", model_for, folder, exc)
            return None
        self._check_same_model(model_name, manifest)
        return manifest

    def exists(self, model_name: str, model_for: str = "fe") -> bool:
        """True if *model_name* itself is stored (not merely a model sharing its folder)."""
        folder = safe_model_folder(model_name)
        model_dir = self._model_dir(model_for, folder)
        if not (model_dir.is_dir() and ModelManifest.exists(str(model_dir))):
            return False
        try:
            return self._same_model(model_name, ModelManifest.load(str(model_dir)).model_name)
        except ManifestError:
            return True  # present but unreadable: pull() decides what to do with it

    def model_dir_path(self, model_name: str, model_for: str = "fe") -> str:
        folder = safe_model_folder(model_name)
        return str(self._model_dir(model_for, folder))

    def _model_dir(self, model_for: str, folder: str) -> Path:
        """The directory for one model: ``<models_root>/<model_for>/<folder>``.

        Both parts are validated, and the normalised result must be exactly two levels
        below the store root. Model directories are created, read and deleted, so this
        is the one place that guarantees none of those can happen outside the store.

        The comparison is on the normalised path, without following symbolic links, so
        a model folder an operator has linked to another disk keeps working.
        """
        root = self._settings.models_root
        candidate = root / safe_model_for(model_for) / safe_model_folder(folder)
        if Path(os.path.abspath(candidate)).parent.parent != Path(os.path.abspath(root)):
            raise InvalidModelReferenceError(f"model_for={model_for!r} / {folder!r} does not resolve inside the model store")
        return candidate

    def _per_model_lock(self, key: str) -> threading.Lock:
        with self._lock_registry:
            if key not in self._export_locks:
                self._export_locks[key] = threading.Lock()
            return self._export_locks[key]

    @staticmethod
    def _same_model(requested: str, stored: str) -> bool:
        """Whether two names refer to the same model.

        The store keeps one folder per short name ("bert-base"), so "org-a/bert-base"
        and "org-b/bert-base" would share it. They are different models only when both
        names carry an organisation and the full names differ; a bare short name
        addresses whatever is in the folder, as it always has.
        """
        a, b = requested.strip().strip("/").lower(), stored.strip().strip("/").lower()
        return a == b or "/" not in a or "/" not in b

    @classmethod
    def _check_same_model(cls, requested: str, manifest: ModelManifest) -> None:
        if not cls._same_model(requested, manifest.model_name):
            raise ModelNameConflictError(
                f"'{requested}' shares its folder with '{manifest.model_name}', which is already stored.",
                detail="Remove the stored model first, or pull with force=True to replace it.",
            )

    @staticmethod
    def _check_same_revision(requested_name: str, revision: Optional[str], manifest: ModelManifest) -> None:
        """A pull that names a revision must not be answered with another revision."""
        if revision is not None and revision != manifest.hf_revision:
            stored = manifest.hf_revision or "the default branch (no revision recorded)"
            raise RevisionMismatchError(
                f"'{requested_name}' is stored from {stored}, not revision '{revision}'.",
                detail="Pull with force=True to replace it with the requested revision.",
            )

    def _collect_export_output(self, model_for: str, folder: str, model_dir: Path) -> None:
        """Bring the exporter's output into *model_dir* when it wrote somewhere else.

        The exporter always writes to ``<root>/models/<model_for>/<folder>``, where
        ``<root>`` is three levels above the model directory. That is the model
        directory itself only when the store root is named "models". With any other
        store root (``onnx_path="/data/store"``) the files land beside the store, and
        are moved in here.
        """
        exported = Path(ExporterBridge.onnx_root_for(str(model_dir))) / "models" / model_for / folder
        if os.path.abspath(exported) == os.path.abspath(model_dir) or not exported.is_dir():
            return
        for item in exported.iterdir():
            target = model_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        # Remove the now-empty folders the exporter created, innermost first.
        for leftover in (exported, exported.parent, exported.parent.parent):
            try:
                leftover.rmdir()
            except OSError:
                break
        log.info("pull: moved export output from '%s' into '%s'", exported, model_dir)

    @staticmethod
    def _remove_if_empty(model_dir: Path) -> None:
        """Drop the model folder, and its model-type folder, if a failed pull left them
        empty. ``rmdir`` refuses a folder that has anything in it."""
        for folder in (model_dir, model_dir.parent):
            try:
                folder.rmdir()
            except OSError:
                return

    @staticmethod
    def _build_manifest(
        model_name: str,
        model_for: str,
        folder: str,
        model_dir: Path,
        export_options: ManifestExportOptions,
        config: Optional[ModelConfig],
    ) -> ModelManifest:
        from flouds_onnx.runtime.strategy import resolve_strategy

        cfg = config or ModelConfig(model_name=model_name, model_for=model_for)
        strategy = resolve_strategy(cfg)
        manifest = ModelManifest(
            model_name=model_name,
            model_for=model_for,
            folder=folder,
            session_strategy=strategy.value,
            export_options=export_options,
            config=cfg,
        )
        manifest.scan_onnx_files(str(model_dir))
        return manifest
