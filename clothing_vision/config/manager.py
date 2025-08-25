"""Runtime configuration manager with hot-reload support."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .models import Settings

logger = logging.getLogger(__name__)

Callback = Callable[[Settings], None]


class _ReloadHandler(FileSystemEventHandler):
    """Watchdog event handler that triggers reload on file save."""

    def __init__(self, path: Path, loader: "ConfigManager", debounce: float = 0.2):
        self._path = path.resolve()
        self._loader = loader
        self._debounce = debounce
        self._last_mtime: float = 0

    def on_modified(self, event):  # type: ignore[override]
        if Path(event.src_path).resolve() != self._path:
            return
        mtime = self._path.stat().st_mtime
        if mtime - self._last_mtime < self._debounce:
            return  # ignore rapid consecutive events
        self._last_mtime = mtime
        logger.info("Config file changed; reloading…")
        self._loader.reload()


class ConfigManager:
    """Singleton-style manager that keeps current runtime settings and reloads on change."""

    def __init__(self, config_path: Optional[str | Path] = None, *, watch: bool = True):
        self._path = Path(config_path) if config_path else None
        self._settings: Settings = Settings.load(self._path) if self._path else Settings()
        self._callbacks: list[Callback] = []
        self._lock = threading.RLock()
        self._observer: Optional[Observer] = None

        if watch and self._path is not None:
            self._start_watcher()

    # ------------------------ Public API ------------------------
    def get(self) -> Settings:
        """Return the current immutable settings object."""
        with self._lock:
            return self._settings

    def reload(self) -> None:
        """Force reload settings from disk and notify callbacks if changed."""
        if self._path is None:
            logger.debug("No config file path; skipping reload.")
            return

        try:
            new_settings = Settings.load(self._path)
        except Exception:  # pragma: no cover
            logger.exception("Failed to reload config; keeping previous settings.")
            return

        with self._lock:
            if new_settings == self._settings:
                logger.debug("Settings unchanged after reload.")
                return
            self._settings = new_settings
            callbacks = list(self._callbacks)

        logger.info("Configuration reloaded successfully.")
        for cb in callbacks:
            try:
                cb(new_settings)
            except Exception:
                logger.exception("Error in config reload callback %s", cb)

    def register(self, callback: Callback) -> None:
        """Register a callback executed when settings change."""
        with self._lock:
            self._callbacks.append(callback)

    # ------------------------ Internal helpers ------------------------
    def _start_watcher(self) -> None:
        assert self._path is not None
        handler = _ReloadHandler(self._path, self)
        self._observer = Observer()
        self._observer.schedule(handler, self._path.parent.as_posix(), recursive=False)
        self._observer.daemon = True
        self._observer.start()
        logger.debug("Started config file watcher on %s", self._path)

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None


# ---- Module-level singleton helpers ----
_DEFAULT_MANAGER: Optional[ConfigManager] = None


def _ensure_default() -> ConfigManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = ConfigManager()
    return _DEFAULT_MANAGER


def get_settings() -> Settings:
    """Get current settings from the default manager."""
    return _ensure_default().get()


def set_settings_path(path: str | Path, *, watch: bool = True) -> None:
    """Initialise default manager with explicit config path."""
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is not None:
        _DEFAULT_MANAGER.stop()
    _DEFAULT_MANAGER = ConfigManager(path, watch=watch)
