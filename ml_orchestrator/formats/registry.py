import importlib
import pkgutil
from pathlib import Path

from ml_orchestrator.formats.base import BaseFormat

_registry: dict[str, BaseFormat] = {}


def _discover() -> None:
    """
    Auto-discover all BaseFormat subclasses in the formats package.
    Imports every module in this package so subclasses register themselves.
    """
    import ml_orchestrator.formats as formats_pkg

    for _, module_name, _ in pkgutil.iter_modules(formats_pkg.__path__):
        if module_name not in ("base", "registry"):
            importlib.import_module(f"ml_orchestrator.formats.{module_name}")

    for cls in BaseFormat.__subclasses__():
        instance = cls()
        for ext in instance.extensions:
            _registry[ext.lower()] = instance


def get_format(path: Path) -> BaseFormat | None:
    """Return the format handler for the given file path, or None if unsupported."""
    if not _registry:
        _discover()
    return _registry.get(path.suffix.lower())


def supported_extensions() -> list[str]:
    """Return all registered file extensions."""
    if not _registry:
        _discover()
    return list(_registry.keys())


def all_formats() -> list[BaseFormat]:
    """Return one instance of each registered format (deduplicated)."""
    if not _registry:
        _discover()
    seen = set()
    result = []
    for fmt in _registry.values():
        if fmt.name not in seen:
            seen.add(fmt.name)
            result.append(fmt)
    return result
