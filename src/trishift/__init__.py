"""TriShift package."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .TriShift import TriShift
    from .TriShiftData import TriShiftData

try:
    __version__ = version("trishift")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__", "TriShift", "TriShiftData"]


def __getattr__(name: str) -> Any:
    if name == "TriShift":
        return import_module(".TriShift", __name__).TriShift
    if name == "TriShiftData":
        return import_module(".TriShiftData", __name__).TriShiftData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
