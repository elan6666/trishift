"""TriShift package."""

from importlib.metadata import PackageNotFoundError, version

from .TriShift import TriShift
from .TriShiftData import TriShiftData

try:
    __version__ = version("trishift")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__", "TriShift", "TriShiftData"]
