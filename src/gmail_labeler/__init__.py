"""Strictly-local Gmail auto-labeling. See README.md and AGENTS.md."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("gmail-labeler")
except PackageNotFoundError:  # source tree without an installed dist
    __version__ = "0.0.0+unknown"
