"""Seasonality of postfire debris flows in the western United States.

Analysis code for the PFDF seasonality project. Bulk data and generated
figures live outside this repository; see :mod:`pfdf_seasonality.paths`.

    from pfdf_seasonality import paths, seasons, style
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import paths, seasons, style  # noqa: F401

__all__ = ["paths", "seasons", "style", "atlas14", "__version__"]
