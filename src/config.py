"""
Global project configuration.

This module holds top-level settings shared across all src sub-packages.
Each sub-package may define its own config dataclass (e.g.
``src.detection.config.DetectorConfig``) for module-specific values.

Values here should only be things that are truly cross-cutting:
  - project metadata
  - log level defaults
  - shared paths

NOT implemented yet — add fields as the project grows.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
"""Absolute path to the repository root directory."""

MODELS_DIR: Path = PROJECT_ROOT / "models"
"""Default directory for storing downloaded model weights."""

LOGS_DIR: Path = PROJECT_ROOT / "logs"
"""Default directory for log files."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = "INFO"
"""Default log level for all modules."""

LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
"""Shared log format string."""
