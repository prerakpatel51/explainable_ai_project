"""Shared repository path helpers."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"

