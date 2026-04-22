#!/usr/bin/env python3
"""CLI wrapper for data preparation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xai_project.prepare_data import main


if __name__ == "__main__":
    main()

