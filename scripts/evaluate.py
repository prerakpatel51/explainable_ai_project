#!/usr/bin/env python3
"""CLI wrapper for evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xai_project.evaluate import main


if __name__ == "__main__":
    main()

