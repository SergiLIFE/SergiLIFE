"""
Lightweight JSON report writer for validation/benchmark artifacts.

Creates a timestamped JSON file under the given folder and returns the path.
Safe for concurrent invocations in CI.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


def write_json_report(
    report: Dict[str, Any],
    folder: str = "results",
    prefix: str = "report",
    filename: Optional[str] = None,
) -> str:
    """Write a JSON report to disk.

    Args:
        report: Dict-like report content to serialize.
        folder: Folder to write artifacts into. Created if missing.
        prefix: Filename prefix when auto-generating a name.
        filename: Optional explicit filename (relative to folder). If not provided, a
            timestamped filename is generated.

    Returns:
        Absolute file path to the written JSON file.
    """
    os.makedirs(folder, exist_ok=True)
    if filename is None:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
        filename = f"{prefix}_{ts}.json"
    path = os.path.join(folder, filename)
    # Ensure ASCII-safe minimal formatting while preserving readability
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
    return os.path.abspath(path)


__all__ = ["write_json_report"]
