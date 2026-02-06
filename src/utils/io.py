# File: src/utils/io.py
import json
from pathlib import Path
from typing import Any, Dict

def save_json(obj: Dict[str, Any], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
