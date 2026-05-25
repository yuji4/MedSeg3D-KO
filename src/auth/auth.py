from __future__ import annotations

import json
from pathlib import Path

CREDS_FILE = Path(__file__).parent.parent.parent / "data" / "credentials.json"

_DEFAULT_CREDS = {"doctor": "medseg2024"}


def _load() -> dict[str, str]:
    if not CREDS_FILE.exists():
        CREDS_FILE.parent.mkdir(exist_ok=True)
        with open(CREDS_FILE, "w", encoding="utf-8") as f:
            json.dump(_DEFAULT_CREDS, f, indent=2)
        return dict(_DEFAULT_CREDS)
    with open(CREDS_FILE, encoding="utf-8") as f:
        return json.load(f)


def check_auth(username: str, password: str) -> bool:
    return _load().get(username) == password


def get_auth_list() -> list[tuple[str, str]]:
    """Gradio demo.launch(auth=...) 에 넘길 (user, pass) 리스트."""
    return list(_load().items())
