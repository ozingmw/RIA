from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROMPT_PATH = Path(__file__).with_name("prompt.json")


def load_prompt() -> Dict[str, Any]:
    with PROMPT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_prompt(data: Dict[str, Any]) -> None:
    tmp_path = PROMPT_PATH.with_name(f"{PROMPT_PATH.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    tmp_path.replace(PROMPT_PATH)


def build_system_prompt(data: Dict[str, Any]) -> str:
    name = data.get("name", "")
    style = data.get("style", "")
    lines: List[str] = []
    if name and style:
        lines.append(f'\ub108\ub294 "{name}". {style}')
    elif name:
        lines.append(f'\ub108\ub294 "{name}".')
    elif style:
        lines.append(style)

    rules = data.get("rules") or []
    if rules:
        lines.append("\uaddc\uce59:")
        lines.extend([f"- {rule}" for rule in rules])

    samples = data.get("samples") or []
    if samples:
        lines.append("\uc608\uc2dc:")
        assistant_name = name or "\uc5b4\uc2dc\uc2a4\ud134\ud2b8"
        for sample in samples:
            user = sample.get("user", "")
            assistant = sample.get("assistant", "")
            lines.append(f'\uc720\uc800: "{user}"')
            lines.append(f'{assistant_name}: "{assistant}"')

    notes = data.get("notes")
    if notes:
        lines.append(f"\uba54\ubaa8: {notes}")

    return "\n".join(lines)


def build_prompt_text() -> str:
    return build_system_prompt(load_prompt())


def get_prompt() -> Dict[str, Any]:
    return load_prompt()


def update_prompt(
    name: Optional[str] = None,
    style: Optional[str] = None,
    rules_set: Optional[List[str]] = None,
    rules_append: Optional[List[str]] = None,
    samples_append: Optional[List[Dict[str, str]]] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    data = load_prompt()
    if name is not None:
        data["name"] = name
    if style is not None:
        data["style"] = style
    if rules_set is not None:
        data["rules"] = rules_set
    if rules_append:
        data.setdefault("rules", []).extend(rules_append)
    if samples_append:
        data.setdefault("samples", []).extend(samples_append)
    if notes is not None:
        data["notes"] = notes
    save_prompt(data)
    return data
