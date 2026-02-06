from __future__ import annotations

import os
from pathlib import Path


def get_example_output_dir(
    script_path: str | Path,
    category: str | None = None,
    *,
    base_env_var: str = "CANNS_EXAMPLES_OUTPUT_DIR",
) -> Path:
    script_path = Path(script_path).resolve()
    base_dir = _resolve_base_output_dir(script_path, base_env_var)
    if category is None:
        category = _infer_category(script_path)
    out_dir = base_dir / category / script_path.stem if category else base_dir / script_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _resolve_base_output_dir(script_path: Path, base_env_var: str) -> Path:
    base_env = os.environ.get(base_env_var)
    if base_env:
        return Path(base_env).expanduser()
    for parent in script_path.parents:
        if parent.name == "examples":
            return parent / "outputs"
    return script_path.parent / "outputs"


def _infer_category(script_path: Path) -> str | None:
    for parent in script_path.parents:
        if parent.name == "examples":
            rel = script_path.relative_to(parent)
            if len(rel.parts) >= 2:
                return rel.parts[0]
            return None
    return None
