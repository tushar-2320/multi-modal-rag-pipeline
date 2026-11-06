import re
import tomllib
from pathlib import Path

FIXED_VERSION = (13, 0, 0)
ROOT = Path(__file__).resolve().parents[1]


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def test_pillow_dependency_requires_fixed_version() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pillow_specs = [
        dependency
        for dependency in project["project"]["dependencies"]
        if dependency.lower().startswith("pillow")
    ]

    assert len(pillow_specs) == 1, pillow_specs
    match = re.fullmatch(r"pillow>=(\d+(?:\.\d+)*)", pillow_specs[0].lower())
    assert match is not None, pillow_specs[0]
    assert _version_tuple(match.group(1)) >= FIXED_VERSION, pillow_specs[0]


def test_pillow_lockfile_is_fixed_version() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    pillow_entries = [
        package for package in lock["package"] if package["name"] == "pillow"
    ]

    assert len(pillow_entries) == 1, pillow_entries
    assert _version_tuple(pillow_entries[0]["version"]) >= FIXED_VERSION, (
        pillow_entries[0]["version"]
    )
