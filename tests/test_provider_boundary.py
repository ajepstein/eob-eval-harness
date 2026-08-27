"""The thesis, enforced mechanically.

The project's claim is that the model is a swappable component: nothing
outside `harness/adapters/` knows which provider ran. The plan checks this
with `grep -ril "anthropic\\|openai\\|together" harness/`, which was fine
until a provider was named "together" — an ordinary English word that
appears in prose like "fields fail together". A grep over all text now
reports false positives.

These tests check the property the grep was proxying for: no module outside
the adapter package *imports* a provider SDK or references a
provider-specific symbol. That is both stricter (a stray mention in a
comment cannot break the build) and more meaningful (an actual dependency
cannot slip through in a docstring).
"""

import ast
from pathlib import Path

import pytest

PROVIDER_PACKAGES = {"anthropic", "openai", "together", "groq", "cohere", "mistralai"}
ADAPTER_DIR = Path("harness/adapters")
# config.py legitimately names models and prices per provider; it is the one
# place outside adapters/ that is allowed to know they exist.
ALLOWED_OUTSIDE = {Path("harness/config.py")}


def _modules() -> list[Path]:
    return sorted(
        p for p in Path("harness").rglob("*.py") if "__pycache__" not in p.parts
    )


def _imported_packages(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_no_module_outside_adapters_imports_a_provider_sdk():
    offenders = {}
    for path in _modules():
        if ADAPTER_DIR in path.parents or path in ALLOWED_OUTSIDE:
            continue
        leaked = _imported_packages(path) & PROVIDER_PACKAGES
        if leaked:
            offenders[str(path)] = sorted(leaked)

    assert offenders == {}, f"provider SDKs imported outside adapters/: {offenders}"


def test_config_imports_no_provider_sdk_either():
    # config.py may *name* providers in strings, but importing one would
    # make every consumer of config depend on that SDK being installed.
    assert _imported_packages(Path("harness/config.py")) & PROVIDER_PACKAGES == set()


def test_the_runner_imports_only_the_adapter_protocol():
    imported = _imported_packages(Path("harness/runner.py"))

    assert imported & PROVIDER_PACKAGES == set()
    assert "harness" in imported  # it does use harness.adapters.base


def test_each_adapter_imports_exactly_one_provider_sdk():
    # An adapter that reached for a second provider's SDK would be a sign
    # the abstraction had started leaking sideways.
    for path in sorted(ADAPTER_DIR.glob("*.py")):
        if path.name in ("__init__.py", "base.py"):
            continue
        used = _imported_packages(path) & PROVIDER_PACKAGES
        assert len(used) == 1, f"{path} imports {sorted(used)}"


def test_the_registry_imports_providers_lazily():
    # Module-level imports would mean selecting one provider requires every
    # other provider's SDK to be installed and keyed.
    tree = ast.parse((ADAPTER_DIR / "__init__.py").read_text())
    module_level = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            module_level.update(n.split(".")[0] for n in names)

    assert module_level & PROVIDER_PACKAGES == set()


def test_adapter_modules_do_not_import_scorers_or_store():
    # Adapters produce a ModelResponse and nothing else; knowing how they
    # are scored or persisted would invert the dependency.
    forbidden = {"harness.scorers", "harness.store", "harness.runner"}
    for path in sorted(ADAPTER_DIR.glob("*.py")):
        text = path.read_text()
        for module in forbidden:
            assert module not in text, f"{path} references {module}"
