"""Local installations must not turn fixture tests into live model calls."""
import pytest


@pytest.fixture(autouse=True)
def no_hosted_jev(monkeypatch, tmp_path_factory):
    # An operator's hosted Jev key would turn the delivery gate's judgment into a paid call.
    monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
    # Nor may the operator's Codex home decide a fixture call (its global AGENTS.md is refused).
    monkeypatch.setenv('CODEX_HOME', str(tmp_path_factory.mktemp('codex-home')))
