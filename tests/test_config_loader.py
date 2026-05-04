from __future__ import annotations

from pathlib import Path

from dl_for_cta.config.loader import load_config


FIXTURES = Path(__file__).parent / "fixtures" / "config_loader"


def test_load_config_without_base_keeps_existing_single_file_behavior() -> None:
    config = load_config(FIXTURES / "single_file.toml")

    assert config.data.symbols == ["a"]
    assert config.outputs.experiment_name == "single_file"
    assert config.project.seed == 42


def test_load_config_merges_base_and_experiment_override() -> None:
    config = load_config(FIXTURES / "experiment.toml")

    assert config.model.hidden_size == 128
    assert config.model.dropout == 0.2
    assert config.outputs.root == "outputs"
    assert config.outputs.experiment_name == "merged"
    assert "base" not in config.raw
    assert config.raw["model"]["hidden_size"] == 128


def test_load_config_replaces_lists_instead_of_appending() -> None:
    config = load_config(FIXTURES / "list_override.toml")

    assert config.data.symbols == ["c"]
