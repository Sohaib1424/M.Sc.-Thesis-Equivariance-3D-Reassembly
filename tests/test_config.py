"""Configuration precedence and validation."""
from __future__ import annotations

import pytest

from vngat.config import Config, parse_config


def test_defaults_match_the_project_requirements():
    cfg = Config()
    assert cfg.num_layers == 4
    assert cfg.num_gpus == 2
    assert cfg.save_every == 10
    assert cfg.amp is True


def test_cli_overrides_yaml_and_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    Config(num_layers=6, hidden_channels=128).save_yaml(str(path))
    cfg = parse_config(["--config", str(path), "--num_layers", "8"])
    assert cfg.num_layers == 8          # CLI wins
    assert cfg.hidden_channels == 128   # YAML wins over the default


def test_boolean_flags_parse():
    assert parse_config(["--amp", "false"]).amp is False
    assert parse_config(["--amp", "true"]).amp is True


def test_validate_rejects_indivisible_hidden_width():
    with pytest.raises(ValueError, match="divisible"):
        Config(hidden_channels=10, heads=4).validate()


def test_validate_rejects_micro_batch_that_does_not_divide():
    with pytest.raises(ValueError, match="multiple of micro_batch_scenes"):
        Config(batch_size=3, micro_batch_scenes=2).validate()


def test_unknown_yaml_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown config keys"):
        Config.from_dict({"not_a_field": 1})


def test_roundtrip_through_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    original = Config(tag="x", lr=1e-3, input_source="frac")
    original.save_yaml(str(path))
    assert Config.from_yaml(str(path)) == original
