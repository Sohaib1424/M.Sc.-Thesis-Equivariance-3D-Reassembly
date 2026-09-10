"""Configuration precedence and validation."""
from __future__ import annotations

import pytest

from pathlib import Path

from vngat.config import Config, parse_config


def test_defaults_match_the_project_requirements():
    cfg = Config()
    assert cfg.num_layers == 4
    assert cfg.num_gpus == 2
    assert cfg.save_every == 10


def test_mixed_precision_is_off_by_default():
    """
    fp32, not AMP -- a measured choice, not a stylistic one.

    Same seed and configuration on 8 objects, differing only in precision:
    fp32 reached 43.18 deg in 90 epochs against AMP's 54.26 in 160, and AMP
    additionally produced non-finite losses from epoch 66 on six of the eight
    training objects, excluding them from training. The gradient corruption
    preceded the visible NaN -- fp32 was already 5-19 deg ahead at epochs 62-65.

    If this assertion is ever flipped back, that comparison should be re-run
    first.
    """
    assert Config().amp is False


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


def test_yaml_configs_agree_with_the_dataclass_defaults_on_precision():
    """
    Catch a default and its shipped configs drifting apart.

    When `amp` was flipped to False, a test asserting the old value survived and
    failed only at the next pytest run -- and a YAML left at the old value would
    not have failed at all, it would just have trained differently from what the
    dataclass documents. This checks the shipped configs directly.
    """
    import yaml

    root = Path(__file__).resolve().parent.parent
    for path in sorted((root / "configs").glob("*.yaml")):
        values = yaml.safe_load(path.read_text())
        assert values["amp"] == Config().amp, (
            f"{path.name} sets amp={values['amp']} but the Config default is "
            f"{Config().amp}; see the docstring for the measurement behind it"
        )


def test_parse_config_records_which_flags_were_typed():
    """A resume restores everything else from the checkpoint, so it has to know
    what the caller actually asked for."""
    cfg = parse_config(["--lr", "5e-4", "--epochs", "80"])
    assert "lr" in cfg._explicit and "epochs" in cfg._explicit
    assert "lr_min" not in cfg._explicit
    assert "hidden_channels" not in cfg._explicit


def test_architecture_is_both_restored_and_checked():
    """
    Architecture is a SUBSET of the restored fields, not a separate category.

    Restored, so a resume needs no `--hidden_channels 128` re-typed; also
    checked, so an EXPLICIT mismatch fails loudly instead of hitting a shape
    error inside load_state_dict. An earlier version only checked, which made a
    bare resume crash against the default architecture.
    """
    from vngat.training.trainer import _ARCHITECTURE_FIELDS, _RESTORED_FIELDS

    assert set(_ARCHITECTURE_FIELDS) <= set(_RESTORED_FIELDS)
    names = {f.name for f in __import__("dataclasses").fields(Config)}
    assert set(_RESTORED_FIELDS) <= names


def test_the_data_definition_is_restored():
    """The worst silent failure available: a resume that quietly changes the
    train/val split leaks validation objects into training."""
    from vngat.training.trainer import _RESTORED_FIELDS

    for name in ("split_source", "data_subsets", "steps_per_epoch", "val_steps",
                 "max_scenes", "input_source", "val_frac", "split_seed"):
        assert name in _RESTORED_FIELDS, name
