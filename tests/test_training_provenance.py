import json
import inspect

import pytest

from avcs.training.train_kaggle import build_parser, main
from avcs.training.train_ui_grounder import train_ui_grounder


def test_synthetic_trainer_requires_explicit_acknowledgment(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2
    assert "synthetic colored blocks only" in capsys.readouterr().err


def test_synthetic_trainer_defaults_to_separate_output_directory():
    args = build_parser().parse_args(["--synthetic-data-only"])

    assert args.output_dir == "runs/synthetic_cortex_2m"


def test_ui_grounder_trainer_rejects_non_synthetic_manifest(tmp_path):
    manifest = tmp_path / "recorded-demo-manifest.json"
    manifest.write_text(
        json.dumps({"dataset_provenance": "recorded_ui_click_demonstrations"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only accepts the procedural synthetic UI dataset"):
        train_ui_grounder(manifest_path=str(manifest))


def test_ui_grounder_trainer_uses_separate_synthetic_checkpoint_directory():
    default_output = inspect.signature(train_ui_grounder).parameters["output_dir"].default

    assert default_output == "runs/synthetic_ui_grounder"
