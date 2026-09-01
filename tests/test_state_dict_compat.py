"""Checkpoint-compatibility tripwire.

Every registered model's state_dict key set is snapshotted in
tests/fixtures/state_dict_keys.json, generated from the code as it stood before
the maintainability refactor. Any change that alters these keys — renaming a
module attribute, adding a parameterised layer to a base class — breaks loading
of every checkpoint users have already saved, so it must fail here first.

If a key change is *intentional* (a new architecture, a deliberate breaking
release), regenerate the fixture:

    python -c "import json, deepecgkit; \
        from deepecgkit.registry import get_model, get_model_names; \
        json.dump({n: sorted(get_model(n)().state_dict()) for n in get_model_names()}, \
                  open('tests/fixtures/state_dict_keys.json', 'w'), indent=2, sort_keys=True)"
"""

import json
from pathlib import Path

import pytest
import torch

from deepecgkit.models.factory import build_model, build_model_from_checkpoint
from deepecgkit.registry import get_model, get_model_names

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "state_dict_keys.json"

with FIXTURE_PATH.open() as f:
    GOLDEN_KEYS = json.load(f)


def test_every_registered_model_is_snapshotted():
    """A newly registered model must be added to the golden fixture."""
    assert sorted(GOLDEN_KEYS) == get_model_names()


@pytest.mark.parametrize("model_name", sorted(GOLDEN_KEYS))
def test_state_dict_keys_unchanged(model_name):
    model = get_model(model_name)()
    assert sorted(model.state_dict().keys()) == GOLDEN_KEYS[model_name], (
        f"state_dict keys for '{model_name}' changed — existing checkpoints "
        "would no longer load. See this module's docstring."
    )


@pytest.mark.parametrize("model_name", ["kanres", "resnet", "gru"])
def test_trainer_checkpoint_round_trip(model_name, tmp_path):
    """A checkpoint in the ECGTrainer save format must rebuild and reload."""
    model = build_model(model_name, input_channels=1, output_size=4)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_config": {},
            "model_config": {"input_channels": 1, "output_size": 4},
            "epoch": 1,
            "best_val_loss": 0.5,
        },
        path,
    )
    restored, checkpoint = build_model_from_checkpoint(path, model_name)
    assert sorted(restored.state_dict().keys()) == sorted(model.state_dict().keys())
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[key])
