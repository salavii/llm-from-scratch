"""Regression tests for the bugs that made this repo report wrong numbers.

Each test here maps to a specific defect that shipped:

  test_initial_loss_matches_uniform_baseline  -> the misplaced `return` in
      calc_loss_loader, which quietly reported pretraining loss as ~1.2
      instead of ~11 and made the Ch5 overfitting table wrong.
  test_calc_loss_loader_averages_*            -> the same bug, pinned exactly.
  test_layernorm_*                            -> sqrt(var - eps) instead of
      sqrt(var + eps), which yields NaN when variance is small.
  test_causal_mask_excluded_from_state_dict   -> the mask buffer being saved
      into every checkpoint.
"""
import math

import pytest
import torch

from Attention import (
    GPTModel,
    LayerNorm,
    calc_loss_batch,
    calc_loss_loader,
    load_gpt_state_dict,
)

VOCAB_SIZE = 50257

TINY_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "context_length": 16,
    "emb_dim": 32,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


@pytest.fixture
def tiny_model():
    torch.manual_seed(123)
    return GPTModel(TINY_CONFIG)


def test_initial_loss_matches_uniform_baseline(tiny_model):
    """A freshly initialised model must score about ln(vocab_size).

    Before any training the output distribution is essentially uniform over the
    vocabulary, so cross-entropy has to land near ln(50257) = 10.82. This is the
    single cheapest sanity check on the whole training pipeline: the loss-
    averaging bug showed up here as 1.22 instead of 10.99.
    """
    torch.manual_seed(0)
    inputs = torch.randint(0, VOCAB_SIZE, (2, TINY_CONFIG["context_length"]))
    targets = torch.randint(0, VOCAB_SIZE, (2, TINY_CONFIG["context_length"]))

    loss = calc_loss_batch(inputs, targets, tiny_model, torch.device("cpu"))

    assert loss.item() == pytest.approx(math.log(VOCAB_SIZE), abs=0.5)


def _fake_loader(values):
    """A minimal stand-in for a DataLoader: supports len() and iteration."""
    return [(torch.tensor(v), torch.tensor(0.0)) for v in values]


def _stub_loss(input_batch, target_batch, model, device):
    """Returns whatever value the batch carries, so averaging is exactly checkable."""
    return input_batch


def test_calc_loss_loader_averages_over_all_batches():
    """The bug: `return` sat inside the loop, so only the first batch counted.

    With losses [1, 2, 3, 4] the mean is 2.5. The buggy version returned
    1 / 4 = 0.25 -- the first batch's loss divided by the batch count.
    """
    loader = _fake_loader([1.0, 2.0, 3.0, 4.0])

    result = calc_loss_loader(loader, model=None, device=torch.device("cpu"),
                              loss_fn=_stub_loss)

    assert result == pytest.approx(2.5)
    assert result != pytest.approx(0.25), "calc_loss_loader returned after one batch"


def test_calc_loss_loader_respects_num_batches():
    loader = _fake_loader([1.0, 2.0, 3.0, 4.0])

    result = calc_loss_loader(loader, model=None, device=torch.device("cpu"),
                              num_batches=2, loss_fn=_stub_loss)

    assert result == pytest.approx(1.5)


def test_calc_loss_loader_caps_num_batches_at_loader_length():
    loader = _fake_loader([2.0, 4.0])

    result = calc_loss_loader(loader, model=None, device=torch.device("cpu"),
                              num_batches=99, loss_fn=_stub_loss)

    assert result == pytest.approx(3.0)


def test_calc_loss_loader_returns_nan_for_empty_loader():
    result = calc_loss_loader([], model=None, device=torch.device("cpu"),
                              loss_fn=_stub_loss)

    assert math.isnan(result)


def test_layernorm_survives_zero_variance():
    """sqrt(var - eps) goes imaginary when var < eps; sqrt(var + eps) does not."""
    norm = LayerNorm(emb_dim=4)
    constant_input = torch.full((2, 4), 3.0)

    out = norm(constant_input)

    assert not torch.isnan(out).any(), "LayerNorm produced NaN on zero-variance input"


def test_layernorm_normalises_to_zero_mean_unit_variance():
    torch.manual_seed(0)
    norm = LayerNorm(emb_dim=64)

    out = norm(torch.randn(4, 64) * 5 + 2)

    assert out.mean(dim=-1).abs().max().item() == pytest.approx(0, abs=1e-5)
    assert out.var(dim=-1, unbiased=False).max().item() == pytest.approx(1, abs=1e-3)


def test_causal_mask_excluded_from_state_dict(tiny_model):
    """The mask is derived from context_length, not learned.

    Persisting it bloats every checkpoint (~100 MB for 24 layers at a 1024
    context) and makes checkpoints fail to load into a model built with a
    different context_length.
    """
    mask_keys = [k for k in tiny_model.state_dict() if k.endswith(".mask")]

    assert mask_keys == []


def test_legacy_checkpoint_with_mask_buffers_still_loads(tiny_model):
    """Checkpoints written before persistent=False carry mask entries.

    A strict load rejects them as unexpected keys, so load_gpt_state_dict drops
    them. Anything else missing must still raise.
    """
    ctx = TINY_CONFIG["context_length"]
    legacy = dict(tiny_model.state_dict())
    for block in range(TINY_CONFIG["n_layers"]):
        legacy[f"trf_blocks.{block}.att.mask"] = torch.triu(
            torch.ones(ctx, ctx), diagonal=1
        )

    fresh = GPTModel(TINY_CONFIG)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        fresh.load_state_dict(legacy)

    load_gpt_state_dict(fresh, legacy)  # must not raise

    with pytest.raises(RuntimeError):
        load_gpt_state_dict(fresh, {"trf_blocks.0.att.mask": torch.zeros(ctx, ctx)})


def test_model_roundtrips_through_state_dict(tiny_model):
    reloaded = GPTModel(TINY_CONFIG)
    reloaded.load_state_dict(tiny_model.state_dict())

    torch.manual_seed(0)
    sample = torch.randint(0, VOCAB_SIZE, (1, TINY_CONFIG["context_length"]))
    tiny_model.eval()
    reloaded.eval()

    with torch.no_grad():
        assert torch.allclose(tiny_model(sample), reloaded(sample))
