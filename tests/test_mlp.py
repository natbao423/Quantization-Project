"""The MLP sweep's run(): the weight_master condition and upd_survive."""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_at_vary_precision as M      # noqa: E402

EPOCHS = 60       # enough steps to move; the tests are about bookkeeping


def test_weight_master_at_23_bits_is_the_fp32_run():
    """QuantizedForward is a no-op at 23 bits, so the shared baseline holds."""
    assert M.run(23, quant_weight=True, master=True, epochs=EPOCHS) == \
           M.run(23, epochs=EPOCHS)


def test_upd_survive_is_one_when_weights_are_not_quantized():
    assert M.run(4, quant_input=True, epochs=EPOCHS)[2] == 1.0
    assert M.run(4, quant_grad=True, block=16, epochs=EPOCHS)[2] == 1.0


@pytest.mark.parametrize("master", [False, True])
def test_upd_survive_is_a_fraction_below_23_bits(master):
    _, _, moved = M.run(3, quant_weight=True, master=master, epochs=EPOCHS)
    assert 0 < moved < 1


def test_weight_master_differs_from_weight():
    """Same seed, same width: only whether a master exists differs."""
    assert M.run(3, quant_weight=True, master=True, epochs=EPOCHS)[0] != \
           M.run(3, quant_weight=True, epochs=EPOCHS)[0]


def test_conditions_include_weight_master_next_to_weight():
    tags = [c.tag for c in M.CONDITIONS]
    assert tags.index("weight_master") == tags.index("weight") + 1
    cond = M.CONDITIONS[tags.index("weight_master")]
    assert cond.quant_weight and cond.master and not cond.quant_input
