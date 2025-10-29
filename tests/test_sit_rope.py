import pytest
import torch

from tread_diffusion import SiT


def _forward_once(rope):
    model = SiT(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=64,
        depth=2,
        num_heads=4,
        num_classes=10,
        rope=rope,
        learn_sigma=False,
    )
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,)).float()
    y = torch.randint(0, 10, (2,))
    out = model(x, t, y)
    return out


@pytest.mark.parametrize("rope", [None, "golden_gate"])
def test_dit_forward_shapes(rope):
    out = _forward_once(rope)
    assert out.shape[0] == 2
    assert out.shape[1] == 4
    assert out.shape[2] == 8
    assert out.shape[3] == 8
