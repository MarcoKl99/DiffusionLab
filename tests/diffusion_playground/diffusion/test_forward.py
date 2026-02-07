import torch

from src.diffusion_playground.diffusion.forward import ForwardDiffusion


SAMPLE_DATA = torch.randn(32, 2)
fd = ForwardDiffusion()


def test_forward_diffusion():
    diffused = fd.diffuse(SAMPLE_DATA)
    assert isinstance(diffused, torch.Tensor)
    assert diffused.shape == torch.Size([fd.time_steps, *SAMPLE_DATA.shape])
