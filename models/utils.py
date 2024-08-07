import torch
import torch.nn as nn
from einops import rearrange


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        super().__init__()
        self.register_buffer(
            "frequencies",
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


def sample_images_at_locs(target_images, sampled_rays_xy):
    sampled_rays_xy = rearrange(sampled_rays_xy, "b ... xy -> b (...) xy")
    ba = target_images.shape[0]
    dim = target_images.shape[-1]
    spatial_size = sampled_rays_xy.shape[1:-1]
    images_sampled = torch.nn.functional.grid_sample(
        target_images.permute(0, 3, 1, 2),
        -sampled_rays_xy.view(ba, -1, 1, 2),  # note the sign inversion
        align_corners=True,
    )
    return images_sampled.permute(0, 2, 3, 1).view(ba, *spatial_size, dim)


def get_render(
    renderer_grid,
    neural_radiance_field,
    camera,
):

    with torch.no_grad():
        rendered_image_silhouette, rays = renderer_grid(
            cameras=camera, volumetric_function=neural_radiance_field.batched_forward
        )

        rendered_image, _ = rendered_image_silhouette[0].split([3, 1], dim=-1)
    return rendered_image


class BaseDenseModule(nn.Module):
    def __init__(self, latent_size, activation=nn.ReLU()):
        super().__init__()

        self.module = nn.Sequential(nn.Linear(latent_size, latent_size), activation)

    def forward(self, x):
        return self.module(x)


class BaseResidualModule(nn.Module):
    def __init__(self, latent_size, activation=nn.ReLU()):
        super().__init__()

        self.module = nn.Sequential(nn.Linear(latent_size, latent_size), activation)
        self.norm = nn.LayerNorm(latent_size)

    def forward(self, x):
        return self.norm(x + self.module(x))


def get_xavier(*shape, **kwargs):
    return nn.init.xavier_uniform_(torch.empty(*shape, **kwargs))
