import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import BaseResidualModule, HarmonicEmbedding, get_xavier
from pytorch3d.renderer import (
    RayBundle,
    ray_bundle_to_ray_points,
)
from pytorch3d.renderer import (
    PerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
)


class NeRFBackbone(nn.Module):

    def __init__(self, in_dim, latent_dim, layers=2):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.ReLU(),
            *[BaseResidualModule(latent_dim) for _ in range(layers)],
        )

    def forward(self, x):
        return self.mlp(x)


class NeRFHead(nn.Module):
    def __init__(self, latent_dim, out_dim, layers=2):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            *[BaseResidualModule(latent_dim) for _ in range(layers)],
            nn.Linear(latent_dim, out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class NeRFEmbedder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.x_pos = nn.Parameter(get_xavier(1, hidden_dim))
        self.y_pos = nn.Parameter(get_xavier(1, hidden_dim))
        self.z_pos = nn.Parameter(get_xavier(1, hidden_dim))

    def forward(self, points):
        x = points[..., 0].unsqueeze(-1) * self.x_pos
        y = points[..., 1].unsqueeze(-1) * self.y_pos
        z = points[..., 2].unsqueeze(-1) * self.z_pos

        return (x + y + z) / 3


def get_n_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class NeRF(nn.Module):
    def __init__(self, backbone_layers=8, head_layers=4, hidden_dim=768, fine_pts=128):
        super().__init__()

        self.fine_pts = fine_pts

        n_harmonic = 6
        self.positional_embeddings = HarmonicEmbedding(n_harmonic)

        harm_dim = 6 * n_harmonic

        self.backbone = NeRFBackbone(harm_dim, hidden_dim, layers=backbone_layers)

        self.color_head = nn.Sequential(
            nn.Linear(hidden_dim + harm_dim, hidden_dim),
            nn.ReLU(),
            NeRFHead(hidden_dim, 3, layers=head_layers),
            nn.Sigmoid(),
        )
        self.density_head = nn.Sequential(
            nn.Linear(hidden_dim + harm_dim, hidden_dim),
            nn.ReLU(),
            NeRFHead(hidden_dim, 1, layers=head_layers),
            nn.Sigmoid(),
        )

    def embed_directions(self, features, directions):
        spatial = features.shape[:-1]
        embedding = self.positional_embeddings(directions)

        expanded = embedding[..., None, :].expand(*spatial, embedding.shape[-1])
        return expanded

    def embed(self, features, directions):
        expanded = self.embed_directions(features, directions)
        embedded = torch.cat([features, expanded], dim=-1)
        return embedded

    def forward(self, ray_bundle, **kwargs):
        # batchsize, num_points, 3
        points_world = ray_bundle_to_ray_points(ray_bundle)

        # num_points, hidden_dim
        points = self.positional_embeddings(points_world)

        # batchsize, num_points, hidden_dim
        features = self.backbone(points)

        # embed dirs
        directions = ray_bundle.directions
        features = self.embed(features, directions)

        colors = self.color_head(features)
        densities = self.density_head(features)

        return densities, colors

    def batched_forward(self, ray_bundle, n_batches=8, **kwargs):
        # ray batching
        pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = [*ray_bundle.origins.shape[:-1], pts_per_ray]

        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        # batched forward
        batch_outputs = [
            self.forward(
                # construct a batched bundle
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch],
                    directions=ray_bundle.directions.view(-1, 3)[batch],
                    lengths=ray_bundle.lengths.view(-1, pts_per_ray)[batch],
                    xys=None,
                ),
                **kwargs,
            )
            for batch in batches
        ]

        densities, colors = [
            torch.cat([batch_output[i] for batch_output in batch_outputs], dim=0).view(
                *spatial_size, -1
            )
            for i in (0, 1)
        ]

        return densities, colors


def get_inference_renderer(image_height, image_width, n_pts_per_ray=128):
    sampler = NDCMultinomialRaysampler(
        image_height=image_height,
        image_width=image_width,
        n_pts_per_ray=n_pts_per_ray,
        min_depth=0.0,
        max_depth=1.0,
    )
    renderer = ImplicitRenderer(
        raymarcher=EmissionAbsorptionRaymarcher(),
        raysampler=sampler,
    )

    return renderer


def get_training_renderer(n_rays_per_image=512, n_pts_per_ray=128):
    sampler = MonteCarloRaysampler(
        min_x=-1.0,
        max_x=1.0,
        min_y=-1.0,
        max_y=1.0,
        n_rays_per_image=n_rays_per_image,
        n_pts_per_ray=n_pts_per_ray,
        min_depth=0.0,
        max_depth=1.0,
    )
    renderer = ImplicitRenderer(
        raymarcher=EmissionAbsorptionRaymarcher(),
        raysampler=sampler,
    )

    return renderer


def render_model_train(cameras, model: nn.Module, renderer: ImplicitRenderer, **kwargs):
    img, ray = renderer.forward(
        cameras=cameras,
        volumetric_function=model.forward,
        # n_batches=kwargs.get("n_batches", 4),
        **kwargs,
    )

    img, sil = img.split([3, 1], dim=-1)
    return img, sil, ray


def render_model_inference(
    cameras, model: nn.Module, renderer: ImplicitRenderer, **kwargs
):
    model.eval()
    with torch.no_grad():
        img, ray = renderer(
            cameras=cameras,
            volumetric_function=model.batched_forward,
            n_batches=kwargs.get("n_batches", 16),
            **kwargs,
        )

        img, sil = img.split([3, 1], dim=-1)
    model.train()
    return img, sil
