import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import HarmonicEmbedding
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
from functools import partial
from einops import rearrange, repeat
from .hash_encoding import *
from .utils import BaseDenseModule


class HashNeRF(nn.Module):
    def __init__(self, latent_size=256, n_layers_mlp=3, n_layers_heads=2):
        super().__init__()

        self.hash_embedder = HashEmbedder([-1, 5])
        self.sh_encoder = HarmonicEmbedding()

        harmonic_dim = 60 * 6
        encode_size = 32 + harmonic_dim

        self.mlp = nn.Sequential(
            nn.Linear(encode_size, latent_size),
            nn.ReLU(),
            *[BaseDenseModule(latent_size) for _ in range(n_layers_mlp)],
        )

        self.color = nn.Sequential(
            nn.Linear(latent_size + harmonic_dim, latent_size),
            nn.ReLU(),
            *[BaseDenseModule(latent_size) for _ in range(n_layers_heads)],
            nn.Linear(latent_size, 3),
            nn.Sigmoid(),
        )

        self.alpha = nn.Sequential(
            nn.Linear(latent_size + harmonic_dim, latent_size),
            nn.ReLU(),
            *[BaseDenseModule(latent_size) for _ in range(n_layers_heads)],
            nn.Linear(latent_size, 1),
            nn.Sigmoid(),
        )

    def batched_forward(
        self,
        ray_bundle,
        n_batches=1,
        **kwargs,
    ):
        """Batches a bunch of rays and calls forward in a batched manner."""

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
            )
            for batch in batches
        ]

        # retrieve the outputs (batched)
        ray_densities, ray_colors = [
            torch.cat([batch_output[i] for batch_output in batch_outputs], dim=0).view(
                *spatial_size, -1
            )
            for i in (0, 1)
        ]

        return ray_densities, ray_colors

    def embed_directions(self, directions):
        directions_normed = F.normalize(directions, p=2, dim=-1)

        # the harmonic embedding is used to encode the positions as per mildenhall et al.
        embedding = self.sh_encoder(directions_normed)
        return embedding

    def embed(self, features, directions):
        expanded = self.embed_directions(directions)
        n_repeats = features.shape[0] // expanded.shape[0]
        expanded = repeat(expanded, "b d -> (b n) d", n=n_repeats)
        embedded = torch.cat([features, expanded], dim=-1)
        return embedded

    def forward(self, ray_bundle):
        points_world = ray_bundle_to_ray_points(ray_bundle)

        # hash encoding
        bs, pts, _ = points_world.shape
        hash_points, keep_mask = self.hash_embedder(points_world.view(-1, 3))

        # sh encoding
        points_world = F.normalize(points_world, p=2, dim=-1)
        sh_points = self.sh_encoder(points_world.view(-1, 3))

        # concat
        encode_points = torch.cat([hash_points, sh_points], dim=-1)

        # mlp
        latent = self.mlp(encode_points)
        latent = self.embed(latent, ray_bundle.directions)

        # color
        color = self.color(latent)
        color = color.view(bs, pts, 3)

        # alpha
        alpha = self.alpha(latent)
        alpha = alpha.view(bs, pts, 1)

        return alpha, color


class HashNeRFWrapper(nn.Module):
    def __init__(
        self, image_dim=(224, 224), latent_size=256, n_layers_mlp=3, n_layers_heads=2
    ):
        super().__init__()

        self.model = torch.compile(HashNeRF(latent_size, n_layers_mlp, n_layers_heads))

        n_pts_ray = 128
        n_rays = 1024

        self.mc_raysampler = MonteCarloRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            n_rays_per_image=n_rays,
            n_pts_per_ray=n_pts_ray,
            min_depth=0.1,
            max_depth=1.0,
        )

        self.grid_raysampler = NDCMultinomialRaysampler(
            image_height=image_dim[0],
            image_width=image_dim[1],
            n_pts_per_ray=n_pts_ray,
            min_depth=0.1,
            max_depth=1.0,
        )

        self.renderer_mc = ImplicitRenderer(
            raysampler=self.mc_raysampler,
            raymarcher=EmissionAbsorptionRaymarcher(),
        )

        self.renderer_grid = ImplicitRenderer(
            raysampler=self.grid_raysampler,
            raymarcher=EmissionAbsorptionRaymarcher(),
        )

    def render_inference(self, cameras, n_batches=16):
        with torch.no_grad():
            self.renderer_grid = self.renderer_grid.to(cameras.device)
            sil_fine, ray_fine = self.renderer_grid(
                cameras=cameras,
                volumetric_function=self.model.batched_forward,
                n_batches=n_batches,
            )

            fine_images, fine_sils = sil_fine.split([3, 1], dim=-1)

        return fine_images

    def render(self, cameras, n_batches=1):
        self.renderer_mc = self.renderer_mc.to(cameras.device)

        sil_fine, ray_fine = self.renderer_mc(
            cameras=cameras,
            volumetric_function=self.model.batched_forward,
            n_batches=n_batches,
        )

        fine_images, fine_sils = sil_fine.split([3, 1], dim=-1)
        return (
            fine_images,
            fine_sils,
            ray_fine,
        )

    def forward(
        self,
        cameras,
    ):
        return self.render(cameras)
