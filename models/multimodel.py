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
from .multimodel_components import AdaptedDense
from .hash_encoding import SHEncoder
from .utils import BaseDenseModule


class MultiNeRFBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        adapter_rank=32,
        n_tasks=1,
        dim=256,
        activation=nn.LeakyReLU(),
    ):
        super().__init__()

        self.adapter = AdaptedDense(in_dim, out_dim, adapter_rank, n_tasks, dim)
        self.activation = activation

    def enable_adapter(self):
        self.adapter.enable_adapter()

    def disable_adapter(self):
        self.adapter.disable_adapter()

    def add_task(self, n=1):
        return self.adapter.add_task(n)

    def forward(self, x, task):
        return self.activation(self.adapter(x, task))


class MultiNeRFLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        layers=8,
        adapter_rank=32,
        n_tasks=1,
        dim=256,
        activation=nn.LeakyReLU(),
    ):
        super().__init__()

        self.in_block = MultiNeRFBlock(
            in_dim, hidden_dim, adapter_rank, n_tasks, dim, activation
        )

        self.mlp = nn.ModuleList(
            [
                MultiNeRFBlock(
                    hidden_dim, hidden_dim, adapter_rank, n_tasks, dim, activation
                )
                for _ in range(layers)
            ]
        )

    def add_task(self, n=1):
        in_tasks = self.in_block.add_task(n)
        for block in self.mlp:
            block.add_task(n)

        return in_tasks

    def enable_adapter(self):
        self.in_block.enable_adapter()
        for block in self.mlp:
            block.enable_adapter()

    def disable_adapter(self):
        self.in_block.disable_adapter()
        for block in self.mlp:
            block.disable_adapter()

    def forward(self, x, task):
        x = self.in_block(x, task)
        for block in self.mlp:
            x = block(x, task)
        return x


class MultiNeRFBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        n_layers=8,
        adapter_rank=32,
        dim=256,
        activation=nn.LeakyReLU(),
    ):
        super().__init__()

        self.positional_encoding = SHEncoder()
        # self.harmonic_embedding = HarmonicEmbedding()

        self.task_map = {
            "default": 0,
            "chair": 1,
            "drums": 2,
            "ficus": 3,
            "hotdog": 4,
            "lego": 5,
            "materials": 6,
            "mic": 7,
            "ship": 8,
        }
        self.n_tasks = 9

        embedding_dim = 16
        other_dim = 16

        # mlp backbone
        self.mlp = MultiNeRFLayer(
            embedding_dim,
            hidden_dim,
            n_layers,
            adapter_rank,
            self.n_tasks,
            dim,
            activation,
        )

        self.mlp.enable_adapter()

        # gets the colors
        self.color_layer = nn.Sequential(
            nn.Linear(hidden_dim + other_dim, hidden_dim),
            activation,
            BaseDenseModule(hidden_dim, activation=activation),
            BaseDenseModule(hidden_dim, activation=activation),
            BaseDenseModule(hidden_dim, activation=activation),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

        # gets the densities (opacity)
        self.density_layer = nn.Sequential(
            nn.Linear(hidden_dim + other_dim, hidden_dim),
            activation,
            BaseDenseModule(hidden_dim, activation=activation),
            BaseDenseModule(hidden_dim, activation=activation),
            BaseDenseModule(hidden_dim, activation=activation),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def embed_directions(self, directions):
        # the harmonic embedding is used to encode the positions as per mildenhall et al.
        embedding = self.positional_encoding(directions)
        return embedding

    def embed(self, features, directions):
        expanded = self.embed_directions(directions)
        n_repeats = features.shape[1]
        expanded = repeat(expanded, "b d -> b n d", n=n_repeats)
        embedded = torch.cat([features, expanded], dim=-1)
        return embedded

    def add_task(self, name):
        self.task_map[name] = self.n_tasks
        self.n_tasks += 1
        return self.mlp.add_task(1)

    def batched_forward(
        self,
        ray_bundle,
        tasks=["default"],
        n_batches=4,
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
                tasks=tasks,
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

    def forward(self, ray_bundle, tasks=["default"]):
        # get the ray points and directional embedding
        points_world = ray_bundle_to_ray_points(ray_bundle)

        points_world = (points_world + 1.0) / 6

        embeds = self.positional_encoding(points_world)

        # forward pass through layers
        for task in tasks:
            if task not in self.task_map:
                self.add_task(task)

        task = torch.tensor(
            [self.task_map[task] for task in tasks], device=embeds.device
        )

        features = self.mlp(embeds, task)
        features = self.embed(features, ray_bundle.directions)

        densities = self.density_layer(features)
        colors = self.color_layer(features)

        return densities, colors


class MultiNeRF(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        adapter_rank=128,
        n_layers=4,
        n_fine_rays=640,
        fine_ray_pts=128,
        dims=256,
        image_dim=(224, 224),
        activation=nn.ReLU(),
    ):
        super().__init__()

        self.fine = torch.compile(
            MultiNeRFBackbone(
                n_layers=n_layers,
                adapter_rank=adapter_rank,
                hidden_dim=hidden_dim,
                activation=activation,
                dim=dims,
            )
        )

        self.n_fine_rays = n_fine_rays
        self.fine_ray_pts = fine_ray_pts

        self.fine_raysampler = MonteCarloRaysampler(
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            n_rays_per_image=self.n_fine_rays,
            n_pts_per_ray=self.fine_ray_pts,
            min_depth=0.0,
            max_depth=1.0,
        )

        self.inference_raysampler = NDCMultinomialRaysampler(
            image_height=image_dim[0],
            image_width=image_dim[1],
            n_pts_per_ray=128,
            min_depth=0.0,
            max_depth=1.0,
        )

        self.renderer_fine = ImplicitRenderer(
            raysampler=self.fine_raysampler,
            raymarcher=EmissionAbsorptionRaymarcher(),
        )

        self.renderer_inferece = ImplicitRenderer(
            raysampler=self.inference_raysampler,
            raymarcher=EmissionAbsorptionRaymarcher(),
        )

    def render_inference(self, cameras, tasks=["default"], n_batches=16):
        with torch.no_grad():
            self.renderer_inference = self.renderer_inferece.to(cameras.device)
            sil_fine, ray_fine = self.renderer_inference(
                cameras=cameras,
                volumetric_function=self.fine.batched_forward,
                tasks=tasks,
                n_batches=n_batches,
            )

            fine_images, fine_sils = sil_fine.split([3, 1], dim=-1)

        return fine_images

    def render(self, cameras, tasks=["default"], n_batches=1):
        self.renderer_fine = self.renderer_fine.to(cameras.device)

        sil_fine, ray_fine = self.renderer_fine(
            cameras=cameras,
            volumetric_function=self.fine.batched_forward,
            tasks=tasks,
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
        tasks=["default"],
    ):
        return self.render(cameras, tasks=tasks)
