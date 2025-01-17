# taken from: https://github.com/yashbhalgat/HashNeRF-pytorch

import torch

# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from math import exp, log, floor


BOX_OFFSETS = torch.tensor(
    [[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]], device="cuda"
)


def _hash(coords, log2_hashmap_size):
    """
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    """
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result


def total_variation_loss(
    embeddings, min_resolution, max_resolution, level, log2_hashmap_size, n_levels=16
):
    # Get resolution
    b = exp((log(max_resolution) - log(min_resolution)) / (n_levels - 1))
    resolution = torch.tensor(floor(min_resolution * b**level))

    # Cube size to apply TV loss
    min_cube_size = min_resolution - 1
    max_cube_size = 50  # can be tuned
    if min_cube_size > max_cube_size:
        print("ALERT! min cuboid size greater than max!")
        pdb.set_trace()
    cube_size = torch.floor(
        torch.clip(resolution / 10.0, min_cube_size, max_cube_size)
    ).int()

    # Sample cuboid
    min_vertex = torch.randint(0, resolution - cube_size, (3,))
    idx = min_vertex + torch.stack(
        [torch.arange(cube_size + 1) for _ in range(3)], dim=-1
    )
    cube_indices = torch.stack(torch.meshgrid(idx[:, 0], idx[:, 1], idx[:, 2]), dim=-1)

    hashed_indices = _hash(cube_indices.cuda(), log2_hashmap_size)
    cube_embeddings = embeddings(hashed_indices)
    # hashed_idx_offset_x = hash(idx+torch.tensor([1,0,0]), log2_hashmap_size)
    # hashed_idx_offset_y = hash(idx+torch.tensor([0,1,0]), log2_hashmap_size)
    # hashed_idx_offset_z = hash(idx+torch.tensor([0,0,1]), log2_hashmap_size)

    # Compute loss
    # tv_x = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_x), 2).sum()
    # tv_y = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_y), 2).sum()
    # tv_z = torch.pow(embeddings(hashed_idx)-embeddings(hashed_idx_offset_z), 2).sum()
    tv_x = torch.pow(
        cube_embeddings[1:, :, :, :] - cube_embeddings[:-1, :, :, :], 2
    ).sum()
    tv_y = torch.pow(
        cube_embeddings[:, 1:, :, :] - cube_embeddings[:, :-1, :, :], 2
    ).sum()
    tv_z = torch.pow(
        cube_embeddings[:, :, 1:, :] - cube_embeddings[:, :, :-1, :], 2
    ).sum()

    return (tv_x + tv_y + tv_z) / cube_size


def sigma_sparsity_loss(sigmas):
    # Using Cauchy Sparsity loss on sigma values
    return torch.log(1.0 + 2 * sigmas**2).sum(dim=-1)


def get_voxel_vertices(xyz, bounding_box, resolution, log2_hashmap_size):
    """
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    """
    box_min, box_max = bounding_box
    masker = torch.clip(xyz, min=box_min, max=box_max)
    keep_mask = xyz == masker
    if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
        # print("ALERT: some points are outside bounding box. Clipping them!")
        xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max - box_min) / resolution

    bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
    voxel_min_vertex = bottom_left_idx * grid_size + box_min
    voxel_max_vertex = (
        voxel_min_vertex + torch.tensor([1.0, 1.0, 1.0], device=xyz.device) * grid_size
    )

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = _hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask


class HashEmbedder(nn.Module):
    def __init__(
        self,
        bounding_box,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        finest_resolution=512,
    ):
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp(
            (torch.log(self.finest_resolution) - torch.log(self.base_resolution))
            / (n_levels - 1)
        )

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(2**self.log2_hashmap_size, self.n_features_per_level)
                for i in range(n_levels)
            ]
        )
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()

    def trilinear_interp(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        """
        x: B x 3
        voxel_min_vertex: B x 3
        voxel_max_vertex: B x 3
        voxel_embedds: B x 8 x 2
        """
        # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
        weights = (x - voxel_min_vertex) / (
            voxel_max_vertex - voxel_min_vertex
        )  # B x 3

        # step 1
        # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
        c00 = (
            voxel_embedds[:, 0] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 4] * weights[:, 0][:, None]
        )
        c01 = (
            voxel_embedds[:, 1] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 5] * weights[:, 0][:, None]
        )
        c10 = (
            voxel_embedds[:, 2] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 6] * weights[:, 0][:, None]
        )
        c11 = (
            voxel_embedds[:, 3] * (1 - weights[:, 0][:, None])
            + voxel_embedds[:, 7] * weights[:, 0][:, None]
        )

        # step 2
        c0 = c00 * (1 - weights[:, 1][:, None]) + c10 * weights[:, 1][:, None]
        c1 = c01 * (1 - weights[:, 1][:, None]) + c11 * weights[:, 1][:, None]

        # step 3
        c = c0 * (1 - weights[:, 2][:, None]) + c1 * weights[:, 2][:, None]

        return c

    def forward(self, x):
        # x is 3D point position: B x 3
        x_embedded_all = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask = (
                get_voxel_vertices(
                    x, self.bounding_box, resolution, self.log2_hashmap_size
                )
            )

            voxel_embedds = self.embeddings[i](hashed_voxel_indices)

            x_embedded = self.trilinear_interp(
                x, voxel_min_vertex, voxel_max_vertex, voxel_embedds
            )
            x_embedded_all.append(x_embedded)

        keep_mask = keep_mask.sum(dim=-1) == keep_mask.shape[-1]
        return torch.cat(x_embedded_all, dim=-1), keep_mask


class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):

        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree**2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396,
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435,
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761,
        ]

    def forward(self, input, **kwargs):

        result = torch.empty(
            (*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device
        )
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                # result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (
                            xx * (xx - 3 * yy) - yy * (3 * xx - yy)
                        )

        return result
