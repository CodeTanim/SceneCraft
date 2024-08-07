import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def get_xavier_weights(shape, device="cpu"):
    return torch.nn.Parameter(
        torch.nn.init.xavier_normal_(torch.empty(*shape, dtype=torch.float32))
    ).to(device)


class MultiEmbedding(nn.Module):
    def __init__(self, dim, other_dim, embed_ct=1):
        super().__init__()

        self.dim = dim
        self.other_dim = other_dim
        self.embed_ct = embed_ct

        self.weights = torch.nn.Parameter(
            get_xavier_weights((embed_ct, dim * other_dim))
        )

    def add_embeds(self, n=1):
        self.embed_ct += n

        self.weights = torch.nn.Parameter(
            torch.cat(
                [
                    self.weights,
                    get_xavier_weights(
                        (n, self.dim * self.other_dim), self.weights.device
                    ),
                ]
            )
        )

        return self.embed_ct

    def forward(self, indices):
        return F.embedding(indices, self.weights)


class Adapter(nn.Module):
    def __init__(self, rank, tasks, dim):
        super().__init__()

        self.rank = rank
        self.tasks = tasks

        self.embedding = MultiEmbedding(rank, dim, tasks)

    def add_task(self, n=1):
        return self.embedding.add_embeds(n)


class LowRankMultiDense(Adapter):
    def __init__(self, in_dim, out_dim, rank=32, tasks=1, dim=256):
        super().__init__(rank, tasks, dim)

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weights_a = get_xavier_weights((rank, in_dim, rank))
        self.weights_b = get_xavier_weights((rank, rank, out_dim))

    def forward(self, x, task):
        # x shape is (batch, pts_per_ray, hidden_dim)
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
            task = task.repeat(x.shape[0])

        if task.shape[0] != x.shape[0]:
            task = task.repeat(x.shape[0])

        embeds = self.embedding(task)
        embeds = rearrange(embeds, "b (r n) -> b r n", r=self.rank)

        A = torch.einsum("brn,rir->bir", embeds, self.weights_a)
        B = self.weights_b

        wt = torch.einsum("bir,bai->bar", A, x)
        wt = torch.einsum("bar,rro->bao", wt, B)
        return wt / self.rank


class AdaptedDense(nn.Module):
    def __init__(self, in_dim, out_dim, adapter_rank=32, init_tasks=1, dim=256):
        """This layer is a wrapper around dense to include the adapter.

        Normally, when adapting, this would be done at runtime, but the code
        for that is closed-source for this adapter framework. This is a workaround
        to avoid getting sued by Microsoft."""

        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.dense = nn.Linear(in_dim, out_dim)
        self.adapter = LowRankMultiDense(
            in_dim, out_dim, adapter_rank, init_tasks, dim=dim
        )

        self.adapter_enabled = False

    def enable_adapter(self):
        self.adapter_enabled = True

    def disable_adapter(self):
        self.adapter_enabled = False

    def add_task(self, n=1):
        return self.adapter.add_task(n)

    def forward(self, x, task):
        dense_out = self.dense(x)

        if self.adapter_enabled:
            dense_out += self.adapter(x, task)

        return dense_out
