import torch
import math
import numpy as np

EPS = 1e-9


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class PolyDummyAdapter(Adapter):
    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward(self, x):
        with torch.no_grad():
            module_out = self.module(x)

        return module_out


class PolyDenseAdapter(Adapter):
    def __init__(
        self,
        route_dict,
        module,
        skills,
        splits=1,
        rank=8,
    ):
        super().__init__()

        self.in_features = module.in_features
        self.out_features = module.out_features
        self.skills = skills
        self.splits = splits
        self.rank = rank

        self.module = module

        assert (
            self.in_features % self.splits == 0
        ), "in_features must be divisible by splits"
        assert (
            self.out_features % self.skills == 0
        ), "out_features must be divisible by skills"

        self._lora_a_shape = [
            self.splits,
            self.skills,
            self.in_features // self.splits,
            self.rank,
        ]

        self._lora_b_shape = [
            self.splits,
            self.skills,
            self.rank,
            self.out_features // self.splits,
        ]

        gain = np.sqrt((2 / (1 + (math.sqrt(5) ** 2))))
        std = gain / math.sqrt(self.in_features)

        self._a = torch.nn.Parameter(
            torch.empty(self._lora_a_shape).uniform_(-std, std)
        )

        self._b = torch.nn.Parameter(
            torch.nn.init.zeros_(torch.empty(self._lora_b_shape))
        )

        self.route_dict = route_dict

    def freeze_module(self):
        for param in self.module.parameters():
            param.requires_grad = False

    def forward(self, x):
        # TODO: x shape
        x = x.squeeze()
        bs = x.shape[0]
        skill_weights = self.route_dict["routes"].repeat(bs, 1, 1)
        add_lora = self.route_dict["add_lora"]

        if not add_lora:
            return self.module(x)

        # following copied from mttl
        # A is    n_splits, n_skills, D // n_splits, rank
        # we want bs,       n_splits, D // n_splits, rank
        A = torch.einsum("bqs,qsdr->bqdr", (skill_weights, self._a))
        B = torch.einsum("bqs,qsrd->bqrd", (skill_weights, self._b))

        A = A.reshape(bs, self.in_features, self.rank)
        B = B.transpose(1, 2).reshape(bs, self.rank, self.out_features)

        module_out = self.module(x)
        adapter_out = x.bmm(A).bmm(B) / self.rank
        adapter_out = adapter_out.squeeze(1)

        return torch.add(module_out, adapter_out).unsqueeze(0)


class PolyConv2dAdapter(Adapter):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
