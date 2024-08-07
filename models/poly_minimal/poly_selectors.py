import torch

EPS = 1e-9
UNIFORM_LOW = -1e-3
UNIFORM_HIGH = 1e-3


class Selector(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, routing_info):
        pass


class PolytroponSelector(Selector):
    def __init__(self, n_tasks, n_skills, n_splits, dropout=None, l2_norm=None):
        super().__init__()

        self.n_tasks = n_tasks
        self.n_skills = n_skills
        self.n_splits = n_splits

        self.dropout = dropout

        self.l2_norm = l2_norm

        self.weights = torch.nn.Parameter(
            torch.empty(n_tasks, n_skills * n_splits).uniform_()
        )

    def add_task(self, n=1):
        self.weights = torch.nn.Parameter(
            torch.cat(
                [
                    self.weights,
                    torch.empty(n, self.n_skills * self.n_splits).uniform_(),
                ],
                dim=0,
            )
        )
        self.n_tasks += 1
        return self.n_tasks - 1

    def forward(self, routing_info):
        """
        routing_info: (bs, 1) -> label encoding of the task
        """

        weights = self.weights[routing_info].view(-1, self.n_splits, self.n_skills)

        if self.l2_norm:
            weights = torch.nn.functional.normalize(weights, dim=-1, p=2)

        module_logits = torch.sigmoid(weights)

        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)

        if self.dropout is not None:
            module_weights = torch.nn.functional.dropout(module_weights, p=self.dropout)

        return module_weights


class TaskInterpolationSelector(PolytroponSelector):
    def __init__(self, n_tasks, n_skills, n_splits, dropout=None, l2_norm=None):
        super().__init__(n_tasks, n_skills, n_splits, dropout, l2_norm)

    def forward(self, task_info):
        routing_a, routing_b, alpha = task_info

        weights_a = self.weights[routing_a].view(-1, self.n_splits, self.n_skills)
        weights_b = self.weights[routing_b].view(-1, self.n_splits, self.n_skills)

        weights = alpha * weights_a + (1 - alpha) * weights_b

        if self.l2_norm is not None:
            weights = torch.nn.functional.normalize(weights, dim=-1, p=2)

        module_logits = torch.sigmoid(weights)
        module_weights = module_logits / (module_logits.sum(dim=-1, keepdim=True) + EPS)

        if self.dropout is not None:
            module_weights = torch.nn.functional.dropout(module_weights, p=self.dropout)

        return module_weights
