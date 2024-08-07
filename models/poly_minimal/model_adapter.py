import torch
import typing
import json
from .poly_selectors import PolytroponSelector
from .poly import PolyDenseAdapter, PolyConv2dAdapter, PolyDummyAdapter
import operator


def get_layers(model):
    return [n for n, _ in model.named_modules()]


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class AdaptedModel(torch.nn.Module):
    def __init__(
        self,
        model,
        adapt_config=typing.Union[str, typing.Dict[str, typing.Any]],
        selector=PolytroponSelector,
    ):
        """
        Adaptation Config Format:
        {
            "n_tasks": int,
            "n_skills": int,
            "n_splits": int,
            "rank": int,
            "dropout": float,
            "adapt_dense_layers": int || None, # -1 = All
            "adapt_conv_layers": int || None, # -1 = All
            "skip_layers": int || None
        }
        """

        super().__init__()
        self.model = model

        if isinstance(adapt_config, str):
            self.adapt_config = self.parse_json(adapt_config)
        elif isinstance(adapt_config, dict):
            self.adapt_config = adapt_config

        self.adapt_config = Struct(**self.adapt_config)

        self.selector = selector(
            self.adapt_config.n_tasks,
            self.adapt_config.n_skills,
            self.adapt_config.n_splits,
            self.adapt_config.__dict__.get("dropout", None),
            self.adapt_config.__dict__.get("l2_norm", None),
        )

        self.route_dict = {}
        self.route_dict["add_lora"] = True
        self.freeze_model()
        self._adapt()

    @staticmethod
    def parse_json(json_path):
        with open(json_path, "r") as f:
            adapt_config = json.load(f)
        return adapt_config

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_all_but_selector(self):
        for name, param in self.named_parameters():
            if "selector" not in name:
                param.requires_grad = False

    def unfreeze_everything(self):
        for param in self.parameters():
            param.requires_grad = True

    def get_selector_weights(self):
        return next(self.selector.parameters()).detach().cpu().numpy()

    def set_selector_weights(self, weights):
        next(self.selector.parameters()).data = torch.from_numpy(weights).cuda()

    def add_task(self, n=1):
        self.selector.add_task(n)

    def remove_lora(self):
        self.route_dict["add_lora"] = False

    def _adapt(self):
        layers = get_layers(self.model)
        for layer in layers:
            try:
                module = operator.attrgetter(layer)(self.model)
            except AttributeError:
                continue
            # todo: add support for conv layers
            # todo: freeze layers
            if (
                isinstance(module, torch.nn.Linear)
                and (
                    self.adapt_config.adapt_dense_layers == -1
                    or self.adapt_config.adapt_dense_layers in layer
                )
                and module.in_features % self.adapt_config.n_skills == 0
                and module.out_features % self.adapt_config.n_skills == 0
            ):
                print("Adapting Dense Layer: ", layer)
                adapter = PolyDenseAdapter(
                    self.route_dict,
                    module,
                    self.adapt_config.n_skills,
                    self.adapt_config.n_splits,
                    self.adapt_config.rank,
                )

                layer_path = ".".join(layer.split(".")[:-1])
                layer_attr = layer.split(".")[-1]

                setattr(
                    operator.attrgetter(layer_path)(self.model), layer_attr, adapter
                )

    def forward(self, **kwargs):
        task_labels = kwargs.get("routes", None)
        del kwargs["routes"]
        skill_weights = self.selector(task_labels)
        self.route_dict["routes"] = skill_weights
        p, w = self.model(**kwargs)
        return p, w


if __name__ == "__main__":
    from torchsummary import summary

    basic_model = torch.nn.Sequential(
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 16),
    ).cuda()

    input = torch.randn(128, 784).cuda()
    tasks = torch.randint(0, 10, (128,)).cuda()

    out = basic_model(input)

    adapted_model = AdaptedModel(basic_model, "poly_configs/default.json").cuda()
    out_adapted = adapted_model(input, tasks)

    print(out.shape, out_adapted.shape)
    print((out == out_adapted).all())
