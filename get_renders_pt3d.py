from train_nerf_pt3d import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

datasets = [
    "materials",
]

image_dim = (224, 224)

dl_val = get_split(
    "./data/nerf_synthetic",
    "val",
    datasets,
    1,
    image_dim,
)

model = LightningRadianceField.load_from_checkpoint(
    args.model_path,
)

trainer = pl.Trainer()

trainer.validate(model, dl_val)
