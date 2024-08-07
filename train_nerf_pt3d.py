import torch
import torch.nn as nn
import numpy as np
from models.nerf_pt3d.nerf_renderer import RadianceFieldRenderer
from data_loading.data_loader import PerspectiveCameras
from models.utils import sample_images_at_locs
import torch.nn.functional as F
import pytorch_lightning as pl
from data_loading.data_loader import NeRFDataLoader, collate_fn
from schedule_free_adam import AdamWScheduleFree
from models.poly_minimal import AdaptedModel
import os
from PIL import Image

USE_WANDB = False

if USE_WANDB:
    import wandb

torch.set_float32_matmul_precision("medium")


USE_POLY = False
NAME_ROUTE_MAP = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]


class LightningRadianceField(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = RadianceFieldRenderer(
            (224, 224), 64, 64, 1024, 1.2, 6.28, True, False, 6000, 10, 4
        )
        if USE_POLY:
            self.adapt_config = {
                "n_tasks": 10,
                "n_skills": 256,
                "n_splits": 32,
                "rank": 4,
                "adapt_dense_layers": -1,
            }
            self.model = AdaptedModel(self.model, adapt_config=self.adapt_config)
            self.model.unfreeze_everything()

    def forward(self, cameras, images, is_training=True, **kwargs):
        self.model = self.model.train(is_training)
        images = images.squeeze(0)
        if USE_POLY:
            names = kwargs.get("names", None)
            routes = torch.tensor(
                [NAME_ROUTE_MAP.index(name) for name in names],
                dtype=torch.long,
                device=cameras.device,
            )
            outs = self.model.forward(
                camera_hash=None,
                camera=cameras,
                image=images,
                routes=routes,
            )
        else:
            outs = self.model.forward(camera_hash=None, camera=cameras, image=images)
        return outs

    def training_step(self, batch, batch_idx):
        cameras, images, masks, names = batch

        out_dict, metrics = self.forward(cameras, images, names=names)

        self.log_dict(
            {
                "mse/coarse": metrics["mse_coarse"],
                "mse/fine": metrics["mse_fine"],
                "psnr/coarse": metrics["psnr_coarse"],
                "psnr/fine": metrics["psnr_fine"],
            },
            on_epoch=True,
            prog_bar=True,
        )

        return metrics["mse_coarse"] + metrics["mse_fine"]

    def validation_step(self, batch, batch_idx):
        cameras, images, masks, names = batch
        out_dict, metrics = self.forward(
            cameras, images, names=names, is_training=False
        )

        self.log_dict(
            {
                "mse_v/coarse": metrics["mse_coarse"],
                "mse_v/fine": metrics["mse_fine"],
                "psnr_v/coarse": metrics["psnr_coarse"],
                "psnr_v/fine": metrics["psnr_fine"],
            },
            on_epoch=True,
            prog_bar=True,
        )

        render = out_dict["rgb_fine"].detach().cpu().numpy()[0]
        gt = images.detach().cpu().numpy()[0]
        render_coarse = out_dict["rgb_coarse"].detach().cpu().numpy()[0]

        final = np.concatenate([gt, render, render_coarse], axis=1)
        final = (final * 255).astype(np.uint8)

        final = Image.fromarray(final)
        if not os.path.exists("renders"):
            os.makedirs("renders")

        final.save(f"renders/{batch_idx}.png")

        return metrics["mse_coarse"] + metrics["mse_fine"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        return optimizer


def get_split(dir, split, datasets, batch_size, image_dim=(224, 224)):
    ds = NeRFDataLoader(
        dir,
        split=split,
        image_size=image_dim,
        datasets=datasets,
    )

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )


if __name__ == "__main__":
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    image_dim = (224, 224)

    datasets = [
        "chair",
        "drums",
        "hotdog",
        "lego",
    ]

    dl_train = get_split(
        "./data/nerf_synthetic",
        "train",
        datasets,
        1,
        image_dim,
    )

    dl_val = get_split(
        "./data/nerf_synthetic",
        "val",
        datasets,
        1,
        image_dim,
    )

    # model = LightningRadianceField.load_from_checkpoint(
    #     "/mnt/research/Projects/paranerf/checkpoints/boring_model.ckpt"
    # )

    model = LightningRadianceField()

    fname = "poly_model" if USE_POLY else "boring_model"

    trainer = pl.Trainer(
        max_epochs=100000,
        enable_progress_bar=True,
        logger=WandbLogger("nerf_base") if USE_WANDB else None,
        callbacks=[
            ModelCheckpoint(
                monitor="psnr/fine",
                mode="max",
                dirpath="checkpoints",
                filename=fname,
                every_n_epochs=5,
            )
        ],
        num_sanity_val_steps=1,
        limit_val_batches=1,
        check_val_every_n_epoch=1,
        log_every_n_steps=25,
        precision="16-mixed",
        gradient_clip_val=1.0,
    )
    trainer.fit(model, dl_train, dl_val)
