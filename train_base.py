import torch
import torch.nn as nn
import numpy as np
from models.base_nerf import (
    NeRF,
    render_model_inference,
    render_model_train,
    get_inference_renderer,
    get_training_renderer,
)
from data_loading.data_loader import PerspectiveCameras
from models.utils import sample_images_at_locs
import torch.nn.functional as F
import pytorch_lightning as pl
from data_loading.data_loader import NeRFDataLoader, collate_fn
from schedule_free_adam import AdamWScheduleFree
import os
from PIL import Image

USE_WANDB = True

if USE_WANDB:
    import wandb

torch.set_float32_matmul_precision("medium")


class LightningNeRF(pl.LightningModule):
    def __init__(
        self,
        image_height,
        image_width,
        backbone_layers=8,
        head_layers=4,
        hidden_dim=768,
        fine_pts=128,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = torch.compile(
            NeRF(backbone_layers, head_layers, hidden_dim, fine_pts)
        )
        self.loss_fn = F.mse_loss

        self.train_renderer = get_training_renderer(128)
        self.inference_renderer = get_inference_renderer(image_height, image_width)

    def forward(self, rays, **kwargs):
        return self.model(rays, **kwargs)

    def step(self, cameras, images, masks, names):
        img, sil, fine = render_model_train(cameras, self.model, self.train_renderer)

        colors = sample_images_at_locs(images, fine.xys)
        sils = sample_images_at_locs(masks.unsqueeze(-1), fine.xys)
        loss_color = self.loss_fn(img.float(), colors.float())

        return loss_color

    def training_step(self, batch, batch_idx):
        cameras, images, masks, names = batch
        loss = self.step(cameras, images, masks, names)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        cameras, images, masks, names = batch
        cameras = PerspectiveCameras(
            focal_length=cameras.focal_length[:1].float(),
            principal_point=cameras.principal_point[:1].float(),
            R=cameras.R[:1].float(),
            T=cameras.T[:1].float(),
        ).to(cameras.focal_length.device)
        image, sil = render_model_inference(
            cameras, self.model, self.inference_renderer
        )
        image = (image[0].cpu().numpy() * 255).astype(np.uint8)

        # stack image with the orignal
        tgt_image = (images[0].cpu().numpy() * 255).astype(np.uint8)
        image = np.concatenate([tgt_image, image], axis=1)
        image = Image.fromarray(image)

        image.save("render.png")

        if USE_WANDB:
            images = wandb.Image(
                np.array(image), caption="Actual Image, Rendered Image"
            )

            wandb.log({"val_images": images, "val_masks": masks})
        return 0

    def configure_optimizers(self):
        optimizer = AdamWScheduleFree(self.model.parameters(), lr=1e-3)
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

    dl_train = get_split(
        "./data/nerf_synthetic",
        "train",
        ["chair"],
        4,
        image_dim,
    )

    dl_val = get_split(
        "./data/nerf_synthetic",
        "train",
        ["chair"],
        4,
        image_dim,
    )

    model = LightningNeRF(image_dim[0], image_dim[1])

    trainer = pl.Trainer(
        max_epochs=100000,
        enable_progress_bar=True,
        logger=WandbLogger("nerf_base") if USE_WANDB else None,
        callbacks=[
            ModelCheckpoint(
                monitor="train_loss",
                dirpath="checkpoints",
                filename="boring_model",
                every_n_epochs=5,
            )
        ],
        num_sanity_val_steps=1,
        limit_val_batches=1,
        check_val_every_n_epoch=25,
        log_every_n_steps=25,
        # precision="16-mixed",
        gradient_clip_val=1.0,
    )
    trainer.fit(model, dl_train, dl_val)
