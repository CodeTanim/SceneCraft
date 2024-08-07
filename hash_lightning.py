import torch
import torch.nn as nn
import numpy as np
from models.hash_multimodel import HashNeRFWrapper, total_variation_loss
from models.utils import sample_images_at_locs
import pytorch_lightning as pl
from data_loading.data_loader import NeRFDataLoader, collate_fn
import os
from PIL import Image

torch.set_float32_matmul_precision("medium")


class LightningNeRF(pl.LightningModule):
    def __init__(self, image_dim=(224, 224)):
        super().__init__()

        self.model = HashNeRFWrapper(
            image_dim, 128, n_layers_mlp=3, n_layers_heads=5
        ).cuda()
        self.loss_fn = nn.MSELoss().cuda()
        self.iters = 0

    def training_step(self, batch, batch_idx):
        cameras, images, tasks = batch
        loss = 0
        for cam, img, task in zip(cameras, images, tasks):
            cam = cam.cuda()
            img = img.cuda()
            img = img.unsqueeze(0)
            image, sil, ray_fine = self.model(cam)
            colors_rays = sample_images_at_locs(img, ray_fine.xys)
            loss += self.loss_fn(image, colors_rays) / len(cameras)
            loss += (
                sum(
                    [
                        total_variation_loss(
                            self.model.model.hash_embedder.embeddings[i],
                            16,
                            512,
                            i,
                            19,
                            16,
                        )
                        for i in range(16)
                    ]
                )
                * 1e-6
            )

        if self.iters % 25 == 0:
            render = self.model.render_inference(cameras[0]).squeeze()
            img = images[0].cpu().numpy()
            rend = render.cpu().numpy()
            stacked = np.hstack((img, rend))
            image = Image.fromarray((stacked * 255).astype(np.uint8))
            image.save("render.png")

        self.log("train_loss", loss.item())

        self.iters += 1
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, eps=1e-15)
        return optimizer


if __name__ == "__main__":
    from pytorch_lightning.callbacks import ModelCheckpoint

    from pytorch_lightning.loggers import WandbLogger

    device = torch.device("cuda:0")

    image_dim = (224, 224)

    ds = NeRFDataLoader(
        "./data/nerf_synthetic", split="train", image_size=image_dim, datasets=["lego"]
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    model = LightningNeRF(image_dim=image_dim)
    trainer = pl.Trainer(
        max_epochs=5000,
        enable_progress_bar=True,
        # logger=WandbLogger("lightning_logs"),
        callbacks=[
            ModelCheckpoint(
                monitor="train_loss",
                dirpath="checkpoints",
                filename="model_stoptim",
                every_n_epochs=5,
            )
        ],
    )
    trainer.fit(model, dl)
