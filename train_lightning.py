import torch
import torch.nn as nn
import numpy as np
from models.multimodel import MultiNeRF
from models.utils import sample_images_at_locs
import pytorch_lightning as pl
from data_loading.data_loader import NeRFDataLoader, collate_fn
import os
from PIL import Image

torch.set_float32_matmul_precision("medium")


class LightningNeRF(pl.LightningModule):
    def __init__(self, image_dim=(224, 224)):
        super().__init__()

        self.model = MultiNeRF(
            hidden_dim=256, adapter_rank=128, dims=8192, image_dim=image_dim, n_layers=4
        ).cuda()
        self.loss_fn = nn.MSELoss().cuda()
        self.iters = 0

    def forward(self, cameras, tasks):
        return self.model(cameras, tasks)

    def training_step(self, batch, batch_idx):
        cameras, images, masks, tasks = batch
        loss = 0
        for cam, img, mask, task in zip(cameras, images, masks, tasks):
            cam = cam.cuda()
            img = img.cuda().unsqueeze(0)

            image, sil, ray_fine = self.model(cam, [task])
            colors_rays = sample_images_at_locs(img, ray_fine.xys)
            loss += self.loss_fn(image, colors_rays) / len(cameras)

        with torch.no_grad():
            if self.iters % 125 == 0:
                render = self.model.render_inference(
                    cameras[0], tasks=[tasks[0]]
                ).squeeze()
                img = images[0].cpu().numpy()
                rend = render.cpu().numpy()
                stacked = np.hstack((img, rend))
                image = Image.fromarray((stacked * 255).astype(np.uint8))
                image.save("render.png")

        self.log("train_loss", loss.item())

        self.iters += 1
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        return optimizer


if __name__ == "__main__":
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    device = torch.device("cuda:0")

    image_dim = (224, 224)

    ds = NeRFDataLoader(
        "./data/nerf_synthetic",
        split="train",
        image_size=image_dim,
        datasets=["chair", "drums", "hotdog", "lego", "mic"],
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )

    model = LightningNeRF()
    trainer = pl.Trainer(
        max_epochs=100000,
        enable_progress_bar=True,
        logger=WandbLogger("lightning_logs"),
        callbacks=[
            ModelCheckpoint(
                monitor="train_loss",
                dirpath="checkpoints",
                filename="model_stoptim",
                every_n_epochs=5,
            )
        ],
        precision="16-mixed",
    )
    trainer.fit(model, dl)
