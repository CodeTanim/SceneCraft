import numpy as np
from .datasets import NeRFDataset
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch3d.renderer import PerspectiveCameras
import numpy as np
from data_loading.datasets import NeRFDataset, ALL_DATASETS
from torch.utils.data.dataloader import default_collate


class _NeRFDataloader:
    def __init__(self, base_paths, datasets):
        self.base_paths = base_paths
        self.datasets = datasets
        self.dataset_objects = self._init_datasets()

    def _init_datasets(self):
        dataset_objects = {}
        for base_path, dataset in zip(self.base_paths, self.datasets):
            dataset_objects[dataset] = NeRFDataset(base_path, dataset)
        return dataset_objects

    def get_random_sample(self, split):
        dataset_name = np.random.choice(self.datasets)
        dataset = self.dataset_objects[dataset_name]
        base_transforms, frames, images = dataset.get_split(split)
        return dataset_name, base_transforms, frames, images


def collate_fn(batch):
    cams, imgs, masks, names = zip(*batch)

    stack_focal = torch.concat([cam.focal_length for cam in cams], dim=0)
    stack_principal = torch.concat([cam.principal_point for cam in cams], dim=0)
    stack_R = torch.concat([cam.R for cam in cams], dim=0)
    stack_T = torch.concat([cam.T for cam in cams], dim=0)

    cams = PerspectiveCameras(
        focal_length=stack_focal.float(),
        principal_point=stack_principal.float(),
        R=stack_R.float(),
        T=stack_T.float(),
    )

    return cams, torch.stack(imgs).float(), torch.stack(masks).float(), names


class NeRFDataLoader(Dataset):
    def __init__(
        self, base_path, datasets=ALL_DATASETS, split="", image_size=(224, 224)
    ):
        """
        Custom dataset for loading NeRF data.

        :param base_path: Path to the dataset directory.
        :param datasets: List of dataset names.
        :param split: Dataset split, e.g., 'train', 'test'.
        :param image_size: Tuple specifying the image dimensions.
        """
        self.base_path = base_path
        self.datasets = datasets
        self.split = split
        self.image_size = image_size
        self.data = []

        for dataset in datasets:
            nerf_dataset = NeRFDataset(base_path, dataset)
            base, frames, (images, masks) = nerf_dataset.get_split(split, image_size)
            for i in range(len(images)):
                if "R" in frames:
                    camera = PerspectiveCameras(
                        focal_length=torch.tensor([[base["fl_x"], base["fl_y"]]]),
                        principal_point=torch.tensor([[base["cx"], base["cy"]]]),
                        R=torch.from_numpy(frames["R"][i].reshape(1, 3, 3)),
                        T=torch.from_numpy(frames["t"][i].reshape(1, 3)),
                    )
                else:
                    mat = np.linalg.inv(frames["transform_matrix"][i])
                    R = mat[:3, :3]
                    T = mat[:3, 3]
                    camera = PerspectiveCameras(
                        R=torch.from_numpy(R.reshape(1, 3, 3)).float(),
                        T=torch.from_numpy(T.reshape(1, 3)).float(),
                    )

                self.data.append(
                    (
                        camera,
                        torch.from_numpy(images[i] / 255.0).float(),
                        torch.from_numpy(masks[i] / 255.0).float(),
                        dataset,
                    )
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        camera, image, mask, dataset = self.data[idx]
        return camera, image, mask, dataset

    def fetch_rt_values(self, idx):
        """
        Fetches the rotation (R) and translation (T) values for the camera at the given index.

        Parameters:
        - idx: Index of the camera in the dataset.

        Returns:
        - R: Rotation matrix of the camera.
        - T: Translation vector of the camera.
        """
        camera, _ = self.data[idx]
        R = camera.R.cpu().numpy()
        T = camera.T.cpu().numpy()
        return R, T
