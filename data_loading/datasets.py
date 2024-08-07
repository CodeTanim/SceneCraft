import json
import os
from PIL import Image
import numpy as np


ALL_DATASETS = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]


class NeRFDataset:
    def __init__(self, base_path, dataset):
        self.base_path = base_path
        self.dataset = dataset

    def _load_json(self, split):
        with open(
            os.path.join(self.base_path, self.dataset, f"transforms_{split}.json"), "r"
        ) as file:
            data = json.load(file)
        return data

    def get_split(self, split, size=None):
        # Load the JSON data
        data = self._load_json(split)
        # print(data)

        # Extract base transforms, which are common across all frames
        base_features = list(data.keys())
        base_features.remove("frames")

        base_transforms = {feature: data[feature] for feature in base_features}

        # Extract frames data
        frames = data["frames"]
        # print(frames)

        # Load images corresponding to the frames
        images = []
        for frame in frames:
            # Construct the correct file path
            relative_image_path = frame["file_path"].lstrip("./") + ".png"
            image_path = os.path.join(self.base_path, self.dataset, relative_image_path)
            try:
                with Image.open(image_path) as img:
                    if size is not None:
                        img = img.resize(size)

                    image_np = np.array(img)
                    images.append(image_np)

            except FileNotFoundError:
                print(f"File not found: {image_path}")
                continue  # Skip this image

        frames_transform = np.array([i["transform_matrix"] for i in frames])

        if "R" in frames[0]:
            frames_R = np.array([i["R"] for i in frames])
            frames_t = np.array([i["t"] for i in frames])

            frames = {
                "transform_matrix": frames_transform,
                "R": frames_R,
                "t": frames_t,
            }
        else:
            frames = {"transform_matrix": frames_transform}

        images = np.array(images)

        if len(images.shape) == 4:
            images = (images[..., :3], images[..., 3])
        else:
            images = (images, None)

        return (
            base_transforms,
            frames,
            images,
        )


if __name__ == "__main__":

    # Example usage:

    # NOTE: the call here assumes that the nerf_synthetic data folder exists in the same directory as the .py file.
    # I didnt push the data folder.
    chair_dataset = NeRFDataset(base_path="nerf_synthetic", dataset="chair")
    intrinsics_train, frames_train, images_train = chair_dataset.get_split("train")

    # print(intrinsics_train)
    # print(type(frames_train))

    # print(frames_train[0])
    # print(images_train[0])
    # print(chair_dataset)

    print(images_train[0])

    # print(images_train)

    # Example usage to get data with masks on.
    intrinsics_train2, frames_train2, images_train2, masks_train = (
        chair_dataset.get_split("train", masks=True)
    )
