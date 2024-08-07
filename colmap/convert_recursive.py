import os, sys
from colmap2nerf import ColmapSolver


def has_image(files):
    for f in files:
        if f.endswith(".jpg") or f.endswith(".png"):
            return True

    return False


def image_folder_iter(path):
    # is an iterator over all image folders. only return the folders that contain images
    # should be recursive
    for root, dirs, files in os.walk(path):
        # make sure it contains images
        if has_image(files):
            yield root


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--colmap",
        type=str,
        required=False,
        help="Path to the colmap binary",
        default="/usr/bin/colmap",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input folder"
    )

    args = parser.parse_args()
    camera_param_solver = ColmapSolver(
        aabb_scale=32.0, keep_colmap_coords=True, skip_early=0
    )

    input_folder = args.input
    for folder in image_folder_iter(input_folder):
        # output folder should be the parent folder of the input folder
        output = os.path.dirname(folder)
        camera_param_solver.solve(
            colmap_path=args.colmap,
            input_path=folder,
            output_path=output,
        )
        print(f"Processed {folder}")
