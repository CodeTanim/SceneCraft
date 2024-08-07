# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# This script is based on an original implementation

import os
import subprocess
import numpy as np
import json
import sys
import math
import cv2
import shutil


class ColmapSolver:
    def __init__(self, aabb_scale, keep_colmap_coords, skip_early):
        self.aabb_scale = aabb_scale
        self.keep_colmap_coords = keep_colmap_coords
        self.skip_early = skip_early

    def run_colmap(self, colmap_path, input_path, output_path):
        print("Running SfM...")

        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        colmap_cmd = f"{colmap_path} feature_extractor --database_path '{input_path}/database.db' --image_path '{input_path}'"
        subprocess.run(colmap_cmd, shell=True, check=True)

        colmap_cmd = f"{colmap_path} exhaustive_matcher --database_path '{input_path}/database.db'"
        subprocess.run(colmap_cmd, shell=True, check=True)

        os.makedirs(f"{output_path}/sparse", exist_ok=True)

        colmap_cmd = f"{colmap_path} mapper --database_path '{input_path}/database.db' --image_path '{input_path}' --output_path '{output_path}/sparse'"
        subprocess.run(colmap_cmd, shell=True, check=True)

        colmap_cmd = f"{colmap_path} model_converter --input_path '{output_path}/sparse/0' --output_path '{output_path}' --output_type TXT"
        subprocess.run(colmap_cmd, shell=True, check=True)

        print("End of running SfM...")

    def variance_of_laplacian(self, image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def sharpness(self, imagePath):
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = self.variance_of_laplacian(gray)
        return fm

    def qvec2rotmat(self, qvec):
        return np.array(
            [
                [
                    1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                    2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                    2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
                ],
                [
                    2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                    1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                    2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
                ],
                [
                    2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                    2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                    1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
                ],
            ]
        )

    def rotmat(self, a, b):
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = np.dot(a, b)
        # handle exception for the opposite direction input
        if c < -1 + 1e-10:
            return self.rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))

    def closest_point_2_lines(self, oa, da, ob, db):
        # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
        da = da / np.linalg.norm(da)
        db = db / np.linalg.norm(db)
        c = np.cross(da, db)
        denom = np.linalg.norm(c) ** 2
        t = ob - oa
        ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
        tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
        if ta > 0:
            ta = 0
        if tb > 0:
            tb = 0
        return (oa + ta * da + ob + tb * db) * 0.5, denom

    def clean_up_files(self, input_path, output_path):
        unwanted_files = ["cameras.txt", "images.txt", "points3D.txt", "database.db"]
        for file_name in unwanted_files:
            file_path = os.path.join(input_path, file_name)
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)

            file_path = os.path.join(output_path, file_name)
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)

        sparse_dir = os.path.join(output_path, "sparse")
        if os.path.exists(sparse_dir):
            shutil.rmtree(sparse_dir)

        with open(os.path.join(output_path, "transforms.json"), "r") as f:
            transforms = json.load(f)

        # replace the input path with the path relative to the json.
        # ./data/lego/train/r_99.png -> ./train/r_99.png
        for frame in transforms["frames"]:
            newpath = os.path.relpath(frame["file_path"], output_path)
            frame["file_path"] = "./" + newpath

        transforms_name = f"transforms_{os.path.basename(input_path)}.json"
        with open(os.path.join(output_path, transforms_name), "w") as f:
            json.dump(transforms, f, indent=4)

        # remove the original transforms.json
        os.remove(os.path.join(output_path, "transforms.json"))

    def generate_transforms_json(self, output_path, input_path):
        cameras = {}
        with open(os.path.join(output_path, "cameras.txt"), "r") as f:
            camera_angle_x = math.pi / 2
            for line in f:
                if line[0] == "#":
                    continue
                els = line.split(" ")
                camera = {}
                camera_id = int(els[0])
                camera["w"] = float(els[2])
                camera["h"] = float(els[3])
                camera["fl_x"] = float(els[4])
                camera["fl_y"] = float(els[4])
                camera["k1"] = 0
                camera["k2"] = 0
                camera["k3"] = 0
                camera["k4"] = 0
                camera["p1"] = 0
                camera["p2"] = 0
                camera["cx"] = camera["w"] / 2
                camera["cy"] = camera["h"] / 2
                camera["is_fisheye"] = False
                if els[1] == "SIMPLE_PINHOLE":
                    camera["cx"] = float(els[5])
                    camera["cy"] = float(els[6])
                elif els[1] == "PINHOLE":
                    camera["fl_y"] = float(els[5])
                    camera["cx"] = float(els[6])
                    camera["cy"] = float(els[7])
                elif els[1] == "SIMPLE_RADIAL":
                    camera["cx"] = float(els[5])
                    camera["cy"] = float(els[6])
                    camera["k1"] = float(els[7])
                elif els[1] == "RADIAL":
                    camera["cx"] = float(els[5])
                    camera["cy"] = float(els[6])
                    camera["k1"] = float(els[7])
                    camera["k2"] = float(els[8])
                elif els[1] == "OPENCV":
                    camera["fl_y"] = float(els[5])
                    camera["cx"] = float(els[6])
                    camera["cy"] = float(els[7])
                    camera["k1"] = float(els[8])
                    camera["k2"] = float(els[9])
                    camera["p1"] = float(els[10])
                    camera["p2"] = float(els[11])
                elif els[1] == "SIMPLE_RADIAL_FISHEYE":
                    camera["is_fisheye"] = True
                    camera["cx"] = float(els[5])
                    camera["cy"] = float(els[6])
                    camera["k1"] = float(els[7])
                elif els[1] == "RADIAL_FISHEYE":
                    camera["is_fisheye"] = True
                    camera["cx"] = float(els[5])
                    camera["cy"] = float(els[6])
                    camera["k1"] = float(els[7])
                    camera["k2"] = float(els[8])
                elif els[1] == "OPENCV_FISHEYE":
                    camera["is_fisheye"] = True
                    camera["fl_y"] = float(els[5])
                    camera["cx"] = float(els[6])
                    camera["cy"] = float(els[7])
                    camera["k1"] = float(els[8])
                    camera["k2"] = float(els[9])
                    camera["k3"] = float(els[10])
                    camera["k4"] = float(els[11])
                else:
                    print("Unknown camera model ", els[1])
                camera["camera_angle_x"] = (
                    math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
                )
                camera["camera_angle_y"] = (
                    math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
                )
                camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
                camera["fovy"] = camera["camera_angle_y"] * 180 / math.pi

                print(
                    f"camera {camera_id}:\n\tres={camera['w'], camera['h']}\n\tcenter={camera['cx'], camera['cy']}\n\tfocal={camera['fl_x'], camera['fl_y']}\n\tfov={camera['fovx'], camera['fovy']}\n\tk={camera['k1'], camera['k2']} p={camera['p1'], camera['p2']} "
                )
                cameras[camera_id] = camera

        if len(cameras) == 0:
            print("No cameras found!")
            sys.exit(1)

        with open(os.path.join(output_path, "images.txt"), "r") as f:
            i = 0
            bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
            if len(cameras) == 1:
                camera = cameras[camera_id]
                out = {
                    "camera_angle_x": camera["camera_angle_x"],
                    "camera_angle_y": camera["camera_angle_y"],
                    "fl_x": camera["fl_x"],
                    "fl_y": camera["fl_y"],
                    "k1": camera["k1"],
                    "k2": camera["k2"],
                    "k3": camera["k3"],
                    "k4": camera["k4"],
                    "p1": camera["p1"],
                    "p2": camera["p2"],
                    "is_fisheye": camera["is_fisheye"],
                    "cx": camera["cx"],
                    "cy": camera["cy"],
                    "w": camera["w"],
                    "h": camera["h"],
                    "aabb_scale": self.aabb_scale,
                    "frames": [],
                }
            else:
                out = {"frames": [], "aabb_scale": self.aabb_scale}

            up = np.zeros(3)
            for line in f:
                line = line.strip()
                if line[0] == "#":
                    continue
                i = i + 1
                if i < self.skip_early * 2:
                    continue
                if i % 2 == 1:
                    elems = line.split(
                        " "
                    )  # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                    image_rel = os.path.relpath(input_path)
                    name = str(f"./{image_rel}/{'_'.join(elems[9:])}")
                    b = self.sharpness(name)
                    print(name, "sharpness=", b)
                    image_id = int(elems[0])
                    qvec = np.array(tuple(map(float, elems[1:5])))
                    tvec = np.array(tuple(map(float, elems[5:8])))
                    R = self.qvec2rotmat(-qvec)
                    t = tvec.reshape([3, 1])
                    m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                    c2w = np.linalg.inv(m)
                    if not self.keep_colmap_coords:
                        c2w[0:3, 2] *= -1  # flip the y and z axis
                        c2w[0:3, 1] *= -1
                        c2w = c2w[[1, 0, 2, 3], :]
                        c2w[2, :] *= -1  # flip whole world upside down

                        up += c2w[0:3, 1]

                    frame = {
                        "file_path": name,
                        "sharpness": b,
                        "transform_matrix": c2w,
                        "R": R,
                        "t": t,
                    }
                    if len(cameras) != 1:
                        frame.update(cameras[int(elems[8])])
                    out["frames"].append(frame)
        nframes = len(out["frames"])

        if self.keep_colmap_coords:
            flip_mat = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
            )

            for f in out["frames"]:
                f["transform_matrix"] = np.matmul(
                    f["transform_matrix"], flip_mat
                )  # flip cameras (it just works)
        else:
            # don't keep colmap coords - reorient the scene to be easier to work with

            up = up / np.linalg.norm(up)
            print("up vector was", up)
            R = self.rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
            R = np.pad(R, [0, 1])
            R[-1, -1] = 1

            for f in out["frames"]:
                f["transform_matrix"] = np.matmul(
                    R, f["transform_matrix"]
                )  # rotate up to be the z axis

            # find a central point they are all looking at
            print("computing center of attention...")
            totw = 0.0
            totp = np.array([0.0, 0.0, 0.0])
            for f in out["frames"]:
                mf = f["transform_matrix"][0:3, :]
                for g in out["frames"]:
                    mg = g["transform_matrix"][0:3, :]
                    p, w = self.closest_point_2_lines(
                        mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2]
                    )
                    if w > 0.00001:
                        totp += p * w
                        totw += w
            if totw > 0.0:
                totp /= totw
            print(totp)  # the cameras are looking at totp
            for f in out["frames"]:
                f["transform_matrix"][0:3, 3] -= totp

            avglen = 0.0
            for f in out["frames"]:
                avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
            avglen /= nframes
            print("avg camera distance from origin", avglen)
            for f in out["frames"]:
                f["transform_matrix"][0:3, 3] *= 4.0 / avglen  # scale to "nerf sized"

        for f in out["frames"]:
            f["transform_matrix"] = f["transform_matrix"].tolist()
            f["R"] = f["R"].tolist()
            f["t"] = f["t"].tolist()

        print(nframes, "frames")

        # Calculate averages
        avg_values = {
            "camera_angle_x": sum(frame["camera_angle_x"] for frame in out["frames"])
            / nframes,
            "camera_angle_y": sum(frame["camera_angle_y"] for frame in out["frames"])
            / nframes,
            "fl_x": sum(frame["fl_x"] for frame in out["frames"]) / nframes,
            "fl_y": sum(frame["fl_y"] for frame in out["frames"]) / nframes,
            "k1": sum(frame["k1"] for frame in out["frames"]) / nframes,
            "k2": sum(frame["k2"] for frame in out["frames"]) / nframes,
            "p1": sum(frame["p1"] for frame in out["frames"]) / nframes,
            "p2": sum(frame["p2"] for frame in out["frames"]) / nframes,
            "cx": sum(frame["cx"] for frame in out["frames"]) / nframes,
            "cy": sum(frame["cy"] for frame in out["frames"]) / nframes,
            "w": sum(frame["w"] for frame in out["frames"]) / nframes,
            "h": sum(frame["h"] for frame in out["frames"]) / nframes,
        }

        # Update contents of frames
        new_frames = [
            {
                "file_path": frame["file_path"],
                "sharpness": frame["sharpness"],
                "transform_matrix": frame["transform_matrix"],
                "R": frame["R"],
                "t": frame["t"],
            }
            for frame in out["frames"]
        ]

        # Construct new JSON format
        transforms = {
            **avg_values,
            "aabb_scale": out["aabb_scale"],
            "frames": new_frames,
        }

        print(f"writing {output_path}/transforms.json...")
        with open(f"{output_path}/transforms.json", "w") as outfile:
            json.dump(transforms, outfile, indent=2)
        print(f"End of writing {output_path}/transforms.json")

    def solve(self, colmap_path, input_path, output_path):
        self.run_colmap(colmap_path, input_path, output_path)
        self.generate_transforms_json(output_path, input_path)
        self.clean_up_files(input_path, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Colmap to NeRF")
    parser.add_argument(
        "--colmap_path",
        type=str,
        help="Path to the colmap executable",
        default="/usr/bin/colmap",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the input directory containing images",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to the output directory",
    )

    args = parser.parse_args()

    camera_param_solver = ColmapSolver(
        aabb_scale=32.0, keep_colmap_coords=True, skip_early=0
    )

    camera_param_solver.solve(
        colmap_path=args.colmap_path,
        input_path=args.input_path,
        output_path=args.output_path,
    )
