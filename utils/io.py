"""Geometry data loading and I/O utilities."""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_geometry_data(image_paths):
    depths, poses = [], []
    intrinsics, alignment_matrix = None, None

    for p in image_paths:
        path = Path(p)
        scene_dir = path.parent.parent
        scene_id = scene_dir.name

        if intrinsics is None:
            intrinsics_path = scene_dir / f"{scene_id}.txt"
            if intrinsics_path.exists():
                params = {}
                with open(intrinsics_path, 'r') as f:
                    for line in f:
                        if '=' in line:
                            k, v = line.split('=')
                            params[k.strip()] = v.strip()

                intrinsics = (float(params['fx_color']), float(params['fy_color']),
                            float(params['mx_color']), float(params['my_color']))
                # The file's colour intrinsics are for the native colour size.
                # Scale them to the frames actually passed (e.g. color_640x480).
                if 'colorWidth' in params and 'colorHeight' in params:
                    w, h = Image.open(path).size
                    sx, sy = w / float(params['colorWidth']), h / float(params['colorHeight'])
                    intrinsics = (intrinsics[0] * sx, intrinsics[1] * sy,
                                  intrinsics[2] * sx, intrinsics[3] * sy)

                # Use the dataset's provided alignment matrix instead of calculating one
                if 'axisAlignment' in params:
                    alignment_matrix = np.array([float(x) for x in params['axisAlignment'].split()]).reshape(4, 4)

        depth_path = scene_dir / "depth" / f"{path.stem}.png"
        pose_path = scene_dir / "pose" / f"{path.stem}.txt"

        if depth_path.exists(): depths.append(Image.open(depth_path))
        if pose_path.exists(): poses.append(np.loadtxt(pose_path).reshape(4, 4))

    return depths, poses, intrinsics, alignment_matrix


class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._devnull = open(os.devnull, 'w')
        sys.stdout = self._devnull
        sys.stderr = self._devnull
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore first, then close our own handle, so an exception in the
        # body can't leak the devnull fd or leave stdio pointing at a closed file.
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._devnull.close()
        return False
