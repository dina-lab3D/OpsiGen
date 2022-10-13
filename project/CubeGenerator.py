import random
import time

import Bio.PDB
import mrcfile
import numpy as np

import graph_builders
import torch
from scipy import ndimage
from scipy.spatial.transform import Rotation as R
from skimage.feature import canny
import scipy.ndimage.filters as filters

CRYO_FILE_TEMPLATE_A = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/a{}.mrc"


class CubeGenerator:

    def __init__(self, map_path):
        self.map_path = map_path
        with mrcfile.open(self.map_path) as protein:
            self.map_data = protein.data
        self.preprocess()

    @staticmethod
    def find_edges(map):
        blured_img = filters.gaussian_filter(map, sigma=5)
        laplace_img = filters.laplace(blured_img)
        counts, vals = np.histogram(laplace_img, bins=25)
        t = 0
        for val in vals[::-1]:
            if val > 0:
                t = val
        return np.argwhere(laplace_img > t)

    def preprocess(self):
        self.edges = self.find_edges(self.map_data)

    def get_interesting_point(self):
        return random.choice(self.edges)

    @staticmethod
    def _to_shape(array, wanted_shape):
        z_, y_, x_ = wanted_shape
        z, y, x = array.shape
        z_pad = (z_ - z)
        y_pad = (y_ - y)
        x_pad = (x_ - x)
        return np.pad(array, ((z_pad // 2, z_pad // 2 + z_pad % 2),
                              (y_pad // 2, y_pad // 2 + y_pad % 2),
                              (x_pad // 2, x_pad // 2 + x_pad % 2))
                      )

    def generate_descriptors(self, point, threas, to_rotate=None):
        atoms_desc = None

        map_desc = self.map_data[point[0] - threas: point[0] + threas, point[1] - threas: point[1] + threas,
                   point[2] - threas: point[2] + threas]
        if not (to_rotate is None):
            degrees = to_rotate.as_euler('xyz', degrees=True)
            z_rotated = ndimage.rotate(map_desc, degrees[2], (0, 1), reshape=False)
            yz_rotated = ndimage.rotate(z_rotated, degrees[1], (0, 2), reshape=False)
            xyz_rotated = ndimage.rotate(yz_rotated, degrees[0], (1, 2), reshape=False)
        else:
            xyz_rotated = map_desc
        map_desc = self._to_shape(xyz_rotated, (threas * 2, threas * 2, threas * 2))

        return torch.tensor(map_desc, requires_grad=True), atoms_desc
