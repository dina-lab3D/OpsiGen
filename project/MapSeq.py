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


class MapSeq:
    funcs = {
        graph_builders.calculate_graph_desc: 100
    }

    def __init__(self, map_path, pdb_path, boxes_shape=(8,8,8)):
        self.map_path = map_path
        self.pdb_path = pdb_path
        # parser = Bio.PDB.PDBParser(PERMISSIVE=True, QUIET=True)
        # structure = parser.get_structure("7qti", self.pdb_path)
        # self.center = structure.center_of_mass()
        # structure.transform(rot=np.eye(3), tran=self.center)
        # print("Center of mass is ", self.center)
        # self.atoms = structure.get_atoms()
        with mrcfile.open(self.map_path) as protein:
            self.map_data = protein.data
        self.boxes_x_size = boxes_shape[0]
        self.boxes_y_size = boxes_shape[1]
        self.boxes_z_size = boxes_shape[2]
        self.good_boxes = None
        self.preprocess()

    def _divide_to_boxes(self):

        # count the number of boxes
        boxes_x_num = int(self.map_data.shape[0] / self.boxes_x_size)
        boxes_y_num = int(self.map_data.shape[1] / self.boxes_y_size)
        boxes_z_num = int(self.map_data.shape[2] / self.boxes_z_size)

        print("num x boxes", boxes_x_num)
        print("map shape x", self.map_data.shape[0])

        # initialize empty boxes
        self.boxes = [[[[] for _ in range(boxes_z_num)] for _ in range(boxes_y_num)] for _ in range(boxes_x_num)]
        for atom in self.atoms:
            atom_vec = atom.get_vector()
            if (int(atom_vec[0] / self.boxes_x_size) > boxes_x_num) or (
                    int(atom_vec[1] / self.boxes_y_size) > boxes_y_num) or (
                    int(atom_vec[2] / self.boxes_z_size) > boxes_z_num):
                continue
            self.boxes[int(atom_vec[0] / self.boxes_x_size) - 1][int(atom_vec[1] / self.boxes_y_size) - 1][
                int(atom_vec[2] / self.boxes_z_size) - 1].append(atom)

    def _find_good_boxes(self, threas):
        self.good_boxes = []
        for i, box2d in enumerate(self.boxes):
            for j, box1d in enumerate(box2d):
                for k, box in enumerate(box1d):
                    if len(box) > threas:
                        self.good_boxes.append((i, j, k))

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
        # self._divide_to_boxes()
        # self._find_good_boxes(15)

    def _create_descriptor_from_atoms(self, atoms, funcs_dict):
        res = np.array([])
        # while len(atoms) < max(funcs_dict.values()):
        #     atoms.append(random.choice(atoms))
        distmat = graph_builders.compute_distance(atoms)
        for func in funcs_dict:
            res = np.concatenate([res, func(atoms, funcs_dict[func], distmat)])

        return res

    def _get_close_atoms(self, point, threas):
        box_x = int(point[0] / self.boxes_x_size) - 1
        box_y = int(point[1] / self.boxes_y_size) - 1
        box_z = int(point[2] / self.boxes_z_size) - 1

        atoms = []
        for x in range(max(box_x - 1, 0), min(box_x + 2, self.map_data.shape[0] + 1)):
            for y in range(max(box_y - 1, 0), min(box_y + 2, self.map_data.shape[1] + 1)):
                for z in range(max(box_z - 1, 0), min(box_z + 2, self.map_data.shape[2] + 1)):
                    try:
                        for atom in self.boxes[x][y][z]:
                            if np.linalg.norm((atom.coord - point), np.inf) < threas:
                                atoms.append(atom)
                    except Exception as e:
                        breakpoint()

        return atoms

    def get_interesting_point(self):
        # x_val = random.choice(range(self.boxes_x_size))
        # y_val = random.choice(range(self.boxes_y_size))
        # z_val = random.choice(range(self.boxes_z_size))
        # good_box = random.choice(self.good_boxes)
        #
        # x_val += good_box[0] * self.boxes_x_size
        # y_val += good_box[1] * self.boxes_y_size
        # z_val += good_box[2] * self.boxes_z_size

        # return x_val, y_val, z_val
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
        # atoms = self._get_close_atoms(point, threas)
        # if len(atoms) == 0:
        #     return np.zeros((threas * 2, threas * 2, threas * 2)), np.zeros((sum(MapSeq.funcs.values()),))
        # atoms_desc = self._create_descriptor_from_atoms(atoms=atoms, funcs_dict=MapSeq.funcs)
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
