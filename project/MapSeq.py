import random
import time

import Bio.PDB
import mrcfile
import numpy as np

import graph_builders


class MapSeq:
    funcs = {
        graph_builders.compute_hydrophobic_feature: 10,
        graph_builders.compute_ionic_feature: 15,
        graph_builders.compute_hydrogen_feature: 15,
        graph_builders.compute_cation_pi_feature: 10,
        graph_builders.compute_disulfide_feature: 10
    }

    def __init__(self, map_path, pdb_path):
        self.map_path = map_path
        self.pdb_path = pdb_path
        parser = Bio.PDB.PDBParser(PERMISSIVE=True, QUIET=True)
        self.atoms = parser.get_structure("7qti", self.pdb_path).get_atoms()
        with mrcfile.open(self.map_path) as protein:
            self.map_data = protein.data
        self.good_boxes = None
        self.preprocess()

    def _divide_to_boxes(self):
        self.boxes_x_size = 8
        self.boxes_y_size = 8
        self.boxes_z_size = 8
        boxes_x_num = int(self.map_data.shape[0] / self.boxes_x_size)
        boxes_y_num = int(self.map_data.shape[1] / self.boxes_y_size)
        boxes_z_num = int(self.map_data.shape[2] / self.boxes_z_size)
        print(boxes_z_num)
        self.boxes = [[[[] for _ in range(boxes_z_num)] for _ in range(boxes_y_num)] for _ in range(boxes_x_num)]
        for atom in self.atoms:
            atom_vec = atom.get_vector()
            if (int(atom_vec[0] / self.boxes_x_size) > boxes_x_num) or (
                    int(atom_vec[1] / self.boxes_y_size) > boxes_y_num) or (
                    int(atom_vec[2] / self.boxes_z_size) > boxes_z_num):
                continue
            self.boxes[int(atom_vec[0] / self.boxes_x_size) - 1][int(atom_vec[1] / self.boxes_y_size) - 1][
                int(atom_vec[2] / self.boxes_z_size) - 1].append(atom)

        counter = 0

    def _find_good_boxes(self, threas):
        good_boxes = []
        for i, box2d in enumerate(self.boxes):
            for j, box1d in enumerate(box2d):
                for k, box in enumerate(box1d):
                    if len(box) > threas:
                        good_boxes.append((i, j, k))

        return good_boxes

    def preprocess(self):
        self._divide_to_boxes()
        self.good_boxes = self._find_good_boxes(15)

    def _create_descriptor_from_atoms(self, atoms, funcs_dict):
        res = np.array([])
        distmat = graph_builders.compute_distance(atoms)
        while len(atoms) < max(funcs_dict.values()):
            atoms.append(random.choice(atoms))
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
                    for atom in self.boxes[x][y][z]:
                        if np.linalg.norm((atom.coord - point), np.inf) < threas:
                            atoms.append(atom)

        return atoms

    def get_interesting_point(self):
        x_val = random.choice(range(self.boxes_x_size))
        y_val = random.choice(range(self.boxes_y_size))
        z_val = random.choice(range(self.boxes_z_size))
        good_box = random.choice(self.good_boxes)

        x_val += good_box[0] * self.boxes_x_size
        y_val += good_box[1] * self.boxes_y_size
        z_val += good_box[2] * self.boxes_z_size

        return x_val, y_val, z_val

    @staticmethod
    def _to_shape(array, wanted_shape):
        z_, y_, x_ = wanted_shape
        z, y, x = array.shape
        z_pad = (z_ - z)
        y_pad = (y_ - y)
        x_pad = (x_ - x)
        return np.pad(array, ((z_pad // 2, z_pad // 2 + z_pad % 2),
                              (y_pad // 2, y_pad // 2 + y_pad % 2),
                              (x_pad // 2, x_pad // 2 + x_pad % 2)),
                      mode='edge')

    def generate_descriptors(self, point, threas):
        atoms = self._get_close_atoms(point, threas)
        if len(atoms) == 0:
            return np.zeros((threas * 2, threas * 2, threas * 2)), np.zeros((sum(MapSeq.funcs.values()),))
        atoms_desc = self._create_descriptor_from_atoms(atoms=atoms, funcs_dict=MapSeq.funcs)

        map_desc = self.map_data[point[0] - threas: point[0] + threas, point[1] - threas: point[1] + threas,
                   point[2] - threas: point[2] + threas]
        map_desc = self._to_shape(map_desc, (threas * 2, threas * 2, threas * 2))

        return map_desc, atoms_desc
