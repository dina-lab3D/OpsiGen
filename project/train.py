import random

from CubeGenerator import CubeGenerator
import time
import models
import torch.optim as optim
import torch
import pickle
import numpy as np
import mrcfile
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from train_utils import PerformanceStats, TrainerData

CRYO_FILE_TEMPLATE_A = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/a{}.mrc"
CRYO_FILE_TEMPLATE_B = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/b{}.mrc"
CRYO_FILE_TEMPLATE_C = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/c{}.mrc"
CRYO_FILE_TEMPLATE_D = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/d{}.mrc"


class Trainer:

    def __init__(self, trainer_data):
        self.data = trainer_data
        self.cube_generators = None

        self.init_cube_generators()

    def init_cube_generators(self):
        self.cube_generators = {
            CubeGenerator(path): CubeGenerator(self.data.path_dict[path]) for path in self.data.path_dict
        }

    def init_model(self):
        if self.data.to_load:
            cryo_net = pickle.load(open(self.data.model_path, 'rb'))
        else:
            cryo_net = models.CryoNet(self.data.threas).float()

        cryo_net.to(torch.device("cuda:0"))
        cryo_net.zero_grad()
        optimizer = optim.Adam(list(cryo_net.parameters()), lr=self.data.lr)

        return cryo_net, optimizer

    def calculate_descriptors(self, cube_generators, num_points=20):
        is_valid_points = False

        descs_original = []
        descs_fake = []

        mat1 = R.random()

        counter = 0
        while not is_valid_points:
            counter += 1
            descs_original = []
            descs_fake = []
            points = []
            for cube_generator in cube_generators:
                points += [cube_generator.get_interesting_point() for _ in range(num_points)]

            mask = np.triu_indices(num_points, num_points - 1)
            dists = cdist(np.array(points), np.array(points))[mask]
            if np.any(dists < self.data.threas / 2):
                continue

            for p in points:
                descs_original.append(cube_generators[0].generate_descriptors(p, self.data.threas))
                descs_fake.append(cube_generators[1].generate_descriptors(p, self.data.threas))  # , to_rotate=mat1)

            is_valid_points = True

        return torch.stack(descs_original), torch.stack(descs_fake)

    def train(self):

        # cube_generator1 = CubeGenerator.CubeGenerator(self.data.path1)
        # cube_generator2 = CubeGenerator.CubeGenerator(self.data.path2)
        cryo_net, optimizer = self.init_model()

        stats = PerformanceStats()

        while True:
            cube_generator_a = random.choice(list(self.cube_generators.keys()))
            cube_generator_b = self.cube_generators[cube_generator_a]
            descs = self.calculate_descriptors((cube_generator_a, cube_generator_b), self.data.threas)
            loss, ratio = self.calculate_loss(cryo_net, descs)
            stats.update_stats(ratio=ratio)
            stats.print()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if stats.advance():
                stats.reset()
                print("Saving")
                pickle.dump(cryo_net, open(self.data.model_path, 'wb'))

    def calculate_loss(self, cryo_net, descs):
        patches_original, patches_fake = descs
        patches_original = patches_original.unsqueeze(dim=1)
        patches_fake = patches_fake.unsqueeze(dim=1)
        num_points = patches_fake.shape[0]
        descs_original = cryo_net(patches_original)
        descs_fake = cryo_net(patches_fake)
        dists = torch.cdist(descs_fake, descs_original)
        similiarity_mask = np.eye(num_points)
        difference_mask = np.triu(np.ones(num_points), 1)
        similiarity_norm = torch.sum(dists[np.where(similiarity_mask)]) / np.sum(similiarity_mask)
        difference_norm = torch.sum(dists[np.where(difference_mask)]) / np.sum(difference_mask)

        loss = similiarity_norm - self.data.alpha * difference_norm

        ratio = similiarity_norm.item() / difference_norm.item()

        return loss, ratio
