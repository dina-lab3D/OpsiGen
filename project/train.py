import CubeGenerator
import time
import models
import torch.optim as optim
import torch
import pickle
import numpy as np
import mrcfile
from scipy.spatial.transform import Rotation as R

PDB_PATH = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/pdb7qti.ent"
CRYO_FILE1 = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/emd_14141.map"
CRYO_FILE2 = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/7qti.mrc"
CRYO_FILE_TEMPLATE_A = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/a{}.mrc"
CRYO_FILE_TEMPLATE_B = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/b{}.mrc"
CRYO_FILE_TEMPLATE_C = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/c{}.mrc"
CRYO_FILE_TEMPLATE_D = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/d{}.mrc"


class PerformanceStats:

    def __init__(self, reset_steps=1000):
        self.correct = 1
        self.wrong = 1
        self.super_correct = 1
        self.super_wrong = 1
        self.step = 0
        self.reset_steps = reset_steps

    def update_stats(self, ratio):
        if ratio < 1:
            self.correct += 1
        else:
            self.wrong += 1

        if ratio > 4:
            self.super_wrong += 1
        if ratio < 0.25:
            self.super_correct += 1

    def advance(self):
        self.step += 1
        return self.step % self.reset_steps == 0

    def print(self):
        print("{:.2f}% success, {:.2f}% super success".format(float(self.correct) * 100 / (self.wrong + self.correct),
                                                              float(self.super_correct) * 100 / (self.super_wrong + self.super_correct)))


def is_empty_point(arr, threashold_for_density = 500):
    volume = 1
    for index in range(len(arr.shape)):
        volume *= arr.shape[index]
    # print("Above the mean values", np.sum(np.where(arr.detach().numpy() > np.mean(arr.detach().numpy()))))
    if np.sum(np.where(arr.detach().numpy() > np.mean(arr.detach().numpy()))) < volume / 8:
        return True
    return False


def calculate_descriptors(cube_generator1, cube_generator2, threas):
    desc1a, desc1b, desc2a, desc2b = None, None, None, None
    is_valid_points = False
    is_close_points = False

    mat1 = R.random()

    while (not is_valid_points) or is_close_points:
        point1 = cube_generator1.get_interesting_point()
        point2 = cube_generator2.get_interesting_point()
        if np.linalg.norm((np.array(point1) - np.array(point2))) < threas / 2:
            is_close_points = True
            continue
        else:
            is_close_points = False
        desc1a, _ = cube_generator1.generate_descriptors(point1, threas)
        desc1b, _ = cube_generator2.generate_descriptors(point1, threas, to_rotate=mat1)
        desc2a, _ = cube_generator1.generate_descriptors(point2, threas)
        desc2b, _ = cube_generator2.generate_descriptors(point2, threas, to_rotate=mat1)

        is_valid_points = not (is_empty_point(desc1b) or is_empty_point(desc2b) or is_empty_point(desc1a)
                               or is_empty_point(desc2a))

    return desc1a, desc1b, desc2a, desc2b


def calculate_loss(cryo_net, desc1a, desc1b, desc2a, desc2b, alpha=1.0):
    similiarity_norm = torch.norm(cryo_net(desc1a) - cryo_net(desc1b)) + \
                       torch.norm(cryo_net(desc2a) - cryo_net(desc2b))
    difference_norm = torch.norm(cryo_net(desc1a) - cryo_net(desc2a)) + \
                      torch.norm(cryo_net(desc1b) - cryo_net(desc2b))
    loss = similiarity_norm - alpha * difference_norm

    print("similiearity", similiarity_norm.item())
    print("difference--", difference_norm.item())
    print("alpha is ", alpha)

    ratio = similiarity_norm.item() / difference_norm.item()

    # if ratio > 4:
    #     mrcfile.write(CRYO_FILE_TEMPLATE_A, desc1a.detach().numpy(), overwrite=True)
    #     mrcfile.write(CRYO_FILE_TEMPLATE_B, desc1b.detach().numpy(), overwrite=True)
    #     mrcfile.write(CRYO_FILE_TEMPLATE_C, desc2a.detach().numpy(), overwrite=True)
    #     mrcfile.write(CRYO_FILE_TEMPLATE_D, desc2b.detach().numpy(), overwrite=True)
    #     print("Bad points")
    #     breakpoint()
    #
    # if ratio < 0.25:
    #     mrcfile.write(CRYO_FILE_TEMPLATE_A, desc1a.detach().numpy(), overwrite=True)
    #     mrcfile.write(CRYO_FILE_TEMPLATE_B, desc1b.detach().numpy(), overwrite=True)
    #     mrcfile.write(CRYO_FILE_TEMPLATE_C, desc2a.detach().numpy(), overwrite=True)
    #     mrcfile.write(CRYO_FILE_TEMPLATE_D, desc2b.detach().numpy(), overwrite=True)
    #     print("Good points")
    #     breakpoint()
    return loss, similiarity_norm.item() / difference_norm.item()


def init_model(to_load, threas, path, lr):
    if to_load:
        cryo_net = pickle.load(open(path, 'rb'))
    else:
        cryo_net = models.CryoNet(threas).float()
    cryo_net.zero_grad()
    optimizer = optim.Adam(list(cryo_net.parameters()), lr=lr)

    return cryo_net, optimizer


def train(alpha, threas, to_load, lr):
    cube_generator1 = CubeGenerator.CubeGenerator(CRYO_FILE1)
    cube_generator2 = CubeGenerator.CubeGenerator(CRYO_FILE2)
    cryo_net, optimizer = init_model(to_load, threas, "cryo_model.pckl", lr)

    stats = PerformanceStats()

    while True:
        map_desc1a, map_desc1b, map_desc2a, map_desc2b = calculate_descriptors(cube_generator1, cube_generator2, threas)
        loss, ratio = calculate_loss(cryo_net, map_desc1a, map_desc1b, map_desc2a, map_desc2b, alpha=alpha)
        stats.update_stats(ratio=ratio)
        stats.print()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if stats.advance():
            print("Saving")
            pickle.dump(cryo_net, open('cryo_model.pckl', 'wb'))