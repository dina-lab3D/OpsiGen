import MapSeq
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


def is_empty_point(arr, threashold_for_density = 500):
    volume = 1
    for index in range(len(arr.shape)):
        volume *= arr.shape[index]
    # print("Above the mean values", np.sum(np.where(arr.detach().numpy() > np.mean(arr.detach().numpy()))))
    if np.sum(np.where(arr.detach().numpy() > np.mean(arr.detach().numpy()))) < volume / 8:
        return True
    return False


def calculate_descriptors(mapseq1, mapseq2, threas):
    desc1a, desc1b, desc2a, desc2b = None, None, None, None
    is_valid_points = False
    is_close_points = False

    mat1 = R.from_euler("xyz", [45, 45, 45])

    while (not is_valid_points) or is_close_points:
        point1 = mapseq1.get_interesting_point()
        point2 = mapseq2.get_interesting_point()
        if np.linalg.norm((np.array(point1) - np.array(point2))) < threas / 2:
            is_close_points = True
            continue
        else:
            is_close_points = False
        desc1a, _ = mapseq1.generate_descriptors(point1, threas)
        desc1b, _ = mapseq2.generate_descriptors(point1, threas) #, to_rotate=mat1)
        desc2a, _ = mapseq1.generate_descriptors(point2, threas)
        desc2b, _ = mapseq2.generate_descriptors(point2, threas) #, to_rotate=mat1)

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

    if ratio > 4:
        mrcfile.write(CRYO_FILE_TEMPLATE_A, desc1a.detach().numpy(), overwrite=True)
        mrcfile.write(CRYO_FILE_TEMPLATE_B, desc1b.detach().numpy(), overwrite=True)
        mrcfile.write(CRYO_FILE_TEMPLATE_C, desc2a.detach().numpy(), overwrite=True)
        mrcfile.write(CRYO_FILE_TEMPLATE_D, desc2b.detach().numpy(), overwrite=True)
        print("Bad points")
        breakpoint()
    #
    # if ratio < 0.25:
    #     mrcfile.write(CRYO_FILE_TEMPLATE_A, desc1a.detach().numpy(), overwrite=True)
    #     mrcfile.write(CRYO_FILE_TEMPLATE_B, desc1b.detach().numpy(), overwrite=True)
    #     mrcfile.write(CRYO_FILE_TEMPLATE_C, desc2a.detach().numpy(), overwrite=True)
    #     mrcfile.write(CRYO_FILE_TEMPLATE_D, desc2b.detach().numpy(), overwrite=True)
    #     print("Good points")
    #     breakpoint()
    return loss, similiarity_norm.item() / difference_norm.item()


def train(alpha, threas, to_load, lr):

    mapseq1 = MapSeq.MapSeq(CRYO_FILE1, PDB_PATH)
    mapseq2 = MapSeq.MapSeq(CRYO_FILE2, PDB_PATH)


    if to_load:
        cryo_net = pickle.load(open("cryo_model.pckl", 'rb'))
    else:
        cryo_net = models.CryoNet(threas).float()
    cryo_net.zero_grad()
    optimizer = optim.Adam(list(cryo_net.parameters()), lr=lr)

    correct = 1
    wrong = 1
    super_correct = 1
    super_wrong = 1
    i = 0
    while True:
        # alpha = 1.5
        map_desc1a, map_desc1b, map_desc2a, map_desc2b = calculate_descriptors(mapseq1, mapseq2, threas)
        loss, ratio = calculate_loss(cryo_net, map_desc1a, map_desc1b, map_desc2a, map_desc2b, alpha=alpha)
        if ratio < 1:
            correct += 1
        else:
            wrong += 1

        if ratio > 4:
            super_wrong += 1
        if ratio < 0.25:
            super_correct += 1
        print("{:.2f}% success, {:.2f}% super success".format(float(correct) * 100/ (wrong + correct),
                                                              float(super_correct) * 100/ (super_wrong + super_correct)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print("Saving")
            pickle.dump(cryo_net, open('cryo_model.pckl', 'wb'))

            i = 0
            correct = 1
            wrong = 1
            super_correct = 1
            super_wrong = 1
        i += 1
