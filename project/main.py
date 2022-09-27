import MapSeq
import time
import models
import torch.optim as optim
import torch
import pickle
import numpy as np
import mrcfile


def check_bad_point(arr, threashold_for_density = 500):
    volume = 1
    for index in range(len(arr.shape)):
        volume *= arr.shape[index]
    # print("Above the mean values", np.sum(np.where(arr.detach().numpy() > np.mean(arr.detach().numpy()))))
    if np.sum(np.where(arr.detach().numpy() > np.mean(arr.detach().numpy()))) < volume / 4:
        return True
    return False


if __name__ == "__main__":

    PDB_PATH = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/pdb7qti.ent"
    CRYO_FILE1 = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/emd_14141.map"
    CRYO_FILE2 = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/7qti.mrc"
    CRYO_FILE_TEMPLATE_A = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/a{}.mrc"
    CRYO_FILE_TEMPLATE_B = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/b{}.mrc"

    my_seq1 = MapSeq.MapSeq(CRYO_FILE1, PDB_PATH)
    my_seq2 = MapSeq.MapSeq(CRYO_FILE2, PDB_PATH)

    to_load = True
    # to_load = False
    threas = 15

    if to_load:
        my_cryo_net = pickle.load(open("cryo_model.pckl", 'rb'))
        # my_protein_net = pickle.load(open("protein_model.pckl", 'rb'))
    else:
        my_cryo_net = models.CryoNet(threas).float()
        # my_protein_net = models.ProteinNet(100, 30).float()
    my_cryo_net.zero_grad()
    # my_protein_net.zero_grad()
    # optimizer = optim.Adam(list(my_protein_net.parameters()) + list(my_cryo_net.parameters()), lr=0.001)
    optimizer = optim.Adam(list(my_cryo_net.parameters()), lr=0.001)

    correct = 0
    wrong = 0
    i = 0
    while True:
        point1 = my_seq1.get_interesting_point()
        map_desc1a, _ = my_seq1.generate_descriptors(point1, threas)
        map_desc1b, _ = my_seq2.generate_descriptors(point1, threas, to_center=True)
        point2 = my_seq2.get_interesting_point()
        map_desc2a, _ = my_seq1.generate_descriptors(point2, threas)
        map_desc2b, _ = my_seq2.generate_descriptors(point2, threas, to_center=True)

        if check_bad_point(map_desc1b) or check_bad_point(map_desc2b) or check_bad_point(map_desc1a) or check_bad_point(map_desc2a):
            # print("Bad point!")
            continue




        similiarity_norm = torch.norm(my_cryo_net(map_desc1a) - my_cryo_net(map_desc1b)) +\
                           torch.norm(my_cryo_net(map_desc2a) - my_cryo_net(map_desc2b))
        difference_norm = torch.norm(my_cryo_net(map_desc1a) - my_cryo_net(map_desc2a)) + \
                          torch.norm(my_cryo_net(map_desc1b) - my_cryo_net(map_desc2b))
        alpha = 1
        loss = similiarity_norm - alpha * difference_norm
        if loss == loss:
            print("similiearity", similiarity_norm.item())
            print("difference--", difference_norm.item())
            print("alpha is ", alpha, i)
            ratio = similiarity_norm.item() / difference_norm.item()
            print("{} vs {} - ratio {}".format(correct, wrong, ratio))
            # if ratio > 1.5:
            #     mrcfile.write(CRYO_FILE_TEMPLATE_A, map_desc1a.detach().numpy(), overwrite=True)
            #     mrcfile.write(CRYO_FILE_TEMPLATE_B, map_desc1b.detach().numpy(), overwrite=True)
            #     breakpoint()

            if similiarity_norm.item() < difference_norm.item():
                correct += 1
            else:
                wrong += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print("Saving")
            pickle.dump(my_cryo_net, open('cryo_model.pckl', 'wb'))

            i = 0
            correct = 0
            wrong = 0

        i += 1