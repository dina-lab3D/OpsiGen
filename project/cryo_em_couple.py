import Bio.PDB
import mrcfile
import numpy as np
import random
import graph_builders
from models import CryoNet, ProteinNet
import torch
import torch.optim as optim
import pickle
import time

PDB_PATH = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/pdb7qti.ent"
CRYO_FILE = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/emd_26597.map"


def get_relevant_area(point, pdb_data, cryo_data):
    THREASHOLD = 10

    area_atoms = []
    area_cryo = None

    if pdb_data:
        for atom in pdb_data.get_atoms():
            if np.linalg.norm((atom.coord - point), np.inf) < THREASHOLD:
                area_atoms.append(atom)

    if not cryo_data is None:
        area_cryo = cryo_data[point[0] - THREASHOLD: point[0] + THREASHOLD,
                    point[1] - THREASHOLD: point[1] + THREASHOLD,
                    point[2] - THREASHOLD: point[2] + THREASHOLD]

    return area_atoms, area_cryo


def create_descriptor_from_atoms(atoms, funcs_dict):
    res = np.array([])
    distmat = graph_builders.compute_distance(atoms)
    for func in funcs_dict:
        res = np.concatenate([res, func(atoms, funcs_dict[func], distmat)])

    return res


def main():
    parser = Bio.PDB.PDBParser(PERMISSIVE=True, QUIET=True)
    pdb_data = parser.get_structure("7qti", PDB_PATH)
    with mrcfile.open(CRYO_FILE) as protein:
        cryo_data = protein.data

    funcs = {
        graph_builders.compute_hydrophobic_feature: 10,
        graph_builders.compute_ionic_feature: 15,
        graph_builders.compute_hydrogen_feature: 15,
        graph_builders.compute_cation_pi_feature: 10,
        graph_builders.compute_disulfide_feature: 10
    }

    # my_cryo_net = CryoNet().float()
    # my_protein_net = ProteinNet(60, 30).float()
    my_cryo_net = pickle.load(open("cryo_model.pckl", 'rb'))
    my_protein_net = pickle.load(open("protein_model.pckl", 'rb'))
    my_cryo_net.zero_grad()
    my_protein_net.zero_grad()
    optimizer = optim.Adam(list(my_protein_net.parameters()) + list(my_cryo_net.parameters()), lr=0.001)

    i = 0

    good_points = np.argwhere((0.0001 < cryo_data) & (0.001 > cryo_data))

    print(len(good_points))

    while True:
        point = random.choice(good_points)
        area_atoms, area_cryo = get_relevant_area(point, pdb_data, cryo_data)
        _, area_cryo_bad = get_relevant_area((random.randint(11, 246), random.randint(11, 246),
                                                           random.randint(11, 246)), None, cryo_data)

        borders = False
        for i in range(3):
            if abs(point[i]) < 10 or abs(256 - point[i]) < 10:
                borders = True
        if borders:
            continue

        if len(area_atoms) < 20:
            print(len(area_atoms))
            print("Bad luck")
            continue
        else:
            print(point)
            print(cryo_data[point[0], point[1], point[2]])

        res = create_descriptor_from_atoms(area_atoms, funcs)
        cryo_descriptor = my_cryo_net(torch.tensor(area_cryo).unsqueeze(dim=0).float())
        protein_descriptor = my_protein_net(torch.tensor(res).float())
        cryo_descriptor_bad = my_cryo_net(torch.tensor(area_cryo_bad).unsqueeze(dim=0).float())
        my_loss = torch.norm(cryo_descriptor - cryo_descriptor_bad)
        loss = torch.norm(protein_descriptor - cryo_descriptor) - 100000 * my_loss
        if loss == loss:
            print(loss.item(), my_loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        i += 1

        if i % 100 == 0:
            print("Saving")
            pickle.dump(my_cryo_net, open('cryo_model.pckl', 'wb'))
            pickle.dump(my_protein_net, open('protein_model.pckl', 'wb'))


if __name__ == "__main__":
    main()
