import MapSeq
import time
import models
import torch.optim as optim
import torch
import pickle

if __name__ == "__main__":

    PDB_PATH = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/pdb7qti.ent"
    CRYO_FILE = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/emd_26597.map"

    my_seq = MapSeq.MapSeq(CRYO_FILE, PDB_PATH)

    my_cryo_net = models.CryoNet().float()
    my_protein_net = models.ProteinNet(60, 30).float()
    my_cryo_net.zero_grad()
    my_protein_net.zero_grad()
    optimizer = optim.Adam(list(my_protein_net.parameters()) + list(my_cryo_net.parameters()), lr=0.001)

    i = 0
    while True:
        point1 = my_seq.get_interesting_point()
        map_desc1, atoms_desc1 = my_seq.generate_descriptors(point1, 5)
        point2 = my_seq.get_interesting_point()
        map_desc2, atoms_desc2 = my_seq.generate_descriptors(point2, 5)

        cryo_descriptor1 = my_cryo_net(torch.tensor(map_desc1).unsqueeze(dim=0).float())
        cryo_descriptor2 = my_cryo_net(torch.tensor(map_desc2).unsqueeze(dim=0).float())
        protein_descriptor1 = my_protein_net(torch.tensor(atoms_desc1).float())
        protein_descriptor2 = my_protein_net(torch.tensor(atoms_desc2).float())
        similiarity_norm = torch.norm(cryo_descriptor1 - protein_descriptor2) + torch.norm(cryo_descriptor2 - protein_descriptor2)
        difference_norm = torch.norm(cryo_descriptor1 - cryo_descriptor2) + torch.norm(protein_descriptor1 - protein_descriptor2)
        alpha = 1.5
        loss = similiarity_norm - alpha * difference_norm
        if loss == loss:
            print(similiarity_norm.item())
            print(difference_norm.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Saving")
            pickle.dump(my_cryo_net, open('cryo_model.pckl', 'wb'))
            pickle.dump(my_protein_net, open('protein_model.pckl', 'wb'))

        i += 1
