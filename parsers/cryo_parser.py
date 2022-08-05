import mrcfile
import torch
from parsers import data_parser
from training import models

FILE_NAME = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/emd_26597.map"


def cryo_to_descriptor(model, cryo_map):
    map_tensor = torch.tensor(cryo_map)
    map_tensor = torch.unsqueeze(map_tensor, 0)
    result = model.forward(map_tensor)
    print(result.shape)


def main():
    with mrcfile.open(FILE_NAME) as protein:
        data = protein.data

    cryo_to_descriptor(models.CryoNet(), data)


if __name__ == "__main__":
    main()
