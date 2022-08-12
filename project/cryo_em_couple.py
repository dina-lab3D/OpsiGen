import Bio.PDB
import mrcfile
import numpy as np
import random
import graph_builders

PDB_PATH = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/pdb7qti.ent"
CRYO_FILE = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/emd_26597.map"


def get_relevant_area(point, pdb_data, cryo_data):
    THREASHOLD = 10

    area_atoms = []
    for atom in pdb_data.get_atoms():
        if np.linalg.norm((atom.coord - point), np.inf) < THREASHOLD:
            area_atoms.append(atom)

    area_cryo = cryo_data[point[0] - THREASHOLD: point[0] + THREASHOLD,
                point[1] - THREASHOLD: point[1] + THREASHOLD,
                point[2] - THREASHOLD: point[2] + THREASHOLD]

    print(area_atoms)
    return area_atoms, area_cryo


def main():
    parser = Bio.PDB.PDBParser(PERMISSIVE=True, QUIET=True)
    pdb_data = parser.get_structure("7qti", PDB_PATH)
    with mrcfile.open(CRYO_FILE) as protein:
        cryo_data = protein.data

    for i in range(1):
        # point = (random.randint(1, 256), random.randint(1, 256), random.randint(1, 256))
        point = (253, 164, 247)
        area_atoms, area_cryo = get_relevant_area(point, pdb_data, cryo_data)
        if len(area_atoms) > 15:
            graph_builders.compute_distance(area_atoms)


if __name__ == "__main__":
    main()
