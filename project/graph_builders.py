import numpy as np


def compute_distance(atoms):
    N = len(atoms)
    result = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            result[i][j] = np.linalg.norm(atoms[i].get_vector() - atoms[j].get_vector())

    breakpoint()
    print(atoms[0])
