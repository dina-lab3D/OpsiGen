import os
import numpy as np

DIRPATH = "/cs/labs/dina/meitar/rhodopsins/their_features/"
RESPATH = "/cs/labs/dina/meitar/rhodopsins/their_features_normalized/"

def normalize_features(dir_name):
    sizes = []
    arrays = []
    features = []

    for i in range(884):
        arr = np.load(DIRPATH + "cutted_parts{}.npz".format(i))
        sizes.append(arr.shape[0])
        arrays.append(arr)

    concatenated = np.concatenate(arrays)
    columns_of_concatenated = np.max(np.abs(concatenated), 0)
    relevant_args = np.argwhere(columns_of_concatenated != 0).squeeze()
    normalized_concatenated = concatenated[:, relevant_args] / columns_of_concatenated[relevant_args]


    counter_index = 0
    current_sum_of_sizes = 0
    for i in range(884):
        feature = []
        for j in range(sizes[i]):
            print(j)
            feature.append(normalized_concatenated[current_sum_of_sizes + j])

        current_sum_of_sizes += sizes[i]
        arr = np.vstack(feature)
        with open(RESPATH + "cutted_parts{}.npz".format(i), "wb") as f:
            np.save(f, arr)
        

def main():
    normalize_features(DIRPATH)
    pass

if __name__ == "__main__":
    main()
