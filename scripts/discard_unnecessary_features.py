import os
import numpy as np

WAVELENGTH_FILE = "/cs/labs/dina/meitar/rhodopsins/excel/wavelength.dat"
FEATURES_DIR = "/cs/labs/dina/meitar/ionet-meitar/Interface_grid/new_features/ca_24/"
# RESULT_FEATURE_DIR = "/cs/labs/dina/meitar/ionet-meitar/Interface_grid/new_features/ca_24_clean2/"


def read_wavelength_file():
    with open(WAVELENGTH_FILE, "r") as f:
        lines = [int(line.strip()) for line in f.readlines() if line != 'NA\n']

    return np.array(lines)

def read_all_features():
    features = []
    for dirpath, _, files in os.walk(FEATURES_DIR):
        for file_name in files:
            if file_name.endswith('.npz'):
                arr = np.load(os.path.join(dirpath,file_name))
                print(arr.shape)
                features.append(arr)

    print(len(features))
    print(np.concatenate(features))

    return np.concatenate(features)

def save_array_in_dir(arr, dirpath):
    for i in range(arr.shape[0]):
        with open(os.path.join(dirpath, "cutted_parts{}.npz".format(i)), "wb") as f:
            np.save(f, arr[i])


def main():
    y = read_wavelength_file()
    x = read_all_features()
    breakpoint()
    #med = np.median(np.std(x, 0))
    arr = np.where((np.std(x, 0) != 0))
    # x = x[:,arr].squeeze()
    
    # save_array_in_dir(x, RESULT_FEATURE_DIR)

if __name__ == "__main__":
    main()
