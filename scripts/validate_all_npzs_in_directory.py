import os
import numpy as np

INPUT_DIR = "/cs/labs/dina/meitar/rhodopsins/features/"

def main():
    bad_files = 0
    good_files = 0
    for (dirpath, dirnames, filenames) in os.walk(INPUT_DIR):
        for filename in filenames:
            if filename.endswith('.npz'): 
                try:
                    arr = np.load(os.path.join(dirpath, filename))
                    nan_arr = np.isnan(arr)
                    if nan_arr.any():
                        print(filename)
                        print(np.argwhere(nan_arr), arr.shape)
                        bad_files += 1
                    else:
                        good_files += 1
                except (ValueError, FileNotFoundError) as e:
                    print(e)
                    print(filename, "is bad")

            print("bad files: ", bad_files, "good files:", good_files)

if __name__ == "__main__":
    main()
