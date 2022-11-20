import os
import numpy as np

INPUT_DIR = "/cs/labs/dina/meitar/rhodopsins/features/"
FEATURE_SIZE=18

def main():
    bad_files = 0
    good_files = 0
    for (dirpath, dirnames, filenames) in os.walk(INPUT_DIR):
        for filename in filenames:
            if filename.endswith('.npz'): 
                try:
                    arr = np.load(os.path.join(dirpath, filename))
                    print(arr.shape, "before")
                    if len(arr.shape) == 1:
                        arr = arr.reshape(int(arr.shape[0] / FEATURE_SIZE), FEATURE_SIZE)
                        np.save(os.path.join(dirpath, filename.replace('.npz', '')), arr)
                        print(arr.shape, "after")
                        bad_files += 1
                    else:
                        good_files += 1
                except (ValueError, FileNotFoundError) as e:
                    print(e)
                    print(filename, "is bad")

            print("fixed files: ", bad_files, "good files:", good_files)

if __name__ == "__main__":
    main()
