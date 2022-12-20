import os
import numpy as np
DIR_PATH = "/cs/labs/dina/meitar/rhodopsins/cutted_parts/"

def get_b_factor_average(lines):
    tokens = [float(line.split()[-1]) for line in lines if line.split()[0] == 'ATOM']
    return np.mean(tokens)

def main():
    for dirpath, _, files in os.walk(DIR_PATH):
        for protein in files:
            with open(dirpath + protein, "r") as f:
                lines = f.readlines()
                avg = get_b_factor_average(lines)
                print(avg)

if __name__ == "__main__":
    main()
