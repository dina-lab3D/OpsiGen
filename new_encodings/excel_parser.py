import pandas as pd
import os
import numpy as np
from Bio import PDB
from scipy.spatial import distance


EXCEL_PATH = "/cs/labs/dina/meitar/rhodopsins/excel/data.xlsx"
RESULT_PATH = "/cs/labs/dina/meitar/rhodopsins/new_encodings/results/"

def letter_to_onehot(letter):
    result = np.zeros(26)
    index = ord(letter.lower()) - ord('a')
    if index in range(26):
        result[index] = 1

    return result


def entry_to_encoding(entry, index, path):
    vectors = []
    for i in range(5, 29):
        vectors.append(letter_to_onehot(entry.iloc[i]))

    print(np.concatenate(vectors, axis=0).shape)
    return np.reshape(np.concatenate(vectors), (24, 26))


def generate_features_files(df, path):
    for i in range(len(df)):
        arr = entry_to_encoding(df.iloc[i], i, path)
        np.save("{}{}-[{}]".format(RESULT_PATH, i, df.iloc[i].iloc[3]), arr)


def main():
    # generate_distance_matrices_from_folder("/cs/labs/dina/meitar/rhodopsins/pdbs/",
                                           # "/cs/labs/dina/meitar/rhodopsins/graphs/")
    df = pd.read_excel(EXCEL_PATH)
    generate_features_files(df, RESULT_PATH)

if __name__ == "__main__":
    main()
