import pandas as pd
import os
import numpy as np
from Bio import PDB
from scipy.spatial import distance
import time


EXCEL_PATH = "/cs/labs/dina/meitar/rhodopsins/excel/data.xlsx"
OUR_PDBS_PATH = "/cs/labs/dina/meitar/rhodopsins/pdbs/"
EXPERIMENTAL_PDBS_PATH = "/cs/labs/dina/meitar/rhodopsins/chains/"
OUTPUT_FOLDER = "/cs/labs/dina/meitar/rhodopsins/matches/"


def entry_to_features_file_names(entry, id, path):
    features_file_names_format = entry['Name'].replace('/', '-').replace(' ', '') + '_' + entry['Wildtype'] + '_' + str(
        id) + '_' + 'unrelaxed_rank_{}_model_{}.pdb'
    result = []
    for i in range(1,6):
        for j in range(1,6):
            result.append(path + features_file_names_format.format(i, j))
    return result


def generate_fasta_name(entry, id, path):
    return str(id) + '-' + entry['Name'].replace('/','.').replace(' ', '_') + '.fasta'


def generate_pdb_name(entry, id, path):
    return generate_fasta_name(entry, id, path).replace('.fasta', '.pdb')


def generate_pdb_files(df, path):
    print("starting to calculate rmsd")
    for i in range(len(df)):
        experimental_pdb = EXPERIMENTAL_PDBS_PATH + generate_pdb_name(df.iloc[i], i, EXPERIMENTAL_PDBS_PATH)
        af_pdbs = entry_to_features_file_names(df.iloc[i], i, OUR_PDBS_PATH)

        for j, af_pdb in enumerate(af_pdbs):
            if os.path.isfile(af_pdb):
                os.system('/cs/labs/dina/meitar/rhodopsins/scripts/my_align.pl {} {} {}/match_{}[{}].stats'.format(experimental_pdb, af_pdb, OUTPUT_FOLDER, i, j))

    print("Done")


def main():
    """
    generate_distance_matrices_from_folder("/cs/labs/dina/meitar/rhodopsins/pdbs/",
                                           "/cs/labs/dina/meitar/rhodopsins/graphs/")
    """
    df = pd.read_excel(EXCEL_PATH)
    generate_pdb_files(df, EXPERIMENTAL_PDBS_PATH)

if __name__ == "__main__":
    main()
