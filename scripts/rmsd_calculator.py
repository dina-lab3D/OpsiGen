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

replacements = {
    '/': '-',
    ' ': '',
    '(': 'O',
    ')': 'O',
    '+': '_',
    ',': '_',
    '\xa0': '_',
}


def entry_to_features_file_names(entry, id, path):
    if id == 175:
        print(entry['Name'])

    replaced_entry = entry['Name']
    if ')' in entry['Name'] and (not '(' in entry['Name']):
        replacements[')'] = '_'

    for key in replacements:
        # print(key, replacements[key])
        replaced_entry = replaced_entry.replace(key, replacements[key])

    features_file_names_format = replaced_entry + '_' + entry['Wildtype'].replace(' ','') + '_' + str(
        id) + '_' + 'unrelaxed_rank_{}_model_{}.pdb'
    result = []
    for i in range(1,6):
        for j in range(1,6):
            result.append(path + features_file_names_format.format(i, j).replace('\xa0',''))
    return result


def generate_fasta_name(entry, id, path):
    res = (str(id) + '-' + entry['Name'].replace('/','.').replace(' ', '_') + '.fasta').replace('\xa0','')
    return res


def generate_pdb_name(entry, id, path):
    files = []
    for root, folds, file_names in os.walk(path):
        files = file_names

    for file_name in files:
        if file_name.startswith(str(id) + "-"):
            return file_name.replace('(', '\(').replace(')', '\)')


    # return generate_fasta_name(entry, id, path).replace('.fasta', '.pdb').replace('(', '\(').replace(')', '\)')


def generate_pdb_files(df, path):
    print("starting to calculate rmsd")
    for i in range(len(df)):
        experimental_pdb = EXPERIMENTAL_PDBS_PATH + generate_pdb_name(df.iloc[i], i, EXPERIMENTAL_PDBS_PATH)
        af_pdbs = entry_to_features_file_names(df.iloc[i], i, OUR_PDBS_PATH)
        if i != 224:
            continue

        print(experimental_pdb)
        print(af_pdbs)
        
        for j, af_pdb in enumerate(af_pdbs):
            if os.path.isfile(af_pdb):
                print('/cs/labs/dina/meitar/rhodopsins/scripts/my_align.pl {} {} {}/match_{}[{}].stats'.format(experimental_pdb, af_pdb, OUTPUT_FOLDER, i, int(j / 5) + 1))

                os.system('/cs/labs/dina/meitar/rhodopsins/scripts/my_align.pl {} {} {}/match_{}[{}].stats'.format(experimental_pdb, af_pdb, OUTPUT_FOLDER, i, int(j / 5) + 1))

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
