import os
from Bio import PDB
from Bio.PDB import PDBIO

CUTTED_PARTS_PATH = "/cs/labs/dina/meitar/rhodopsins/their_cutted_parts/"
WL_FILE = "/cs/labs/dina/meitar/rhodopsins/excel/wavelength.dat"
SEQUENCES = "/cs/labs/dina/meitar/rhodopsins/excel/sequences.fas"

AMINO_DICT = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLU": "E",
        "GLN": "Q",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V"
}


class Sequences:

    POSITIONS = [
        429,
        465,
        469,
        597,
        599,
        600,
        603,
        604,
        607,
        651,
        652,
        655,
        687,
        690,
        691,
        694,
        774,
        777,
        778,
        781,
        828,
        832,
        835,
        836,
    ]

    def __init__(self, seq_path):
        with open(seq_path, "r") as f:
            self.lines = f.readlines()

    def _get_relevant_line(self, protein_index):
        return self.lines[2 * protein_index + 1].strip()

    def get_item(self, protein_index, seq_index):
        line = self._get_relevant_line(protein_index)
        amino_index = -1
        seq_running_index = 0
        while seq_running_index < seq_index:

            if line[seq_running_index] != '-':
                amino_index += 1
            
            seq_running_index += 1

        max_length = len(line.replace('-',''))
        
        if amino_index >= max_length:
            print("Bad sequence!!")
            exit()

        return amino_index 

    def get_interesting_positions(self, protein_index):
        res = [self.get_item(protein_index, p) for p in self.POSITIONS]
        for i in range(len(res) - 1):
            if res[i] == res[i+1]:
                res[i+1] += 1
        line = self._get_relevant_line(protein_index).replace('-','')
        print([line[i] for i in res])

        # line = self.lines[2 * protein_index + 1].strip()
        # arr = [line.strip().replace('-','')[i] for i in res]

        return res

def get_wavelength_indexes_from_file(wl_file):
    with open(wl_file, "r") as f:
        wl_lines = f.readlines()

    indexes = []
    for i in range(len(wl_lines)):
        if wl_lines[i] != 'NA\n':
            indexes.append(i)


    return {i: indexes[i] for i in range(len(indexes))}

def get_index_from_file_name(filename):
    tokens = filename.split('_')
    return int(tokens[-6])

def compare_amino_to_res(index, res_list, line):
    for i, res in enumerate(res_list):
        if AMINO_DICT[res.get_resname()] != line[i]:
            print("index is bad: ", index)
            return


def cut_pdb(parser, pdb_path, sequences, io):
    filename = pdb_path.split('/')[-1]
    rhodopsin_index = get_index_from_file_name(filename)
    index_mapping = get_wavelength_indexes_from_file(WL_FILE)
    # print(rhodopsin_index, index_mapping[rhodopsin_index])
    amino_indexes = sequences.get_interesting_positions(index_mapping[rhodopsin_index])
    print(amino_indexes)
    line = sequences._get_relevant_line(rhodopsin_index).replace('-','')
    struct = parser.get_structure(pdb_path, pdb_path)
    model = struct.child_list[0]
    chain = model.child_list[0]
    length = len(chain.child_list)
    for residue in range(1, length+1):
        if residue not in amino_indexes:
            chain.detach_child((" ", residue, " "))

    if (len(chain.child_list)) != 24:
        print("Length before", length)
        print("Length of chain.child_list", len(chain.child_list))
        print("Amino indexes", amino_indexes)
        print("Line length", len(line))
        print("Filename", filename)
        print(rhodopsin_index, index_mapping[rhodopsin_index])
        # breakpoint()

        return True


    model.child_list[0] = chain
    struct.child_list[0] = model

    # io.set_structure(struct)
    # io.save(os.path.join(CUTTED_PARTS_PATH, "cutted_parts{}.pdb".format(rhodopsin_index)))

    return False

def cut_pdbs(input_dir):
    parser = PDB.PDBParser()
    io = PDBIO()
    s = Sequences(SEQUENCES)
    counter = 0
    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".pdb"):
                res = cut_pdb(parser, dirpath + "/" + filename, s, io)
                if res:
                    counter += 1

def print_interesting_amino_acids():
    print("Hello")
    s = Sequences("/cs/labs/dina/meitar/rhodopsins/scripts/sequences.fas")
    for i in range(884):
        print(s.get_interesting_positions(i))

def print_excel_lines():
    s = Sequences("/cs/labs/dina/meitar/rhodopsins/scripts/sequences.fas")
    for i in range(884):
        print(s.lines[i * 2])
    

def main():
    # print_excel_lines()
    cut_pdbs("/cs/labs/dina/meitar/rhodopsins/pdbs")
    # print_interesting_amino_acids()

if __name__ == "__main__":
    main()
