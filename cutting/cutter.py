from Bio import PDB
from Bio.PDB import PDBIO
import csv
from Bio.SeqUtils import seq1
INDICES_PATH = "./indices.csv"
POSITIONS_PATH = "./predicted_positions.csv"

class Cutter:

    def __init__(self, config):
        self.pdb_path = config["pdb_path"]
        self.parser = PDB.PDBParser()
        self.io = PDBIO()
        self.result_path = config["cutted_result_pdb_path"]
        self.failures_path = "../failures.csv"

    def record_index(self, index):
        with open(INDICES_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.pdb_path, index])

    def cut_pdb(self, aligner_obj):
        positions = aligner_obj.get_amino_acids_of_aligned_sequence()
        # self.record_index(aligner_obj.best_index)
        print("Cutting the relevant amino acids, and outputing them to a result pdb")

        struct = self.parser.get_structure(self.pdb_path, self.pdb_path)
        model = struct.child_list[0]
        chain = model.child_list[0]
        length = len(chain.child_list)
        chosen = ""
        for residue in range(1, length+1):
            if residue not in positions:
                chain.detach_child((" ", residue, " "))
            else:
                resname = chain[residue].get_resname()
                chosen = chosen + seq1(resname)
                a = 5
            children_under_res = [key[1] for key in chain.child_dict.keys() if key[1] <= residue]
            positions_under_res = [pos for pos in positions if pos <= residue]
            if len(children_under_res) != len(positions_under_res):
                a = 5
        # self.record_positions(self.pdb_path, positions, chosen)
        try:
            assert len(chain.child_list) == len(positions)
        except AssertionError as e:
            self.record_failure(len(chain.child_list), len(positions))
            assert 0, e

        model.child_list[0] = chain
        struct.child_list[0] = model
        self.io.set_structure(struct)
        self.io.save(self.result_path)

    def record_failure(self, children, positions):
        with open(self.failures_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.pdb_path, str(children), str(positions)])

    @staticmethod
    def record_positions(pdb, positions, residues):
        with open(POSITIONS_PATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([pdb, str(positions), residues])



def main():
    Cutter("./sample_pdb")

if __name__ == "__main__":
    main()
