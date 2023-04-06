from Bio import PDB
from Bio.PDB import PDBIO

class Cutter:

    def __init__(self, config):
        self.pdb_path = config["pdb_path"]
        self.parser = PDB.PDBParser()
        self.io = PDBIO()
        self.result_path = config["cutted_result_pdb_path"]

    def cut_pdb(self, aligner_obj):
        positions = aligner_obj.get_amino_acids_of_aligned_sequence()

        print("Cutting the relevant amino acids, and outputing them to a result pdb")

        struct = self.parser.get_structure(self.pdb_path, self.pdb_path)
        model = struct.child_list[0]
        chain = model.child_list[0]
        length = len(chain.child_list)
        for residue in range(1, length+1):
            if residue not in positions:
                chain.detach_child((" ", residue, " "))

        assert len(chain.child_list) == len(positions)

        model.child_list[0] = chain
        struct.child_list[0] = model
        self.io.set_structure(struct)
        self.io.save(self.result_path)

def main():
    Cutter("./sample_pdb")

if __name__ == "__main__":
    main()
