import os
from tqdm import tqdm
from Bio import PDB
from Bio.PDB import PDBIO
from Bio import pairwise2
from Bio import SeqIO
from Bio.pairwise2 import format_alignment
import subprocess
from .sequences import Sequences
import json
import shutil

ALIGNED_RESULT_PATH = "./alignments/fm_ams/"
ALL_PDBS = "./alignments/fm_ams/pdb_folder/"
BR_POS = [19, 48, 52, 82, 84, 85, 88, 89, 92, 117, 118, 121, 137, 140, 141, 144, 181, 184, 185, 188, 207, 211, 214, 215]

class Aligner:

    def __init__(self, config):
        self.seq = Aligner._parse_seq(config["pdb_path"])
        self.db = Sequences(config)
        self.best_match, self.best_index = None, None
        self.aligning_index = config["aligning_index"]
        self.alignment_path = config["sequences"]
        self.seq_path = config["fasta_path"]
        self.aligned_id = config["aligned_id"]
        self.pdb_path = config["pdb_path"]


    @staticmethod
    def _parse_seq(pdb_file):
        parser = PDB.PDBParser(QUIET=True)
        # Parse the structure from the PDB file
        structure = parser.get_structure('structure', pdb_file)
        # Initialize variables
        sequence = None
        # Iterate through the models (usually just one model in most PDB files)
        for model in structure:
            # Check if the model contains any chains
            if len(model) > 0:
                # Get the first chain
                chain = list(model)[0]

                # Extract the sequence of the chain
                polypeptides = PDB.PPBuilder().build_peptides(chain)
                if polypeptides:
                    # Assuming there is only one polypeptide chain
                    poly = polypeptides[0]
                    sequence = str(poly.get_sequence())
                    break
        if sequence is None:
            raise ValueError("No sequences found in the PDB file.")
        return sequence

    def _align_to_entry(self, entry_number):
        curr_seq = self.db._get_relevant_line(entry_number)
        aligned = pairwise2.align.globalxx(self.seq, curr_seq)[0]

        return aligned

    def _align_to_known_index(self, aligning_index):
        print("Aligning the given sequence to the given entry in the DB")
        aligned = self._align_to_entry(aligning_index)
        best_match = aligned.seqA
        best_score = aligned.score

        return best_match, aligning_index

    def _align_to_best_match(self):
        print("Aligning the given sequence to the DB and finding best match:")
        best_index = 0
        best_match = ''
        best_score = 0
        for i in tqdm(range(self.db.length)):
            aligned = self._align_to_entry(i)
            if aligned.score > best_score:
                best_index = i
                best_match = aligned.seqA
                best_score = aligned.score

        print("Best index is", best_index)

        return best_match, best_index

    def align(self):
        if self.aligning_index > 0:
            best_match, best_index = self._align_to_known_index(self.aligning_index)
        else:
            best_match, best_index = self._align_to_best_match()

        return best_match, best_index

    @staticmethod
    def get_sequence_by_name(fasta_file, sequence_name):
        """
        Extract sequence from a multi-FASTA file by sequence name.

        Parameters:
        - fasta_file: path to the multi-FASTA file
        - sequence_name: name of the sequence to extract

        Returns:
        - sequence: string containing the sequence
        """
        with open(fasta_file, "r") as f:
            lines = f.readlines()
        seq_start_ind = [i for i in range(len(lines)) if lines[i].strip() == ">" + sequence_name][0] + 1
        i = 0
        res = ""
        while seq_start_ind + i < len(lines) and lines[seq_start_ind + i][0] != ">":
            res += lines[seq_start_ind + i].strip()
            i += 1
        return res

    @staticmethod
    def calc_single_pos(pos, seq):
        return len(seq[:pos].replace("-", ""))

    @staticmethod
    def global_by_local_positions(seq, lp):
        return [[i for i in range(len(seq)) if len(seq[:i + 1].replace("-", "")) == pos + 1][0] for pos in lp]

    def get_amino_acids_of_aligned_sequence(self):
        new_align_path = ALIGNED_RESULT_PATH + self.aligned_id
        file_name = os.path.basename(self.pdb_path)
        dest_file = os.path.join(ALL_PDBS, "second_" + file_name)
        shutil.copy2(self.pdb_path, dest_file)
        mason_command = f"./alignments/foldmason/run_mason.sh" \
                        f" {ALL_PDBS} {new_align_path} > ./alignments/foldmason/run_output.txt"
        subprocess.run(mason_command, shell=True)
        os.remove(dest_file)
        seq = Aligner.get_sequence_by_name(new_align_path + "_aa.fa", "second_" + file_name[:-len(".pdb")])
        br_seq = Aligner.get_sequence_by_name(new_align_path + "_aa.fa", "59_af_full")
        glob_pos = Aligner.global_by_local_positions(br_seq, BR_POS)
        positions = [Aligner.calc_single_pos(pos+1, seq) + (seq[pos] == "-") for pos in glob_pos]

        return [pos for pos in positions]


def main():
    with open(TEMP_CONFIG, "r") as f:
        data = f.read()
    config = json.loads(data)
    aligner = Aligner(config)
    positions = aligner.get_amino_acids_of_aligned_sequence()
    residues = [aligner.seq[pos] for pos in positions]
    print(''.join(residues))
    print(residues[8] + residues[18] + residues[23])


if __name__ == "__main__":
    main()
