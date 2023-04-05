import os
from tqdm import tqdm
from Bio import PDB
from Bio.PDB import PDBIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment

from sequences import Sequences

class Aligner:

    def __init__(self, config):
        self.seq = Aligner._parse_fasta(config["fasta_path"])
        self.db = Sequences(config)
        self.best_match, self.best_index = None, None
        self.aligning_index = config["aligning_index"]

    def _parse_fasta(fasta_file):
        with open(fasta_file, "r") as f:
            lines = f.readlines()

        return lines[1]

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

    def get_amino_acids_of_aligned_sequence(self):
        self.best_match, self.best_index = self.align()
        positions = self.db.get_interesting_positions(self.best_index)

        return positions


def main():
    aligner = Aligner(TEMP_SEQ, 170)
    aligner.get_amino_acids_of_aligned_sequence()

if __name__ == "__main__":
    main()
