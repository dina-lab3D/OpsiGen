import os
import numpy as np

SEQFILE = "./new_sequence.fas"



class Seq:

    RETINA_INDEXES = [429, 465, 469, 597, 599, 600, 603, 604, 607, 651, 652, 655, 687, 690, 691, 694, 774, 777, 778, 781, 828, 832, 835, 836]
    
    def __init__(self, seq):
        self.seq = seq
        self.aligned_indexes, self.special_indexes= self._get_aligned_indexes()

    def _get_aligned_indexes(self):
        aligned_indexes = []
        special_indexes = []
        current_index = 0
        current_amino = 0
        while current_index < len(self.seq):
            if self.seq[current_index] != '-':
                aligned_indexes.append(current_index)
                current_amino += 1

            if current_index in Seq.RETINA_INDEXES:
                if self.seq[current_index] == "-":
                    special_indexes.append(-1)
                else:
                    special_indexes.append(current_amino)

            current_index += 1

        return aligned_indexes, special_indexes

    def get_num_feature(self, num):
        result = np.zeros(len(self.special_indexes))
        if num == -1:
            return result

        if num in self.special_indexes:
            result[self.special_indexes.index(num)] = 1

        return result


class Alignment:

    def __init__(self, seqfile):
        sequences = self._parse_seq_file(seqfile)
        self.seqs = [Seq(string) for string in sequences]

    def _parse_seq_file(self, seqfile):
        with open(seqfile, "r") as f:
            lines = f.readlines()
        
        return [lines[i][1:-1] for i in range(len(lines)) if i % 2 == 1]

    def __getitem__(self, i):
        return self.seqs[i]

    def __len__(self):
        return len(self.seqs)



def main():
    align = Alignment(SEQFILE)

if __name__ == "__main__":
    main()
