"""
Read the sequences from the fasta file.
This class has an API for dealing with the 24 amino acids that are most
related to the wavelength absorption
"""

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
    """
    Read the sequences from the fasta file.
    This class has an API for dealing with the 24 amino acids that are most
    related to the wavelength absorption
    """

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

    def __init__(self, config):
        self.config = config
        with open(config["sequences"], "r") as f:
            self.lines = f.readlines()

        self.index_mapping = self.get_wavelength_indexes_from_file()
        self.length = len(self.index_mapping)

    def get_wavelength_indexes_from_file(self):
        """
        Get only the non-NA rhodopsins according to the known wavelengths
        file
        """
        with open(self.config["wavelength_file"], "r") as f:
            wl_lines = f.readlines()

        indexes = []
        for i, wl_line in enumerate(wl_lines):
            if wl_line != 'NA\n':
                indexes.append(i)


        return {i: indexes[i] for i in range(len(indexes))}

    def _get_relevant_line(self, protein_index):
        """
        Get the sequence from the .fas file according to the index
        """
        return self.lines[2 * protein_index + 1].strip()

    def get_item(self, protein_index, seq_index):
        """
        get rhodopsin[protein_index][seq_index]
        """
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
            return -1

        return amino_index

    def get_interesting_positions(self, protein_index):
        """
        get the list of 24 amino acids that are corelated with 
        wavelength absorption
        """
        res = [self.get_item(protein_index, p) for p in self.POSITIONS]
        for i in range(len(res) - 1):
            if res[i] == res[i+1]:
                res[i+1] += 1
        self._get_relevant_line(protein_index).replace('-','')

        return res
