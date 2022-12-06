import os

OUTPUT_DIR="/cs/labs/dina/meitar/rhodopsins/protein_bank_pdbs/"
INPUT_DIR = "/cs/labs/dina/meitar/rhodopsins/pdb_validator/"


index_dict = {
        #742:2,
        #746:2,
        #735:2,
        #724:2,
        #734:2,
        #725:2,
        #735:2,
        #752:2,
        #736:2,
        #749:2,
        #744:2,
        #719:2,
        #751:2,
        #751:2,
        #485: 2,
        #750: 2,
        #747: 2,
        #743: 2,
        #752: 2,
        #723: 2,
        #748: 2,
        794: 3,
        170: 3,
        174: 3,
        171: 3,
        172: 3,
        173: 3,
        175: 3,
        176: 3,
        177: 3,
        178: 3,
        179: 3,
        180: 3,
        181: 3,
        182: 3,
        183: 3,
        617: 3,
        80: 6,
        81: 6,
        82: 6,
        83: 6,
        84: 6,
        85: 6,
        182: 3,
        101: 5,
        102: 5,
        602: 6,
        551: 9,
        550: 9,
        566: 9,
        604: 9,
        772: 3,
        65: 6,
        128: 5,
}

def get_index_from_name(full_file_name):
    file_name = full_file_name.split('/')[-1]
    end_of_index = file_name.index('-')

    return int(file_name[:end_of_index])
    

def extract_pdb_from_found_blast_seq(input_file, output_dir):
    with open(input_file, "r") as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if 'alignments' in lines[i]:
            break

    relevant_lines = lines[i+2:i+12]
    tokens = []
    protein_ids = []
    for line in relevant_lines:
        tokens = line.split()
        protein_ids.append(tokens[0])

    if not protein_ids:
        print(input_file, "failed")
        return

    entry_index = get_index_from_name(input_file)
    special_place = entry_index in index_dict.keys()

    index_from_seq_file = (index_dict[entry_index] - 1) if special_place else 0
    file_name = input_file.split('/')[-1]

    if entry_index != 128:
        return

    """
    if index_from_seq_file == 0:
        print("Not relevant")
        return
    """

    print(entry_index)

    # pdb_id = protein_ids[0][:4]
    # print(protein_ids[index_from_seq_file][:4])
    # print(input_file.replace('.seq', '[{}].pdb'.format(protein_ids[0][5])))

    os.system('wget https://files.rcsb.org/download/{}.pdb -O {}/{}'.format(protein_ids[index_from_seq_file][:4], output_dir, file_name.replace('.seq', '[{}].pdb'.format(protein_ids[index_from_seq_file][5]))))


def main():
    for dirpath, _, files in os.walk(INPUT_DIR):
        for file_name in files:
            if file_name.endswith('.seq'):
                # print(file_name)
                extract_pdb_from_found_blast_seq(INPUT_DIR + file_name, OUTPUT_DIR)

    # print(lines[i+2: i+12])


if __name__ == "__main__":
    main()
