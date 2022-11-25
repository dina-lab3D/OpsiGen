import os

OUTPUT_DIR="/cs/labs/dina/meitar/rhodopsins/protein_bank_pdbs"
INPUT_DIR = "/cs/labs/dina/meitar/rhodopsins/pdb_validator/"

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

    # print(input_file.replace('.seq', '[{}].pdb'.format(protein_ids[0][5])))

    os.system('wget https://files.rcsb.org/download/{}.pdb -O {}/{}'.format(protein_ids[0][:4], output_dir, input_file.replace('.seq', '[{}].pdb'.format(protein_ids[0][5]))))


def main():
    for dirpath, _, files in os.walk(INPUT_DIR):
        for file_name in files:
            if file_name.endswith('.seq'):
                # print(file_name)
                extract_pdb_from_found_blast_seq(file_name, OUTPUT_DIR)

    # print(lines[i+2: i+12])


if __name__ == "__main__":
    main()
