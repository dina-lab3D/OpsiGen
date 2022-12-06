import os

INPUT_DIRECTORY = "/cs/labs/dina/meitar/rhodopsins/protein_bank_pdbs/"
OUTPUT_DIRECTORY = "/cs/labs/dina/meitar/rhodopsins/chains/"

SPECIAL_CASES_DICT = {
    480: 'AE',
}

def get_chain_from_file(file_name):
    start = file_name.index('[')
    end = file_name.index(']')

    for key in SPECIAL_CASES_DICT.keys():
        if str(key) in file_name:
            return SPECIAL_CASES_DICT[key], OUTPUT_DIRECTORY + file_name[:start] + file_name[end + 1:]
        else:
            return file_name[start + 1], OUTPUT_DIRECTORY + file_name[:start] + file_name[end + 1:]


def main():
    for dirpath, _, file_names in os.walk(INPUT_DIRECTORY):
        for file_name in file_names:
            if '128' not in file_name:
                continue
            if file_name.endswith('].pdb'):
                chain, new_file_path = get_chain_from_file(file_name)
                print(file_name)
                cmd = '/cs/staff/dina/utils/getChain.Linux {} {} > {}'.format(chain, dirpath + file_name, new_file_path)
                print(cmd)
                os.system(cmd)

if __name__ == "__main__":
    main()
