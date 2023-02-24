import os

DIR = "/cs/labs/dina/meitar/rhodopsins/new_fastas/"

def search_for_missing_numbers(files_list):
    missing_files = []
    for i in range(884):
        found = False
        for file_name in files_list:
            if not file_name.endswith(".pdb"):
                continue
            tokens = file_name.split("_")
            unrelaxed_index = tokens.index("unrelaxed")
            if i == int(tokens[unrelaxed_index - 1]):
                found = True

        if not found:
            missing_files.append(i)

    print("missing files are", missing_files, len(missing_files))


def main():
    for dirpath, _, file_names in os.walk(DIR):
        search_for_missing_numbers(file_names)

if __name__ == "__main__":
    main()
