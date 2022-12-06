import os

DIR = "/cs/labs/dina/meitar/rhodopsins/aligned_pdbs/"

def search_for_missing_numbers(files_list):
    missing_files = []
    for i in range(850):
        found = False
        for file_name in files_list:
            search_str = '_'+str(i)+'['
            print(search_str, file_name)
            if search_str in file_name:
                found = True

        if not found:
            missing_files.append(i)

    print("missing files are", missing_files, len(missing_files))


def main():
    for dirpath, _, file_names in os.walk(DIR):
        search_for_missing_numbers(file_names)

if __name__ == "__main__":
    main()
