import os

DIR = "/cs/labs/dina/meitar/rhodopsins/new_fastas"

def find_missing_files(folder):
    file_names = []
    for root,dirs,files in os.walk(folder):
        file_names = files

    good_files = []
    for file_name in files:
        if not file_name.endswith(".pdb"):
            continue
        file_num = int(file_name.split("_")[0])
        good_files.append(file_num)

    print(list(set(range(928)) - set(good_files)))
    print(len(list(set(range(928)) - set(good_files))))

def main():
    find_missing_files(DIR)
    

if __name__ == "__main__":
    main()
