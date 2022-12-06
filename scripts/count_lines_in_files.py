import os

INPUT_DIR = "/cs/labs/dina/meitar/rhodopsins/retinas/"



def count_lines(full_path):
    is_bad = False
    with open(full_path, "r+") as f:
        lines = f.readlines()
        if len(lines) != 23:
            is_bad = True

    with open("no_ret.txt", "a+") as f:
        if is_bad:
            f.write(full_path + '\n')

def main():

    for dirpath, _, file_names in os.walk(INPUT_DIR):
        for file_name in file_names:
            full_path = dirpath + file_name
            count_lines(full_path)
    

if __name__ == "__main__":
    main()
