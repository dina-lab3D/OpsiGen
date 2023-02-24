import os

DIR = "/cs/labs/dina/meitar/rhodopsins/fastas"

def find_missing_files(folder):
    file_names = []
    for root,dirs,files in os.walk(folder):
        file_names = files

    missing_fastas = []
    for i in range(884):
        fasta_here = False
        for file_name in files:
            if file_name.startswith(str(i)):
                fasta_here = True

        if not fasta_here:
            missing_fastas.append(i)

    print(missing_fastas)

def main():
    find_missing_files(DIR)
    

if __name__ == "__main__":
    main()
