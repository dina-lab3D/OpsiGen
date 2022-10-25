import pandas as pd
import os

DATA_FILE_PATH = "/mnt/c/Users/Zlils/Documents/university/biology/cryo-folding/cryo-em-data/csvs/1.csv"

EMDB_DOWNLOADER_PATH = "/mnt/c/Users/Zlils/Documents/university/biology/cryo-folding/project/download_emdb.sh"

def download_map_data(emdb_id, emdb_path):
    os.system('{} {} {}'.format(EMDB_DOWNLOADER_PATH, emdb_id, emdb_path))

def parse_csv(file_name):
    breakpoint()
    data = pd.read_csv(file_name)
    for i in range(len(data)):
        download_map_data(data.iloc[i][2][4:], "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/my_data")

def main():
    print(parse_csv(DATA_FILE_PATH))


if __name__ == "__main__":
    main()
