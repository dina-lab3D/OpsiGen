import os

def main():
    for dirpath, _, filenames in os.walk("/cs/labs/dina/meitar/rhodopsins"):
        for filename in filenames:
            if filename.startswith("slurm-") and filename.endswith(".out"):
                slurm_number = (filename.split(".")[0]).split("-")[1]
                print(slurm_number)
                os.system("scancel {}".format(slurm_number))
                

if __name__ == "__main__":
    main()

