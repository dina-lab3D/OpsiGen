import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

HISTFILE = "/cs/labs/dina/meitar/rhodopsins/outputs/model{}_GAT21_30_40"
TITLE_TEST = "Split {} test wavelength histogram"
TITLE_FULL = "Split {} histogram"

def create_img_full(filename, title, name_end):
    with open(filename + name_end, "r") as f:
        data = f.read()

    print(filename + name_end + "_img.png")
   
    wavelengths = [float(i) for i in data.split()]
    plt.figure()
    plt.xlabel("wavelength")
    sns.kdeplot(data=wavelengths, fill=True)
    plt.title("Full Dataset Wavelength Spectra")
    plt.savefig("/cs/labs/dina/meitar/rhodopsins/outputs/full_dataset_histogram" + "_img.png", dpi=200)

def create_img_all():
    arr = {}
    for i in range(4):
        with open(HISTFILE.format(i) + "_spctra_test", "r") as f:
            data = f.read()

        wavelengths = [float(i) for i in data.split()]
        arr[str(i)] = np.array(wavelengths)
    plt.figure()
    plt.xlabel("wavelength")
    for i in range(4):
        sns.kdeplot(data=arr[str(i)], fill=True, common_grid=True, label="split {}".format(i), palette="RdBu")
    plt.title("Test Splits Wavelength Spectra")
    plt.legend()
    plt.savefig("/cs/labs/dina/meitar/rhodopsins/outputs/splits_histogram" + "_img.png", dpi=200)



def main():
    filename = HISTFILE.format(0)
    test_title = TITLE_TEST.format(0)
    full_title = TITLE_FULL.format(0)
    create_img_full(filename, full_title, "_spectra_full")
    create_img_all()


if __name__ == "__main__":
    main()
