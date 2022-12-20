import matplotlib.pyplot as plt
import numpy as np

IMG_PATH = "loss_train.npy"

def main():
    arr = np.load(IMG_PATH)
    plt.plot(arr)
    plt.savefig("try.png")


if __name__ == "__main__":
    main()
