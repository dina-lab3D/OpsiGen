from descriptor_creator import DescriptorCreator

CRYO_FILE1 = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/emd_14141.map"
CRYO_FILE2 = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/7qti.mrc"
MODEL = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/project/cryo_model.pckl"


def main():
    print("Hello world")
    DescriptorCreator(CRYO_FILE1, CRYO_FILE2, MODEL, 15)


if __name__ == "__main__":
    main()
