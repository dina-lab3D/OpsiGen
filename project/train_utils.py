from dataclasses import dataclass
import numpy as np

MODEL_PATH = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/project/cryo_model.pckl"
CRYO_FILE1 = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/emd_14141.map"
CRYO_FILE2 = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/7qti.mrc"


class PerformanceStats:

    def __init__(self, reset_steps=1000):
        self.correct = 1
        self.wrong = 1
        self.super_correct = 1
        self.super_wrong = 1
        self.step = 0
        self.reset_steps = reset_steps
        self.ratios = []

    def reset(self):
        self.correct = 1
        self.wrong = 1
        self.super_correct = 1
        self.super_wrong = 1
        self.step = 0
        self.ratios = []

    def update_stats(self, ratio):
        self.ratios.append(ratio)

    def advance(self):
        self.step += 1
        return self.step % self.reset_steps == 0

    def print(self):
        print("average ratio is {}".format(np.mean(np.array(self.ratios))))


@dataclass
class TrainerData:
    alpha = 1
    lr = 0.001
    threas = 15
    to_load = True
    model_path = MODEL_PATH
    path_dict = {}

    SEPERATOR = " "

    def __init__(self, data_file):
        self.parse_data_file(data_file)

    def parse_data_file(self, data_file):
        with open(data_file, "r") as f:
            lines = f.readlines()
        parsed_lines = [line.replace('\n', '').split(TrainerData.SEPERATOR) for line in lines]
        assert any([(len(arr) == 2) for arr in parsed_lines])
        self.path_dict = {line[0]: line[1] for line in parsed_lines}


if __name__ == "__main__":
    td = TrainerData("./data.txt")
