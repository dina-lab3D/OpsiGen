import os

LYS_DIR = "/cs/labs/dina/meitar/rhodopsins/lysins/"
RET_DIR = "/cs/labs/dina/meitar/rhodopsins/retinas/"
LYSIN_FORMAT = "lys{}.pdb"
RET_FORMAT = "match_{}[1].pdb"
OUT_DIR = LYS_DIR
OUT_FORMAT = "chosen_lys{}.pdb"

class Retinal:

    BASE_LEN = 20

    def __init__(self, lines):
        self.lines = lines
        self.get_atom(15)

    def get_atom(self, i):
        tokens = self.lines[i].split()
        return Atom(float(tokens[-5]), float(tokens[-4]), float(tokens[-3]))

class Atom:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def _dist(self, other_atom):
        return (self.x - other_atom.x) ** 2 + (self.y - other_atom.y) ** 2 + (self.z - other_atom.z) ** 2

    def find_closest(self, list_of_atoms):
        closest_index = 0
        closest_dist = 100000000000
        for i, atom in enumerate(list_of_atoms):
            if self._dist(atom) < closest_dist:
                closest_index = i
                closest_dist = self._dist(atom)

        return closest_index

class Lysin:

    BASE_LEN = 9

    def __init__(self, lines):
        self.lines = lines
        self.lysins = self.seperate_to_different_lysins()
        nitrogens = self.get_nitrogen_atoms()

    def seperate_to_different_lysins(self):
        result = []
        for i in range(int(len(self.lines) / self.BASE_LEN)):
            result.append(self.lines[i * self.BASE_LEN: (i+1) * self.BASE_LEN])

        return result

    def get_nitrogen_atoms(self):
        result = []
        for lys in self.lysins:
            tokens = lys[8].split()
            result.append(Atom(float(tokens[-5]), float(tokens[-4]), float(tokens[-3])))

        return result



def main():
    for i in range(884):
        with open(LYS_DIR + LYSIN_FORMAT.format(i), "r") as f:
            lys_lines = f.readlines()

        with open(RET_DIR + RET_FORMAT.format(i), "r") as f:
            ret_lines = f.readlines()

        my_lys = Lysin(lys_lines)
        my_ret = Retinal(ret_lines)
        c15_atom = my_ret.get_atom(15)
        chosen_lys = c15_atom.find_closest(my_lys.get_nitrogen_atoms())
        with open(OUT_DIR + OUT_FORMAT.format(i), "w") as f:
            for i in range(Lysin.BASE_LEN * chosen_lys, (chosen_lys + 1) * Lysin.BASE_LEN):
                f.write(my_lys.lines[i])
        print(chosen_lys)

if __name__ == "__main__":
    main()
