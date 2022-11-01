from scipy.spatial import distance

import networkx as nx
import numpy as np
import torch

ELEMENTS = [
    'H',
    'HE',
    'LI',
    'BE',
    'B',
    'C',
    'N',
    'O',
    'F',
    'NE',
    'NA',
    'MG',
    'AL',
    'SI',
    'P',
    'S',
    'CL',
    'AR',
    'K',
    'CA',
    'SC',
    'TI',
    'V',
    'CR',
    'MN',
    'FE',
    'CO',
    'NI',
    'CU',
    'ZN',
    'GA',
    'GE',
    'AS',
    'SE',
    'BR',
    'KR',
    'RB',
    'SR',
    'Y',
    'ZR',
    'NB',
    'MO',
    'TC',
    'RU',
    'RH',
    'PD',
    'AG',
    'CD',
    'IN',
    'SN',
    'SB',
    'TE',
    'I',
    'XE',
    'CS',
    'BA',
    'LA',
    'CE',
    'PR',
    'ND',
    'PM',
    'SM',
    'EU',
    'GD',
    'TB',
    'DY',
    'HO',
    'ER',
    'TM',
    'YB',
    'LU',
    'HF',
    'TA',
    'W',
    'RE',
    'OS',
    'IR',
    'PT',
    'AU',
    'HG',
    'TL',
    'PB',
    'BI',
    'PO',
    'AT',
    'RN',
    'FR',
    'RA',
    'AC',
    'TH',
    'PA',
    'U',
    'NP',
    'PU',
    'AM',
    'CM',
    'BK',
    'CF',
    'ES',
    'FM',
    'MD',
    'NO',
    'LR',
    'RF',
    'DB',
    'SG',
    'BH',
    'HS',
    'MT',
    'DS',
    'RG',
    'CN',
    'UUT',
    'FL',
    'UUP',
    'LV',
    'UUS',
    'UUO']


def encode_atom(atom):
    encoding = np.zeros(len(ELEMENTS))
    atom_index = ELEMENTS.index(atom.element)
    encoding[atom_index] += 1
    return encoding


def build_graph_from_atoms(atoms):
    coords = np.array([atom.coord for atom in atoms]).astype(np.int16)
    breakpoint()
    dists = distance.cdist(coords, coords)
    encodings = torch.tensor(np.array([encode_atom(atom) for atom in atoms]), requires_grad=True)
    return dists, encodings.to(torch.float32)


def main():
    print("Hello world")


if __name__ == "__main__":
    main()
