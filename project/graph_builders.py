import numpy as np
import networkx as nx
import biological_data
import time

def compute_distance(atoms):
    N = len(atoms)
    result = np.zeros((N, N))

    print("N is ", N)

    for i in range(N):
        for j in range(N):
            result[i][j] = np.linalg.norm(atoms[i].get_vector() - atoms[j].get_vector())

    return result


def calculate_graph_desc(atoms, feature_length, distmat):
    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    for i in range(len(atoms)):
        for j in range((len(atoms))):
            G.add_edge(i, j, weight=distmat[i][j])
    result = nx.laplacian_spectrum(G)[::-1][:feature_length]
    return np.hstack((result, np.zeros(feature_length - result.shape[0])))


def compute_hydrophobic_feature(atoms, feature_length, distmat):
    """
    Find all hydrophobic interactions.
    Performs searches between the following residues:
    ALA, VAL, LEU, ILE, MET, PHE, TRP, PRO, TYR
    Criteria: atoms of this residues are within 5A distance.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    # distmat = compute_distance(atoms)
    atom_couples = np.argwhere(distmat < 5)
    for i, j in atom_couples:
        res_i = atoms[i].parent.get_resname()
        res_j = atoms[j].parent.get_resname()
        if res_i in biological_data.HYDROPHOBIC_RESIS and res_j in biological_data.HYDROPHOBIC_RESIS:
            G.add_edge(i, j)

    return nx.laplacian_spectrum(G)[::-1][:feature_length]


def compute_ionic_feature(atoms, feature_length, distmat):
    """
    Find all ionic interactions.
    Criteria: ARG, LYS, HIS, ASP, and GLU residues are within 6A.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    # distmat = compute_distance(atoms)
    atom_couples = np.argwhere(distmat < 6)
    for i, j in atom_couples:
        res_i = atoms[i].parent.get_resname()
        res_j = atoms[j].parent.get_resname()
        if (res_i in biological_data.POS_AA and res_j in biological_data.NEG_AA) or \
                (res_i in biological_data.NEG_AA and res_j in biological_data.POS_AA):
            G.add_edge(i, j)

    return nx.laplacian_spectrum(G)[::-1][:feature_length]


def compute_hydrogen_feature(atoms, feature_length, distmat):
    """Add all hydrogen-bond interactions."""
    # For these atoms, find those that are within 3.5A of one another.
    HBOND_ATOMS = [
        "ND",  # histidine and asparagine
        "NE",  # glutamate, tryptophan, arginine, histidine
        "NH",  # arginine
        "NZ",  # lysine
        "OD1",
        "OD2",
        "OE",
        "OG",
        "OH",
        "N",
        "O",
    ]

    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    # distmat = compute_distance(atoms)
    atom_couples = np.argwhere(distmat < 3.5)
    for i, j in atom_couples:
        atom_i = atoms[i].name
        atom_j = atoms[j].name
        if atom_i in HBOND_ATOMS and atom_j in HBOND_ATOMS:
            G.add_edge(i, j)

    HBOND_ATOMS_SULPHUR = ["SD", "SG"]
    atom_couples = np.argwhere(distmat < 4)
    for i, j in atom_couples:
        atom_i = atoms[i].name
        atom_j = atoms[j].name
        if atom_i in HBOND_ATOMS_SULPHUR and atom_j in HBOND_ATOMS_SULPHUR:
            G.add_edge(i, j)

    return nx.laplacian_spectrum(G)[::-1][:feature_length]


def compute_cation_pi_feature(atoms, feature_length, distmat):
    """Add cation-pi interactions."""
    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    # distmat = compute_distance(atoms)
    atom_couples = np.argwhere(distmat < 6)
    for i, j in atom_couples:
        res_i = atoms[i].parent.get_resname()
        res_j = atoms[j].parent.get_resname()
        if (res_i in biological_data.CATION_RESIS and res_j in biological_data.PI_RESIS) or \
                (res_i in biological_data.PI_RESIS and res_j in biological_data.CATION_RESIS):
            G.add_edge(i, j)

    return nx.laplacian_spectrum(G)[::-1][:feature_length]


def compute_disulfide_feature(atoms, feature_length, distmat):
    """
    Find all disulfide interactions between CYS residues.
    Criteria: sulfur atom pairs are within 2.2A of each other.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    # distmat = compute_distance(atoms)
    atom_couples = np.argwhere(distmat < 2.2)
    for i, j in atom_couples:
        res_i = atoms[i].parent.get_resname()
        atom_i = atoms[i].name
        res_j = atoms[j].parent.get_resname()
        atom_j = atoms[j].name
        if (res_i in biological_data.DISULFIDE_RESIS and atom_i in biological_data.DISULFIDE_ATOMS) and \
                (res_j in biological_data.DISULFIDE_RESIS and atom_j in biological_data.DISULFIDE_ATOMS):
            G.add_edge(i, j)

    return nx.laplacian_spectrum(G)[::-1][:feature_length]
