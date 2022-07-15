from torch.utils.data.dataset import Dataset
import Bio.PDB
from proteingraph import read_pdb
from parsers import data_parser


class PDBDataset(Dataset):
    """
    This dataset reads from a pdblist and returns a graph instance of this pdb
    """

    def __init__(self, pdb_ids_list, path):
        """
        This dataset reads from a pdblist and returns a graph instance of this pdb
        :param pdb_ids_list: list of PDBs
        :param path: the path to save the PDBs in
        """
        self.ids_list = pdb_ids_list
        self.pdbl = Bio.PDB.PDBList()
        self.path = path

    def __len__(self):
        """
        :return: length of the PDB
        """
        return len(self.ids_list)

    def __getitem__(self, index):
        """
        Get the item in the given index
        :param index: index of the list to retrieve
        """
        try:
            graph = read_pdb("{}/{}".format(self.path, ("pdb" + self.ids_list[index] + ".ent")))
            return data_parser.graph_to_data(graph)
        except FileNotFoundError:
            raise ValueError("File not found")
            # print(self.pdbl.retrieve_pdb_file(pdb_code=self.ids_list[index], pdir=self.path, file_format='pdb'))
            # graph = read_pdb("{}/{}".format(self.path, ("pdb" + self.ids_list[index] + ".ent")))
        # finally:
        #     return data_parser.graph_to_data(graph)
