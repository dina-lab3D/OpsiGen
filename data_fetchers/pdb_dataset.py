from torch.utils.data.dataset import Dataset
import Bio.PDB
from proteingraph import read_pdb
from parsers import data_parser


class PDBDataset(Dataset):
    def __init__(self, pdb_ids_list, path):
        self.ids_list = pdb_ids_list
        self.pdbl = Bio.PDB.PDBList()
        self.path = path

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, index):
        print("file is ", self.ids_list[index])
        try:
            graph = read_pdb("{}/{}".format(self.path, ("pdb" + self.ids_list[index] + ".ent")))
            return data_parser.graph_to_data(graph)
        except FileNotFoundError:
            raise ValueError("File not found")
            # print(self.pdbl.retrieve_pdb_file(pdb_code=self.ids_list[index], pdir=self.path, file_format='pdb'))
            # graph = read_pdb("{}/{}".format(self.path, ("pdb" + self.ids_list[index] + ".ent")))
        # finally:
        #     return data_parser.graph_to_data(graph)
