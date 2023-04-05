#include "MolecularInterface.h"
#include "InterfacePointCloud.h"

#include <unistd.h>
#include <stdio.h>
#include <limits.h>
#include <sys/stat.h>
#include <cstdlib>


#define CHEM_LIB_IDX 1
#define MOL_IDX 2
#define CHAIN_IDX 3
#define LIGAND_IDX 4
#define RES_IDX 5
#define VOX_SIZE_IDX 6
#define X_IDX 7
#define Y_IDX 8
#define Z_IDX 9
#define OUT_FOLDER_IDX 10
#define DATASET_IDX 11



using std::ifstream;

class ChainNoWaterHydrogenSelector : public PDB::Selector {
public:
    // constructor
    ChainNoWaterHydrogenSelector(const std::string &chainID) : chains(chainID) {}

    virtual ~ChainNoWaterHydrogenSelector() {}

    bool operator()(const char *PDBrec) const
    {
        PDB::WaterHydrogenUnSelector sel;
        if (sel(PDBrec)) {
            PDB::ChainSelector chainSel(chains);
            return chainSel(PDBrec);
        }
        return false;
    }
private:
    std::string chains;
};

std::string getFileName(const std::string &filePath, bool withExtension = true, char seperator = '/')
{
    // Get last dot position
    std::size_t dotPos = filePath.rfind('.');
    std::size_t sepPos = filePath.rfind(seperator);

    if(sepPos != std::string::npos) {
        return filePath.substr(sepPos + 1, filePath.size() - (withExtension || dotPos != std::string::npos ? 1 : dotPos));
    }
    return "";
}

ChemMolecule get_contacts_molecule(const ChemLib &chem_lib,	const std::string &interface_chain_id,
		const std::string &ligand_chain_id, const std::string &filename, char *res)
{
	ChemMolecule interface, ligand;
	ifstream molFile1(filename);
	if (!molFile1) {
		std::cerr << "Can't open file " << filename << std::endl;
		exit(5);
	}

	interface.loadMolecule(molFile1, chem_lib, ChainNoWaterHydrogenSelector(interface_chain_id));
	molFile1.close();
	ifstream molFile2(filename);
    ligand.loadMolecule(molFile2, chem_lib, ChainNoWaterHydrogenSelector(ligand_chain_id));
	molFile2.close();
	interface.addMol2Type();
	ligand.addMol2Type();

	interface.computeASAperAtom();
	ligand.computeASAperAtom();

	// generate Interface
	float interfaceThr = atof(res);
	MolecularInterface molInterface(interface, ligand, interfaceThr);
	ChemMolecule contacts_mol;
	molInterface.outputInterfaceResidues(contacts_mol);

	return contacts_mol;
}


void dump_to_h5(ChemMolecule &contacts_mol, double vox_size, int x, int y, int z, const std::string &filename,
                const std::string &out_folder, double radius, const std::string &dataset_name)
{
    Vector3 center = contacts_mol.centroid();

    // move the contacts mol to grid center
    double factor = vox_size / 2.0;
    Vector3 gridCentroid = Vector3(x, y, z);
    gridCentroid *= factor;
    Vector3 translation = gridCentroid - center;
    contacts_mol.rigidTrans(RigidTrans3(Vector3(0, 0, 0), translation));
    InterfacePointCloud pointCloud(contacts_mol);


    const char *pdbName = basename(filename.c_str());
    size_t lastindex = std::string(pdbName).find_last_of('.');
    std::string pdbNameNoExt = std::string(pdbName).substr(0, lastindex);
//	std::cout << "Mol read " << pdbNameNoExt << ": " << interface.size() << " " << ligand.size() << std::endl;
    const std::string gridFileNameHDF5 = out_folder + '/' + pdbNameNoExt + "_" +
                                         std::to_string(int(radius)) + "_i.h5";

    struct stat info;

    if( stat( out_folder.c_str(), &info ) != 0 ) {
        printf("cannot access %s\n", out_folder.c_str());
        const std::string mkdir_ = "mkdir -p " + out_folder;
        const int dir_err = system(mkdir_.c_str());
        if (-1 == dir_err)
        {
            printf("Error creating directory!n");
            exit(1);
        }
    }
    else if( info.st_mode & S_IFDIR )  // S_ISDIR() doesn't exist on my windows
        printf( "%s is a directory\n", out_folder.c_str() );
    else
        printf( "%s is no directory\n", out_folder.c_str() );

    pointCloud.outputH5grid(gridFileNameHDF5, dataset_name);
}

void pipeline_to_interface(ChemLib& clib, char *argv[], const std::string &filename, const std::string &out_folder)
{
    ChemMolecule contacts_mol = get_contacts_molecule(clib, argv[CHAIN_IDX], argv[LIGAND_IDX], filename, argv[RES_IDX]);
    double radius = std::strtod(argv[RES_IDX], nullptr), vox_size = std::strtod(argv[VOX_SIZE_IDX], nullptr);
    int x = atoi(argv[X_IDX]), y = atoi(argv[Y_IDX]), z = atoi(argv[Z_IDX]);
    dump_to_h5(contacts_mol, vox_size, x, y, z, filename, out_folder, radius, argv[DATASET_IDX]);
}

int main(int argc, char *argv[]) {
    if (argc != 12) {
        std::cout << "usage " << argv[0] << "/path/to/chem.lib your_protein.pdb AB C 8.0 1.0 64 64 64 /path/to/outputfolder dataset" << std::endl;
        std::cout << "or with a file starting with pairs and containing in each line a couple of input pdb and output folder" << std::endl;
        return 0;
    }

    /* read molecules */
    ChemLib clib(argv[CHEM_LIB_IDX]);

    std::cerr << "chem lib done" << std::endl;

    std::string inputFile(getFileName(argv[MOL_IDX]));
    std::size_t found = inputFile.find("pairs");
    if (found!=std::string::npos) {
        std::cout << "parsing pairs file" << std::endl;
        ifstream pairs(argv[MOL_IDX]);
        if(!pairs) {
            std::cerr << "Can't open file " << argv[MOL_IDX] << std::endl;
            return 0;
        }

        std::string line;
        while (std::getline(pairs, line)) {
            std::istringstream lineStream(line);
            std::string mol_filename;
            std::string out_folder;
            std::getline(lineStream, mol_filename, '\t');
            std::getline(lineStream, out_folder, '\n');
            pipeline_to_interface(clib, argv, mol_filename, out_folder);
        }
        return 0;
    }
    else
        pipeline_to_interface(clib, argv, argv[MOL_IDX], argv[OUT_FOLDER_IDX]);
    return 0;
}
