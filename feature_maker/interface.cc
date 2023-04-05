#include "MolecularInterface.h"
#include "cnpy.h"

#include <unistd.h>
#include <stdio.h>
#include <limits.h>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <boost/program_options.hpp>

#include <boost/filesystem.hpp>
#include <boost/lambda/bind.hpp>

using namespace boost;
using namespace boost::lambda;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

using std::ifstream;
class ALLSelector : public PDB::Selector {
public:
    bool operator()(const char *PDBrec) const
    {
        return true;
    }
};


/***
 * Function that does the same as pipeline to interface but instead works specifically for graph representations on probes.
 * most importantly the function saves all probe graphs under one hierarchical file
 * @param clib
 * @param filename
 * @param radius
 * @param voxelSize
 * @param n
 * @param out_folder
 * @param selector
 * @param graph_representation
 * @param num_of_mg
 * @param ratio
 * @param probe
 * @return
 */
int graph_inference_pipeline(ChemLib& clib, std::string filename, const float radius, std::string out_folder)
{
	ChemMolecule mol;
	ifstream molFile1(filename);
	if(!molFile1) {
		std::cerr << "Can't open file " << filename << std::endl;
		return 0;
	}
	mol.loadMolecule(molFile1, clib, ALLSelector());
	molFile1.clear();
	molFile1.close();
	// mgIons.setMol2TypeToUnk(); //Make sure Mgions / Water molecules don't give away what type they are!
	mol.addMol2Type();
	mol.computeASAperAtom();

    std::vector<float> data_;

    // TODO make constant
    int FEATURE_SIZE = 18; // 15 mol2 types-3, charge and asa

    // and radius and 3 position
    data_.insert(data_.begin(), mol.size() * FEATURE_SIZE, 0.0);
    for (unsigned int i = 0; i < mol.size(); i++) {
        ChemAtom atom = mol[i];
        Vector3 point = atom.position();
        float radius = atom.getRadius();
        MOL2_TYPE mol2type = atom.getMol2Type();
        float charge = atom.getCharge();
        float asa = atom.getASA();
	data_[i * FEATURE_SIZE + mol2type] = 1;
        data_[i * FEATURE_SIZE + 15] = asa;
        data_[i * FEATURE_SIZE + 16] = charge;
        data_[i * FEATURE_SIZE + 17] = radius;
    }
    

    if (data_.size() > 0) {
        cnpy::npy_save(out_folder, data_, "w");
    }

    return 0;
}


int main(int argc, char** argv) {
    double radius = 8.0, cs = 1.0, vox_size = 1.0;
    int dimension = 3, max_clouds = 10, x = 32, y = 32, z = 32;

    std::string selector = "MG";
    std::string input_dir = ".", input_text_file, output_dir, meta_path, atom_str, chem_lib_path = "../feature_maker/chem.lib";
    po::options_description desc("Allowed options");
    desc.add_options()
            ("input-dir,i", po::value<std::string>(&input_dir)->default_value(input_dir),
             "a path to the directory that contains pdb entries")
            ("selector, s", po::value<std::string>(&selector)->default_value(selector),
             "Choose selector which chooses which atoms have their surroundings examined. Default is MGselector; Options: MG, H20")
            ("output-dir,o", po::value<std::string>(&output_dir),
             "a path to the directory where the cuts will be stored (=input_dir/cuts)")
            ("meta,m", po::value<std::string>(&meta_path),
             "a path to the file that describes the output in form\n"
             "cloud_number (center_atom): num_of_cloud_atoms (=output_dir/meta.txt)")
            ("atom,a", po::value<std::string>(&atom_str),
             "the name of the center atom")
            ("help,h", "produce help message")
            ("cube-size,c", po::value<double>(&cs)->default_value(cs), "geohash resolution")
            ("dimension,d", po::value<int>(&dimension)->default_value(dimension), "geohash dimension")
            ("radius,r", po::value<double>(&radius)->default_value(radius), "radius of analysis in angstrom")
            ("max-clouds,M", po::value<int>(&max_clouds)->default_value(max_clouds),
             "restricts the maximum number of clouds per pdb entry")
            ("chem-lib,l", po::value<std::string>(&chem_lib_path)->default_value(chem_lib_path),
             "a path to the 'chem.lib' file")
            ("voxel-size", po::value<double>(&vox_size)->default_value(vox_size))
            ("x,x", po::value<int>(&x)->default_value(x), "the number of Angstroms one voxel has in x direction")
            ("y,y", po::value<int>(&y)->default_value(y), "the number of Angstroms one voxel has in y direction")
            ("z,z", po::value<int>(&z)->default_value(z), "the number of Angstroms one voxel has in z direction");


	po::positional_options_description p;
	p.add("input-dir", 1).add("atom", 1).add("dataset-name", 1);
	po::variables_map vm;
	po::command_line_parser parser{argc, argv};
	po::store(parser.options(desc).positional(p).run(), vm);
	po::notify(vm);


	if (vm.count("help")) {
		std::cout << "Usage: options_description [options]" << std::endl;
		std::cout << desc;
		return 0;
	}

	// here is the old main
	const char* cstr_chem_path = chem_lib_path.c_str();
	ChemLib clib((cstr_chem_path));
	std::cerr << "chem lib done" << std::endl;

	std::vector<fs::path> dir_vec;
    dir_vec = std::vector<fs::path>(begin(fs::directory_iterator(input_dir)), end(fs::directory_iterator(input_dir)));

	std::cout << "files that this batch is working on are\n:";
	for(auto& dirEntry : dir_vec){
		std::cout << dirEntry << '\n';
	}

	for(auto& dirEntry : dir_vec)
	{
		// notice x,y,z should all be the same size
		std::string new_output = output_dir + "/" + dirEntry.filename().string();

		if(dirEntry.extension().string() == ".pdb")
		{
		    //check if file already exists in some capacity in the raw folder.
		    const std::size_t dotIndex = dirEntry.native().rfind("/");
		    const std::size_t length = dirEntry.native().length();
		    const std::string pdbName = dirEntry.native().substr(dotIndex + 1, (length - 4) - (dotIndex + 1));
		    const std::string output_file_name = output_dir + "/" + pdbName + ".npz";

		    graph_inference_pipeline(clib,  dirEntry.native(), radius, output_dir + pdbName + ".npz");
		}
	}
}
