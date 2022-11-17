import Bio.PDB

FIRST_PDB_PATH = "/cs/labs/dina/meitar/rhodopsins/pdbs/SRII.-Rh225-253_NpSRII_745_unrelaxed_rank_1_model_5.pdb"

SECOND_PDB_PATH = "/cs/labs/dina/meitar/rhodopsins/protein_bank_pdbs/744-SRII..Rh225-252.pdb"

def main():
    # Start the parser
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)

    # Get the structures
    ref_structure = pdb_parser.get_structure("reference", FIRST_PDB_PATH)
    sample_structure = pdb_parser.get_structure("samle", SECOND_PDB_PATH)

    # Use the first model in the pdb-files for alignment
    # Change the number 0 if you want to align to another structure
    ref_model    = ref_structure[0]
    sample_model = sample_structure[0]

    # Make a list of the atoms (in the structures) you wish to align.
    # In this case we use CA atoms whose index is in the specified range
    ref_atoms = [atom for atom in ref_model.get_atoms()]
    sample_atoms = [atom for atom in sample_model.get_atoms()]

    min_length = min(len(ref_atoms), len(sample_atoms))


    # Now we initiate the superimposer:
    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms[:min_length], sample_atoms[:min_length])
    super_imposer.apply(sample_model.get_atoms())

    # Print RMSD:
    print(super_imposer.rms)

if __name__ == "__main__":
    main()
