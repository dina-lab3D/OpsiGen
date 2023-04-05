#ifndef MOLECULARINTERFACE_H
#define MOLECULARINTERFACE_H

#include <fstream>
#include <HashInterface.h>
#include <ChemMolecule.h>

using std::ifstream;
 
/*
  CLASS

  KEYWORDS

  AUTHORS
  Dina (mailto: duhovka@tau.ac.il)

  Copyright: SAMBA group, Tel-Aviv Univ. Israel, 2004.

  OVERVIEW TEXT

*/

class MolecularInterface {
public:
  MolecularInterface(const ChemMolecule& mol1, const ChemMolecule& mol2, float thr) :
    mol1_(mol1), mol2_(mol2), thr_(thr) {
    buildResidueInterfaces();
  }

  // output
  void outputInterfaceResidues(std::ostream& outStream);
  void outputResiduesContacts(std::ostream& outStream);
  void outputAtomContacts(std::ostream& outStream);
  void outputCAContacts(std::ostream& outStream);
  void outputResiduesRasmol(std::ostream& rasmolFile);
  void outputSequenceForSqwrl(std::ostream& outStream);
  void outputAtoms(std::ostream& outStream);

  void outputInterfaceResidues(ChemMolecule& mol) const;

protected:
  void createAtom2ResidueMapping(const ChemMolecule& mol, std::vector<unsigned int>& mapping);
  void buildResidueInterfaces();
  void buildAtomInterfaces();

  // output
  void outputInterfaceResidues(const ChemMolecule& mol, const Interface& residueInterface, std::ostream& outStream);

  void outputResidueRasmol(unsigned int resIndex, char chainId, std::ostream& outStream);
  void outputRasmol(const ChemMolecule& mol, const Interface& interface, std::ostream& outStream);
  void outputSequenceForSqwrl(const ChemMolecule& mol, const Interface& residueInterface, std::ostream& outStream);


protected:
  const ChemMolecule& mol1_;
  const ChemMolecule& mol2_;

  Interface mol1AtomInterface_, mol2AtomInterface_;
  Interface mol1ResidueInterface_, mol2ResidueInterface_;

  float thr_;
};

#endif //MOLECULARINTERFACE_H
