#include "MolecularInterface.h"

void MolecularInterface::createAtom2ResidueMapping(const ChemMolecule& mol, std::vector<unsigned int>& mapping) {
  mapping.clear();
  mapping.reserve(mol.size());
  for(unsigned int i=0; i<mol.size(); i++) {
    mapping.push_back(mol.residueEntry(mol[i].chainId(), mol[i].residueSequenceID()));
  }
}

void MolecularInterface::buildAtomInterfaces() {
  HashInterface<ChemMolecule> hashInterface(mol1_, mol2_);
  hashInterface.buildInterface(mol1AtomInterface_, mol2AtomInterface_, thr_);
}

void MolecularInterface::buildResidueInterfaces() {
  if(mol1AtomInterface_.size() == 0) {
    buildAtomInterfaces();
  }

  std::vector<unsigned int> mol1Map, mol2Map;
  createAtom2ResidueMapping(mol1_, mol1Map);
  createAtom2ResidueMapping(mol2_, mol2Map);
  mol1AtomInterface_.buildHigherLevel(mol1ResidueInterface_, mol1Map, mol2Map);
  mol2AtomInterface_.buildHigherLevel(mol2ResidueInterface_, mol2Map, mol1Map);

  std::cout << "Molecule 1 interface: size =  "
       << mol1AtomInterface_.size() << " atoms "
       << mol1ResidueInterface_.size() << " residues" << std::endl;
  std::cout << "Molecule 2 interface: size =  "
       << mol2AtomInterface_.size() << " atoms "
       << mol2ResidueInterface_.size() << " residues"  << std::endl;
}

void MolecularInterface::outputResidueRasmol(unsigned int resIndex, char chainId, std::ostream& outStream) {
  if(chainId == ' ') {
    outStream << resIndex;
  } else {
    outStream << "(*" << chainId << " and " << resIndex << ")";
  }
}

void MolecularInterface::outputRasmol(const ChemMolecule& mol, const Interface& interface, std::ostream& outStream) {
  outStream << "select ";
  bool first = true;
  for(unsigned int resIndex=0; resIndex<mol.numberOfRes(); resIndex++) {
    if(interface.isInterface(resIndex)) {
      int atomIndex = mol.getFirstAtomEntryForResEntry(resIndex);
      if(resIndex > 0 && resIndex%10 == 0) {
	outStream << std::endl << "select selected or ";
      } else {
	if(! first)
	  outStream << ',';
      }
      outputResidueRasmol(mol[atomIndex].residueIndex(), mol[atomIndex].chainId(), outStream);
      first = false;
    }
  }
  outStream << std::endl;
}

void MolecularInterface::outputInterfaceResidues(const ChemMolecule& mol,
						 const Interface& residueInterface,
						 std::ostream& outStream) {
  for(unsigned int resIndex=0; resIndex<mol.numberOfRes(); resIndex++) {
    if(residueInterface.isInterface(resIndex)) {
      int atomIndex = mol.getFirstAtomEntryForResEntry(resIndex);
      outStream <<  mol[atomIndex].residueSequenceID() << ' ' << mol[atomIndex].chainId() << std::endl;
    }
  }
}

void MolecularInterface::outputInterfaceResidues(std::ostream& outStream) {
  outputInterfaceResidues(mol1_, mol1ResidueInterface_, outStream);
  outputInterfaceResidues(mol2_, mol2ResidueInterface_, outStream);
}

void MolecularInterface::outputResiduesContacts(std::ostream& outStream) {
  std::cout << "Interface: number of contacts = "
       << mol1ResidueInterface_.adjacenciesNumber()
       << " residue-residue pairs" << std::endl;

  for(Interface::iterator it = mol1ResidueInterface_.begin();
      it != mol1ResidueInterface_.end(); ++it) {
    const Interface::Adjacency& adj = *it;
    int mol1AtomIndex = mol1_.getFirstAtomEntryForResEntry(adj.first());
    int mol2AtomIndex = mol2_.getFirstAtomEntryForResEntry(adj.second());
    outStream << mol1_[mol1AtomIndex].residueSequenceID() << ' ' << mol1_[mol1AtomIndex].chainId() << " - "
	      << mol2_[mol2AtomIndex].residueSequenceID() << ' ' << mol2_[mol2AtomIndex].chainId() << std::endl;
  }
}

void MolecularInterface::outputResiduesRasmol(std::ostream& rasmolFile) {
  outputRasmol(mol1_, mol1ResidueInterface_, rasmolFile);
  rasmolFile << "spacefill" << std::endl;
  outputRasmol(mol2_, mol2ResidueInterface_, rasmolFile);
  rasmolFile << "spacefill" << std::endl;
}

void MolecularInterface::outputAtomContacts(std::ostream& outStream) {
  std::cout << "Interface: number of atom-atom contacts = "
       << mol1AtomInterface_.adjacenciesNumber() << std::endl;
  for(Interface::iterator it = mol1AtomInterface_.begin();
      it != mol1AtomInterface_.end(); ++it) {
    const Interface::Adjacency& adj = *it;
    int mol1AtomIndex = adj.first();
    int mol2AtomIndex = adj.second();
    outStream << mol1_[mol1AtomIndex].residueSequenceID() << ' '
	      << mol1_[mol1AtomIndex].chainId() << ' '
	      << mol1_[mol1AtomIndex].type() << " - "
	      << mol2_[mol2AtomIndex].residueSequenceID() << ' '
	      << mol2_[mol2AtomIndex].chainId()  << ' '
	      << mol2_[mol2AtomIndex].type() << std::endl;
  }
}

void MolecularInterface::outputAtoms(std::ostream& outStream) {
  for(Interface::iterator it = mol1AtomInterface_.begin();
      it != mol1AtomInterface_.end(); ++it) {
    const Interface::Adjacency& adj = *it;
    int mol1AtomIndex = adj.first();
    outStream << mol1_[mol1AtomIndex] << std::endl;
  }
  for(Interface::iterator it = mol2AtomInterface_.begin();
      it != mol2AtomInterface_.end(); ++it) {
    const Interface::Adjacency& adj = *it;
    int mol2AtomIndex = adj.first();
    outStream << mol2_[mol2AtomIndex] << std::endl;
  }
}

void MolecularInterface::outputCAContacts(std::ostream& outStream) {
  std::cout << "Interface: number of atom-atom contacts = "
       << mol1AtomInterface_.adjacenciesNumber() << std::endl;
  for(Interface::iterator it = mol1AtomInterface_.begin();
      it != mol1AtomInterface_.end(); ++it) {
    const Interface::Adjacency& adj = *it;
    int mol1AtomIndex = adj.first();
    int mol2AtomIndex = adj.second();
    if(mol1_[mol1AtomIndex].isCA() && mol2_[mol2AtomIndex].isCA()) {
      outStream << mol1_[mol1AtomIndex].residueSequenceID() << ' '
		<< mol1_[mol1AtomIndex].chainId() << ' '
		<< mol1_[mol1AtomIndex].type() << " - "
		<< mol2_[mol2AtomIndex].residueSequenceID() << ' '
		<< mol2_[mol2AtomIndex].chainId()  << ' '
		<< mol2_[mol2AtomIndex].type() << std::endl;
    }
  }
}

void MolecularInterface::outputSequenceForSqwrl(const ChemMolecule& mol,
						const Interface& residueInterface,
						std::ostream& outStream) {
  for(unsigned int resIndex=0; resIndex<mol.numberOfRes(); resIndex++) {
    int i = mol.getFirstAtomEntryForResEntry(resIndex);
    if(mol[i].isCA()) {
      if(residueInterface.isInterface(resIndex)) {
	outStream << (char)toupper(mol[i].residueType());
      } else {
	outStream << (char)tolower(mol[i].residueType());
      }
    }
  }
}

void MolecularInterface::outputSequenceForSqwrl(std::ostream& outStream) {
  outputSequenceForSqwrl(mol1_, mol1ResidueInterface_, outStream);
  outputSequenceForSqwrl(mol2_, mol2ResidueInterface_, outStream);
  outStream << std::endl;
}



void MolecularInterface::outputInterfaceResidues(ChemMolecule& mol) const {
  for(unsigned int i=0; i<mol1_.size(); i++) {
    if(mol1ResidueInterface_.isInterface(mol1_.residueEntry(mol1_[i].chainId(),
                                                            mol1_[i].residueSequenceID()))) {
      mol.push_back(mol1_[i]);
    }
  }

  for(unsigned int i=0; i<mol2_.size(); i++) {
    if(mol2ResidueInterface_.isInterface(mol2_.residueEntry(mol2_[i].chainId(),
                                                            mol2_[i].residueSequenceID()))) {
      mol.push_back(mol2_[i]);
    }
  }
}
