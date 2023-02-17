import numpy as np
from rdkit.Chem import MolFromInchi
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import AddHs
class Simple_Molecule:
    """Class for loading an manipulating a dataset"""
    def __init__(self, mol_str, str_type='smiles'):
        """ 
        Parameters
        ----------
        mol_string : str
            smiles string.
        """
        if str_type.lower() == 'smiles':
            if mol_str[0:2] == "['" and mol_str[-2:] == "']":
                mol_str = mol_str[2:-2]
            self.molecule = AddHs(MolFromSmiles(mol_str))
        elif str_type.lower() == 'inchi':
            self.molecule = AddHs(MolFromInchi(mol_str))
        elif str_type.lower() == 'formula':
            self.molecule = None
        self.str_type = str_type.lower()
        self.mol_str = mol_str
        self._atoms = None
        self._atomic_numbers = None
        self._bonds = None
        self._num_atoms = None
        self._num_bonds = None
        self._get_molecule()
    
    def _get_molecule(self):
        atoms = []
        if self.str_type == 'formula':
            num_atoms = []
            for count, char in enumerate(self.mol_str):
                if char.isupper():
                    atoms.append(char)
                    num_atoms.append('1')
                elif char.islower():
                    atoms[-1] += char
                elif char.isdecimal():
                    if self.mol_str[count-1].isdecimal():
                        num_atoms[-1] += char
                    else:
                        num_atoms[-1] = char
            self._atoms = np.array(atoms).astype('<U2')
            self._num_atoms = np.array([int(char) for char in num_atoms]).astype(int)
        else:
            for atom in self.molecule.GetAtoms():
                atoms.append([atom.GetSymbol(), atom.GetAtomicNum()])
            atoms = np.array(atoms)
            atoms, num_atoms = np.unique(atoms, return_counts=True, axis=0)
            atomic_numbers = atoms[:,1].astype(int)
            index_sort = np.argsort(atomic_numbers)
            self._atoms = atoms[:,0].astype('<U2')[index_sort]
            self._atomic_numbers = atoms[:,1].astype(int)[index_sort]
            self._num_atoms = num_atoms[index_sort]
        
    def get_atoms(self, atom_type='all'):
        if atom_type == 'all':
            return self._atoms
        elif atom_type == 'no_H':
            return np.array([atom for atom in self._atoms if atom not in ['H', 'D', 'T']])
    
    def get_atomic_numbers(self):
        return self._atomic_numbers
    
    def get_num_atoms(self, atom_type='all'):
        if atom_type == 'all':
            return self._num_atoms
        elif atom_type == 'sum':
            return self._num_atoms.sum()
        elif atom_type == 'heavy':
            total = 0
            for atom in self.get_atoms():
                if atom not in ['H', 'D', 'T']:
                    total += self._num_atoms[list(self._atoms).index(atom)]
            return total
        elif atom_type == 'no_H':
            num_atoms = []
            for count, atom in enumerate(self.get_atoms()):
                if atom not in ['H', 'D', 'T']:
                    num_atoms.append(self._num_atoms[count])
            return np.array(num_atoms)
        else:
            if atom_type in self.get_atoms():
                return self._num_atoms[list(self._atoms).index(atom_type)]
            else:
                return 0