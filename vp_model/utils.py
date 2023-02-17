# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""
from rdkit.Chem import AllChem
import os
import pandas as pd
from numpy.random import default_rng
import numpy as np
from vp_model.molecule_objects import Simple_Molecule

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def get_df_w_number(dataframe, column):
    dataframe = dataframe[dataframe[column].apply(lambda x: is_number(x))].copy()
    return dataframe
    
def get_df_unique(dataframe, column):
    unique, unique_indices = np.unique(dataframe[column], return_index=True)
    unique_indices.sort()
    dataframe = dataframe.iloc[unique_indices]
    return dataframe

def get_df_filtered(dataframe, column, in_list, is_in=False):
    if type(in_list) != list:
        in_list = in_list.to_list()
    def in_func(x):
        if is_in == True:
            return x in in_list
        else:
            return x not in in_list
    dataframe = dataframe[dataframe[column].apply(lambda x: in_func(x))].copy()
    return dataframe

def get_df_large_mols(dataframe, column, min_atoms=2, min_heavy_atoms=1):
    def func(x):
        mol = AllChem.MolFromSmiles(x)
        mol = AllChem.AddHs(mol)
        is_true = (mol.GetNumAtoms() >= min_atoms
                   and mol.GetNumHeavyAtoms() >= min_heavy_atoms)
        return is_true
    dataframe = dataframe[dataframe[column].apply(lambda x: func(x))].copy()
    return dataframe

def get_df_formula_match(dataframe, smiles_column, formula_column, match=True):
    mols_from_smiles = [Simple_Molecule(smile, str_type='smiles')
                   for smile in dataframe[smiles_column]]
    mols_from_formula = [Simple_Molecule(smile, str_type='formula')
                       for smile in dataframe[formula_column]]
    match_list = []
    for count in range(len(mols_from_smiles)):
        atoms_1 = mols_from_smiles[count].get_atoms('no_H')
        argsort_1 = atoms_1.argsort()
        atoms_1 = atoms_1[argsort_1]
        atoms_2 = mols_from_formula[count].get_atoms('no_H')
        argsort_2 = atoms_2.argsort()
        atoms_2 = atoms_2[argsort_2]
        num_atoms_1 = mols_from_smiles[count].get_num_atoms('no_H')[argsort_1]
        num_atoms_2 = mols_from_formula[count].get_num_atoms('no_H')[argsort_2]
        if (match == True and tuple(atoms_1) == tuple(atoms_2)
            and tuple(num_atoms_1) == tuple(num_atoms_2)):
            match_list.append(count)
        elif (match == False and tuple(atoms_1) != tuple(atoms_2)
            or tuple(num_atoms_1) != tuple(num_atoms_2)):
            match_list.append(count)
    dataframe = dataframe.iloc[match_list].copy()
    return dataframe

def get_similarity(smile_1, smile_2):
        mol1 = AllChem.MolFromSmiles(smile_1)
        mol2 = AllChem.MolFromSmiles(smile_2)
        radius = 4
        fp_1 = AllChem.GetMorganFingerprint(mol1, radius)
        fp_2 = AllChem.GetMorganFingerprint(mol2, radius)
        similarity = AllChem.DataStructs.TanimotoSimilarity(fp_1, fp_2)
        return similarity
    
def canonicalize(smiles, exclude=[None]):
    new_smiles = []
    invalid = []
    for index, smile in enumerate(smiles):
        if smile not in exclude:
            try:
                mol = AllChem.MolFromSmiles(smile)
                new_smiles.append(AllChem.MolToSmiles(mol))
            except:
                new_smiles.append('invalid')
                invalid.append(index)
        else:
            new_smiles.append(smile)
    return new_smiles, invalid

def split_data(input_path, save_dir, split_tuple, num_folds=1, seed=1
               , weights_path=None):
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    data = pd.read_csv(input_path)
    seed_list = [seed + i for i in range(num_folds)]
    rng = default_rng(seed_list[0])
    indices = list(range(data.shape[0]))
    rng.shuffle(indices)
    train_size = int(split_tuple[0] * data.shape[0])
    train_val_size = int((split_tuple[0] + split_tuple[1]) * data.shape[0])
    test_data = data.iloc[indices[train_val_size:]]
    test_full = os.path.join(save_dir, 'test_full.csv')
    test_data.to_csv(test_full, index=False)
    data = data.iloc[indices[0:train_val_size]]
    if weights_path is not None:
        data_weights = pd.read_csv(weights_path)
        data_weights = data_weights.iloc[indices[0:train_val_size]]
    train_full = []
    train_weights = []
    val_full = []
    fold_dirs = []
    for i in range(num_folds):
        fold_dirs.append(os.path.join(save_dir,'fold_'+str(i)))
        if os.path.exists(fold_dirs[i]) == False:
            os.mkdir(fold_dirs[i])
        rng = default_rng(seed_list[i])
        indices = list(range(data.shape[0]))
        rng.shuffle(indices)
        train = data.iloc[indices[0:train_size]]
        if weights_path is not None:
            weights_train = data_weights.iloc[indices[0:train_size]]
            train_weights.append(os.path.join(fold_dirs[i],'train_weights.csv'))
            weights_train.to_csv(train_weights[-1], index=False)    
        val = data.iloc[indices[train_size:]]
        train_full.append(os.path.join(fold_dirs[i],'train_full.csv'))
        val_full.append(os.path.join(fold_dirs[i],'val_full.csv'))
        train.to_csv(train_full[-1], index=False)
        val.to_csv(val_full[-1], index=False)
    
    if weights_path is None:
        return (train_full, val_full, test_full, fold_dirs)
    else:
        return (train_full, val_full, test_full, fold_dirs, train_weights)

def get_duplicates(smiles):
    indices = []
    duplicates=[]
    for i in range(len(smiles)):
        if smiles[i] in smiles[0:i] or smiles[i] in smiles[i+1:]:
            indices.append(i),
            duplicates.append(smiles[i])
    return indices, duplicates

class words_to_int():
    """A :class:`words_to_int` is a class for converting words to integers."""

    def __init__(self, words):
        """
        
        Parameters
        ----------
        words : List[str]
            List of words.

        """
        self._word_dict = self._generate_dictionary(words)
        
    def _generate_dictionary(self, words):
        _word_dict = dict()
        for word in words:
            if word not in _word_dict.keys():
                _word_dict.update({word:len(_word_dict)})
        return _word_dict
    
    def add_words(self, words):
        for word in words:
            if word not in self._word_dict.keys():
                self._word_dict.update({word:len(self._word_dict)})
                
    def get_int(self, word):
        return self._word_dict[word]