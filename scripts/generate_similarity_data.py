# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 10:37:40 2022

@author: joshua.l.lansford
"""

import pandas as pd
import numpy as np
from vp_model import utils
import multiprocessing as mp
if __name__ == '__main__':
    num_cpus = 6
    vp_data = pd.read_csv('../data/Yaw_vp_o_data_liquids.csv')
    vp_data_explosives = pd.read_csv('../data/expl_name_2_smiles.csv')
    
    expl_smiles, indices = np.unique(utils.canonicalize(
                                     vp_data_explosives['smiles'])[0]
                            , return_index=True)
    
    similarities = np.zeros((expl_smiles.shape[0], vp_data.shape[0]))
    #for count_1, smile_1 in enumerate(expl_smiles):
    #    print(count_1)
    #    for count_2, smile_2 in enumerate(vp_data['smiles']):
    #        similarities[count_1][count_2] = utils.get_similarity(smile_1, smile_2)
            
    for count_1, smile_1 in enumerate(expl_smiles):
            print(count_1)
            input_smiles = [(smile_1, smile_2) for smile_2 in vp_data['smiles']]
            pool = mp.Pool(num_cpus)
            values = pool.starmap(utils.get_similarity, input_smiles)
            pool.close()
            pool.join()
            similarities[count_1] = values
       
    self_sim = np.zeros((expl_smiles.shape[0], expl_smiles.shape[0]-1))
    for count_1, smile_1 in enumerate(expl_smiles):
        index_shift = 0
        for count_2, smile_2 in enumerate(expl_smiles):
            if count_1 != count_2:
                self_sim[count_1][count_2+index_shift] = utils.get_similarity(smile_1, smile_2)
            else:
                index_shift = -1
            
       
    df_similarities = pd.DataFrame()
    df_similarities['abbr'] = vp_data_explosives.iloc[indices]['abbr'].to_list()
    df_similarities['name'] = vp_data_explosives.iloc[indices]['name'].to_list()
    df_similarities['formula'] = vp_data_explosives.iloc[indices]['formula'].to_list()
    df_similarities['CAS'] = vp_data_explosives.iloc[indices]['CAS'].to_list()
    df_similarities['IUPAC'] = vp_data_explosives.iloc[indices]['IUPAC'].to_list()
    df_similarities['smiles'] = expl_smiles
    df_similarities['sim_avg'] = similarities.mean(axis=1)
    df_similarities['sim_max'] = similarities.max(axis=1)
    df_similarities['sim_min'] = similarities.min(axis=1)
    df_similarities['sim_std'] = similarities.std(axis=1, ddof=1)
    df_similarities['self_avg'] = self_sim.mean(axis=1)
    df_similarities['self_max'] = self_sim.max(axis=1)
    df_similarities['self_min'] = self_sim.min(axis=1)
    df_similarities['self_std'] = self_sim.std(axis=1, ddof=1)
    df_similarities.to_csv('../data/expl_similarities_liquids_o.csv', index=False)
    
    df_sim_max = pd.DataFrame()
    df_sim_max['weights'] = similarities.max(axis=0)
    df_sim_max.to_csv('../data/sim_max_liquids_o.csv', index=False)


