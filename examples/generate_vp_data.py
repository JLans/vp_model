# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:07:54 2021

@author: lansf
"""
from vp_model.convert_data import split_antoine_data
num_folds = 3
Yaw_liquid_path = '../data/Yaw_vp_o_data_liquids.csv'
save_dir = './splits_o_w_weights'
split_antoine_data(Yaw_liquid_path, save_dir, (0.8, 0.1, 0.1), num_folds, seed=1
                   , weights_path='../vp_model/data/sim_max_liquids_o.csv')
