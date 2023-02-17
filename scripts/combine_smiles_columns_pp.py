# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import os
from vp_model.utils import canonicalize
output_columns = ['smiles', 'mp [K]', 'mp_code', 'bp [K]', 'bp_code'
                  , 'g/mol', 'density_T [C]', 'density [g/cm3]', 'density_code'
                  , 'nD_T [C]', 'nD', 'nD_code'
                  ,'FORMULA', 'NAME', 'CAS No']
#'HTTPError', 'nan'
df = pd.DataFrame()
for num in np.linspace(0,60000, num=7, endpoint=True, dtype=str):
    file_loc = os.path.join('../data/','Yaw_pp_data_' + num[:-2]+'.csv')
    df = df.append(pd.read_csv(file_loc), ignore_index=False)
exclude = ['HTTPError', 'nan', 'invalid']
smiles_decode, invalid = canonicalize(df['SMILES_decode'].astype('str'), exclude = exclude)

smiles_cas, invalid = canonicalize(df['SMILES_CAS'].astype('str'), exclude = exclude)
good_idxs = []
smiles = []
bad_idxs = []
for count in range(df.shape[0]):
    if smiles_cas[count] not in exclude:
        smiles.append(smiles_cas[count])
    else:
        smiles.append(smiles_decode[count])
    if smiles[count] not in exclude:
        good_idxs.append(count)
    else:
        bad_idxs.append(count)
df['smiles'] = smiles
df = df.iloc[good_idxs][output_columns]

df.to_csv('../data/Yaw_pp_data_wS_MILES.csv', index=False)
