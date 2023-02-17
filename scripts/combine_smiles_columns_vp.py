# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import os
from vp_model.utils import canonicalize
input_columns = ['NO', 'FORMULA', 'NAME', 'CAS No','A [log10(mmHg)]'
                  , 'B [C]', 'C [C]', 'TMIN [C]', 'TMAX [C]', 'code'
                  , 'SMILES_decode	', 'SMILES_CAS']

output_columns = ['smiles', 'code', 'A [log10(atm)]', 'B [C]', 'C [C]'
                  , 'TMIN [C]', 'TMAX [C]', 'FORMULA', 'NAME', 'CAS No']

df = pd.DataFrame()
file_names = ['Yaw_vp_o_w_SMILES.csv', 'Yaw_vp_io_w_SMILES.csv']
for file_name in file_names:
    file_loc = os.path.join('../data/', file_name)
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
df['A [log10(atm)]'] = df['A [log10(mmHg)]'] - np.log10(760)
df = df.iloc[good_idxs][output_columns]
df.to_csv('../data/Yaw_vp_w_SMILES.csv', index=False)
