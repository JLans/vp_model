# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:25:13 2022

@author: joshua.l.lansford
"""
import pandas as pd
from vp_model import utils
from vp_model.molecule_objects import Simple_Molecule
import urllib.request
import json
end_str = ['0', '10000', '20000', '30000', '40000', '50000', '60000']
Yaw_data = pd.read_csv('../../data/Yaw_pp_data_'+end_str[0]+'.csv')
for end in end_str[1:]:
    Yaw_data = Yaw_data.append(pd.read_csv('../../data/Yaw_pp_data_'+end+'.csv'))
Yaw_cas = Yaw_data[(Yaw_data['SMILES_CAS'].isna()==False) 
                   & (Yaw_data['SMILES_CAS'] != 'HTTPError')]

valid, invalid = utils.canonicalize(Yaw_cas['SMILES_CAS'].tolist(), exclude=[None])
valid_index = [i for i in range(Yaw_cas.shape[0]) if i not in invalid]
Yaw_invalid = Yaw_cas.iloc[invalid]
Yaw_mis_match = utils.get_df_formula_match(Yaw_cas.iloc[valid_index]
                                           , 'SMILES_CAS', 'FORMULA', match=False)
non_ionic = [i for i in range(Yaw_mis_match.shape[0]) if '.' not in Yaw_mis_match['SMILES_CAS'].iloc[i]]

FORMULA_CAS = []
for count, cas_no in enumerate(Yaw_mis_match.iloc[non_ionic]['CAS No']):
    url = 'https://commonchemistry.cas.org/api/detail?cas_rn=' + cas_no
    try:
        uf = urllib.request.urlopen(url)
        html = uf.read()
        data = json.loads(html.decode())
        FORMULA_CAS.append(data['molecularFormula'])
    except:
        FORMULA_CAS.append('HTTPError')
        print('error')
    print(count)

formulas = []
for count in range(len(FORMULA_CAS)):
    formula = FORMULA_CAS[count].replace('<sub>','')
    formulas.append(formula.replace('</sub>', ''))
    
Yaw_non_ionic = Yaw_mis_match.iloc[non_ionic]
Yaw_non_ionic['formula'] = formulas
ni_mis_match = utils.get_df_formula_match(Yaw_non_ionic
                                           , 'SMILES_CAS', 'formula', match=False)