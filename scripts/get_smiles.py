# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import urllib.request
import pandas as pd
import json

df = pd.read_csv('../data/Yaw_vp_io_data.csv')
SMILES_decode = []
url_1 = 'https://cactus.nci.nih.gov/chemical/structure/'
url_2 = '/SMILES'
for count, name in enumerate(df['NAME']):
    url = url_1 + name.replace(' ','%20') + url_2
    try:
        uf = urllib.request.urlopen(url)
        html = uf.read()
        SMILES_decode.append(html.decode())
    except:
        SMILES_decode.append('HTTPError')
        print('error')
    print(count)
df['SMILES_decode'] = SMILES_decode

SMILES_CAS = []
for count, cas_no in enumerate(df['CAS No']):
    if cas_no != '---':
        url = 'https://commonchemistry.cas.org/api/detail?cas_rn=' + cas_no
        try:
            uf = urllib.request.urlopen(url)
            html = uf.read()
            data = json.loads(html.decode())
            SMILES_CAS.append(data['canonicalSmile'])
        except:
            SMILES_CAS.append('HTTPError')
            print('error')
    else:
        SMILES_CAS.append('nan')
    print(count)

df['SMILES_CAS'] = SMILES_CAS
df.to_csv('../data/Yaw_vp_data_io_w_SMILES.csv', index=False)