# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 10:37:40 2022

@author: joshua.l.lansford
"""
import pandas as pd
import numpy as np

df_explosives = pd.read_csv(r'../data/expl_vp_data.csv')
smiles_path = (r"../data/expl_name_2_smiles.csv")
df_smiles = pd.read_csv(smiles_path)
df = pd.merge(df_explosives, df_smiles,on='abbr', how='left')
for i in range(df.shape[0]):
    if df.iloc[i]['name_x'] != df.iloc[i]['name_y']:
        print(df.iloc[i])
        
df = pd.merge(df_explosives, df_smiles,on=['abbr', 'name'], how='left')

df['phase'] = 'Unknown'
for column in ['mp [C]', 'A [log10(atm)]', 'B [K]', 'C [K]'
               , 'Tmin [C]', 'Tmax [C]']:
    df[column] = df[column].astype(float)
df.loc[df['Tmin [C]'] >= df['mp [C]'] - 5
       , 'phase'] = 'liquid'
df.loc[df['Tmax [C]'] <= df['mp [C]'] + 5
       ,'phase'] = 'solid'

df['Tmid [C]'] = (df['Tmin [C]'] + df['Tmax [C]']) / 2

df['P@25 [log10(atm)]'] = (df['A [log10(atm)]'] - df['B [K]']/(298.15+ df['C [K]']))
df['P@100 [log10(atm)]'] = (df['A [log10(atm)]'] - df['B [K]']/(373.15+ df['C [K]']))
df['P@200 [log10(atm)]'] = (df['A [log10(atm)]'] - df['B [K]']/(473.15+ df['C [K]']))
df['P@500 [log10(atm)]'] = (df['A [log10(atm)]'] - df['B [K]']/(773.15+ df['C [K]']))
df['P@min [log10(atm)]'] = (df['A [log10(atm)]'] - df['B [K]']
                       / (df['Tmin [C]'] + 273.15+ df['C [K]']) )
df['P@max [log10(atm)]'] = (df['A [log10(atm)]'] - df['B [K]']
                       / (df['Tmax [C]'] + 273.15+ df['C [K]']) )

df['P@mid [log10(atm)]'] = (df['A [log10(atm)]'] - df['B [K]']
                       / (df['Tmid [C]'] + 273.15+ df['C [K]']) )



df['data_type'] = 'valid'
df['P25_MAD'] = float('nan')
df['P25_mean'] = float('nan')
df['P25_SD'] = float('nan')
df['P25_median'] = float('nan')
df['num_valid'] = int(1)
df['num_outliers'] = int(0)

for abbr in np.unique(df['abbr']):
    for phase in np.unique(df[df['abbr']==abbr]['phase']):
        selection = df[(df['abbr']==abbr) & (df['phase']==phase)]
        P25 = selection['P@25 [log10(atm)]']
        median = np.median(P25)
        error = np.abs(P25 - median)
        MAD = np.median(error)
        outliers = P25[(error > 4.5 * MAD) & (error > 0.17)].index #remove outside 1SD = 1.5 MAD
        if len(outliers) > 0:
            df.loc[outliers, 'data_type'] = 'outlier'
            df.loc[P25.index, 'num_outliers'] =  int(outliers.shape[0])
        selection = df[(df['abbr']==abbr) & (df['phase']==phase)]
        valid = selection[selection['data_type']=='valid'].index
        
        if P25[valid].shape[0] > 1:
            df.loc[P25.index, 'P25_SD'] =  P25[valid].std(ddof=1)
            df.loc[P25.index, 'P25_median'] = median
            df.loc[P25.index,'P25_MAD'] = MAD
            df.loc[P25.index, 'P25_mean'] =  P25[valid].mean()
            df.loc[P25.index, 'num_valid'] =  valid.shape[0]
        
for column in ['mp [C]', 'A [log10(atm)]', 'B [K]', 'C [K]', 'SE [atm]'
               , 'Tmin [C]', 'Tmax [C]', 'Tmid [C]'
               , 'P@25 [log10(atm)]', 'P@min [log10(atm)]'
               , 'P@max [log10(atm)]', 'P@mid [log10(atm)]'
               , 'P25_MAD', 'P25_mean', 'P25_SD', 'P25_median'
               , 'P@100 [log10(atm)]', 'P@200 [log10(atm)]', 'P@500 [log10(atm)]']:
    df[column] = df[column].astype(float)

df.to_csv('../data/expl_vp_data_w_pressures.csv', index=False)