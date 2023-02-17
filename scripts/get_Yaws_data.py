# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 10:13:49 2021

@author: joshua.l.lansford
"""
import pdfplumber
import pandas as pd
import numpy as np

Yaws = (r"../data/Yaws handbook chapter 1 - organic VP.pdf")
pdf = pdfplumber.open(Yaws)
table=[]
for i in range(313):
    p = pdf.pages[i+1]
    t = p.find_tables({'vertical_strategy': 'lines'
                           , 'horizontal_strategy': 'lines'
                           ,'keep_blank_chars': True})[0]
    table_i = t.extract(x_tolerance=2)
    table += table_i

header = table[1]
table_filtered = []
for value in table:
    if value[-1] not in (None, 'code'):
        table_filtered.append(value)
        
Yaw_vp = pd.DataFrame(table_filtered
                        , columns = header)

Yaw_vp.to_csv('../data/Yaw_vp_o_data.csv', index=False)

Yaws = (r"../data/Yaws handbook chapter 2 - inorganic VP.pdf")
pdf = pdfplumber.open(Yaws)
table=[]
for i in range(7):
    p = pdf.pages[i+1]
    t = p.find_tables({'vertical_strategy': 'lines'
                           , 'horizontal_strategy': 'lines'
                           ,'keep_blank_chars': True})[0]
    table_i = t.extract(x_tolerance=2)
    table += table_i

header = table[1]
table_filtered = []
for value in table:
    if value[-1] not in (None, 'code'):
        table_filtered.append(value)
        
Yaw_vp = pd.DataFrame(table_filtered
                        , columns = header)

Yaw_vp.to_csv('../data/Yaw_vp_io_data.csv', index=False)

Yaws_pp = (r"../data/Yaws handbook - organic physical properties.pdf")
pdf = pdfplumber.open(Yaws_pp)
table=[]
for i in range(681):
    p = pdf.pages[i+2]
    t = p.find_tables({'vertical_strategy': 'lines'
                           , 'horizontal_strategy': 'lines'
                           ,'keep_blank_chars': True})[0]
    table_i = t.extract(x_tolerance=2)
    for row in table_i:
        if row[2] not in (None, 'NAME'):
            table.append(row)
Yaws_pp = (r"../data/Yaws handbook - inorganic physical properties.pdf")
pdf = pdfplumber.open(Yaws_pp)
for i in range(126):
    p = pdf.pages[i+1]
    t = p.find_tables({'vertical_strategy': 'lines'
                           , 'horizontal_strategy': 'lines'
                           ,'keep_blank_chars': True})[0]
    table_i = t.extract(x_tolerance=2)
    for row in table_i:
        if row[2] not in (None, 'NAME'):
            table.append(row)

Yaw_pp = pd.DataFrame(table
                        , columns = ['NO', 'FORMULA', 'NAME', 'CAS No', 'g/mol'
                                     , 'mp [K]', 'mp_code', 'bp [K]', 'bp_code'
                                     , 'density_T [C]', 'density [g/cm3]'
                                     , 'density_code', 'nD_T [C]', 'nD'
                                     , 'nD_code'])
Yaw_pp.to_csv(r'../data/Yaw_pp_data.csv', index=False)