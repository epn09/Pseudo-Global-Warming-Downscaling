#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:32:31 2020

@author: doan
"""


import pandas as pd
import os, sys

df = pd.read_csv('cmip6_list.csv',index_col=0)
for c in df.columns: 
    print(c)
    print(df[c].unique())
    do = pd.DataFrame(df[c].unique(),columns=[c] )
    odir = 'cmip6_info/'
    if not os.path.exists(odir): os.makedirs(odir)
    do.to_csv(odir+c+'.csv')
    
    
    
    
    
    
    
    
    
    
    
    
    
        