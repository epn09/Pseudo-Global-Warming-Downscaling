#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # This is script for downloading CMIP6 GCM data using intake lib
#
#
# + Please install intake and intake-esm if not exit
#
#
# conda install -c conda-forge intake
#
# conda install -c conda-forge intake-esm
#
#


# %%
import intake

# %%
import sys
print(sys.executable)

# %%
import intake
import xarray as xr 
#import proplot as plot 
import matplotlib.pyplot as plt
import sys
import pandas as pd
import os
# necessary url
url = "https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json"
# open the catalog
dataframe = intake.open_esm_datastore(url)

# %%
dataframe.df.columns
df = dataframe.df

# %%
c = 'institution_id'
inss = pd.read_csv('cmip6_info/'+c+'.csv',index_col=0)[c].values
c = 'experiment_id'
exps = pd.read_csv('cmip6_info/'+c+'.csv',index_col=0)[c].values
c = 'variable_id'
varss = pd.read_csv('cmip6_info/'+c+'.csv',index_col=0)[c].values
c = 'member_id'
mems = pd.read_csv('cmip6_info/'+c+'.csv',index_col=0)[c].values

# %%

# %%

# %%
#for ins in ['MIROC', 'NCAR', 'MOHC',  'ECMWF'][:3]: #inss[:0]:
for ins in inss[:]:    
    for exp in ['historical','ssp585','ssp370','ssp245','ssp126'][: ]:
        for mem in ['r2i1p1f1' ]:
            
            for var in ['tas','ta','ua','va','hur','zg','ts'][:]:
                
                models = dataframe.search(experiment_id=exp,table_id='Amon',
                                          variable_id=var,
                                          institution_id=ins,
                                          member_id=mem)  
                
                
                print(var, exp, ins, mem, len(models.df))
                
                if len(models.df) > 0:
                    print('Download')
                    
                    if True:
                        
                        try:
                        
                            datasets = models.to_dataset_dict()
                            for k, v in datasets.items():
                                odir = 'Download_all/'+ins+'/'+exp+'/'
                                if not os.path.exists(odir): os.makedirs(odir)
                                ofile = odir + var + '_'+ k + '_'+mem+'.nc'
                                print('write to ',ofile)
                                v.to_netcdf(ofile)
                        except:
                            print('fail')

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
sys.exit()
models = dataframe.search(experiment_id='historical',
                              table_id='Amon',
                              variable_id='tas',
                              institution_id='NCAR',
                              member_id='r11i1p1f1')


# %%
datasets = models.to_dataset_dict()
dset = datasets['CMIP.NCAR.CESM2.historical.Amon.gn']

# %%
fig, ax = plot.subplots(axwidth=4.5, tight=True,
                        proj='robin', proj_kw={'lon_0': 180},)
# format options
ax.format(land=False, coast=True, innerborders=True, borders=True,
          labels=True, geogridlinewidth=0,)
map1 = ax.contourf(dset['lon'], dset['lat'], dset['tas'][0,0,:,:],
                   cmap='IceFire', extend='both')
ax.colorbar(map1, loc='b', shrink=0.5, extendrect=True)
plt.show()

