# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# conda install -c conda-forge xarray dask netCDF4 bottleneck

# %%
import xarray as xr
import numpy as np
import os, glob, sys
#import matplotlib.pyplot as plt
import datetime
import numpy.ma as ma
import pandas as pd
from collections import OrderedDict 

# %%
def fstr(Base_clim, Futr_clim): 
    tbs, tbe = Base_clim 
    tts, tte = Futr_clim       
    return str(tts)+'-'+str(tte)+'_'+str(tbs)+'-'+str(tbe)

# %%
# ls ../00_Download_CMIP6/Download_s8/EC-Earth3/


# %%
var = ['ts', 'ta', 'hur', 'ua', 'va', 'zg']

Base_clim = (1995,2014)
Base_clim = (2000, 2019)
Futr_clim = (2080, 2099)
Futr_clim = (2040, 2059)

#anom = cal_anomaly(Base_clim, Futr_clim, rcp, g, var, input_dir, output_dir)
#g = 'MIROC'
#rcp = 'ssp585'

output_dir = 'diff/'
input_dir = '../00_Download_CMIP6/Download/'


gcms = [ g.split('/')[-1] for g in sorted(glob.glob('Download_all/*'))]

exps = ['historical','ssp585','ssp370','ssp245','ssp126']

varn = ['ta','ua','va','hur','zg','ts']

# %%



g, rcp = 'EC-Earth3', 'ssp585'

anom = OrderedDict()

for v in varn[:]:
    tbs, tbe = Base_clim 
    tts, tte = Futr_clim   
    
    odir = output_dir  + '/diff/' +rcp+'/' +fstr(Base_clim, Futr_clim)+'/'+v+'/'
    ofile = v+'_'+g+'.nc' 
    if not os.path.exists(odir): os.makedirs(odir)

    hfile = glob.glob(input_dir+ '/'+g+ '/historical/'+v+'_*'+g+'*')[0]
    rfile = glob.glob(input_dir+'/'+ g + '/'+ rcp+'/'+v+'_*'+g+'*')[0]


    print(g, rcp, v)

    #xr.coding.times._STANDARD_CALENDARS.add("Gregorian")
    #xr.coding.times._decode_datetime_with_pandas
    
    
    rd = xr.open_dataset(rfile)[v] #.isel(member_id=0) # rcp data
    hd = xr.open_dataset(hfile)[v] #.isel(member_id=0) # historical data
    
    #dims = rd.dims
    #dimsx = ('time', 'plev', 'lat', 'lon')
    #diff_dims = np.setdiff1d(dims, dimsx)
    
    hr_base1 = hd.loc[str(Base_clim[0]):str(Base_clim[1])]
    hr_base2 = rd.loc[str(Base_clim[0]):str(Base_clim[1])]
    
    hr_base1 = hd.sel(time = slice(str(Base_clim[0]), str(Base_clim[1])))
    hr_base2 = rd.sel(time = slice(str(Base_clim[0]), str(Base_clim[1])))
    
    if not type(hr_base2.time.values[0]) == np.datetime64:
        xx  = [ datetime.datetime.strptime(str(t),'%Y-%m-%d %H:%M:%S') for t in hr_base2.time.values]
        hr_base2.time.values[:] =xx
    
    hr_base = xr.concat([hr_base1, hr_base2], dim='time')
    if hr_base.time.size != (Base_clim[1] - Base_clim[0] + 1)*12: print('problem')

    
    hr_targ = rd.sel(time = slice(str(Futr_clim[0]), str(Futr_clim[1])))
    
    bmean = hr_base.groupby('time.month').mean('time')
    tmean = hr_targ.groupby('time.month').mean('time')
        
    diff = tmean.copy()
    diff[:] = tmean.values[:] - bmean.values[:]
    # Assuming you have an xarray dataset called 'ds'
    excluded_dims = ['month', 'lat', 'lon', 'plev']
    # Identify all the dimensions in the dataset excluding the specified ones
    dims_to_select = [dim for dim in diff.dims if dim not in excluded_dims]
    # Select the first point along each of these dimensions
    selection = {dim: 0 for dim in dims_to_select}
    # Apply the isel method
    selected_data = diff.isel(selection)
    selected_data.to_netcdf(odir+ofile)
    
    #hr_base = hr_targ = hr = diff = rd = hd = None
    #anom[v] = diff            
   

















sys.exit()
for g in [ 'EC-Earth3', 'MIROC6',  'MRI-ESM2-0', 'ACCESS-CM2', 'IPSL-CM6A-LR', 'MPI-ESM1-2-HR'][:1]:
    for rcp in ['ssp585', 'ssp126'][:1]:

        anom = OrderedDict()
        err = False            

        for v in varn[:]:
            try:
                if err: continue
                tbs, tbe = Base_clim 
                tts, tte = Futr_clim   
                
                odir = output_dir  + '/diff/' +rcp+'/' +fstr(Base_clim, Futr_clim)+'/'+v+'/'
                ofile = v+'_'+g+'.nc' 
                if not os.path.exists(odir): os.makedirs(odir)
            
                hfile = glob.glob(input_dir+ '/'+g+ '/historical/'+v+'_*'+g+'*')[0]
                rfile = glob.glob(input_dir+'/'+ g + '/'+ rcp+'/'+v+'_*'+g+'*')[0]
    
    
                print(g, rcp, v)
    
                #xr.coding.times._STANDARD_CALENDARS.add("Gregorian")
                #xr.coding.times._decode_datetime_with_pandas
                
                rd = xr.open_dataset(rfile)[v].isel(member_id=0) # rcp data
                hd = xr.open_dataset(hfile)[v].isel(member_id=0) # historical data
                
                
                
                hr_base1 = hd.loc[str(Base_clim[0]):str(Base_clim[1])]
                hr_base2 = rd.loc[str(Base_clim[0]):str(Base_clim[1])]
                
                if not type(hr_base2.time.values[0]) == np.datetime64:
                    xx  = [ datetime.datetime.strptime(str(t),'%Y-%m-%d %H:%M:%S') for t in hr_base2.time.values]
                    hr_base2.time.values[:] =xx
                
                hr_base = xr.concat([hr_base1, hr_base2], dim='time')
                if hr_base.time.size != (Base_clim[1] - Base_clim[0] + 1)*12: print('problem')
            
                
                hr_targ = rd.loc[str(Futr_clim[0]):str(Futr_clim[1])]
                
                bmean = hr_base.groupby('time.month').mean('time')
                tmean = hr_targ.groupby('time.month').mean('time')
                    
                diff = tmean.copy()
                diff[:] = tmean.values[:] - bmean.values[:]
                diff.to_netcdf(odir+ofile)
                
                #hr_base = hr_targ = hr = diff = rd = hd = None
                anom[v] = diff            
   
            except:
                print('\nError')
                err = True
