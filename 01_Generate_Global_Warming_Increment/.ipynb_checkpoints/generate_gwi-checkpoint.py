#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:57:47 2022

@author: doan
"""





import xarray as xr

import numpy as np
import os, glob, sys
import xarray as xr
import matplotlib.pyplot as plt
import iris
import iris.plot as iplt
import iris.quickplot as qplt
import datetime
import numpy as np
import numpy.ma as ma
import pandas as pd
from iris.util import unify_time_units
from iris.coords import DimCoord
from iris.cube import Cube
#iris.FUTURE.netcdf_no_unlimited = True
from gridfill import fill
kw = dict(eps=1e-4, relax=0.6, itermax=1e4, initzonal=False,cyclic=True, verbose=True)
from collections import OrderedDict 



#==============================================================================
def fstr(Base_clim, Futr_clim): 
    tbs, tbe = Base_clim 
    tts, tte = Futr_clim       
    return str(tts)+'-'+str(tte)+'_'+str(tbs)+'-'+str(tbe)
#==============================================================================
    

#==============================================================================
def get_wps_info(wps_typ):
    f = open('get_grib_info/'+wps_typ+'.meta', 'r')
    lines = f.read().split('\n')
    f.close()
    lev = list(map(float,lines[0].split('=')[1].split(',')))
    y = list(map(float,lines[1].split('=')[1].split(',')))
    x = list(map(float,lines[2].split('=')[1].split(',')))
    return {'lev':lev, 'y':y, 'x':x}
#==============================================================================
    

var = ['ts', 'ta', 'hur', 'ua', 'va', 'zg']


Base_clim = (1995,2014)
Futr_clim = (2080, 2099)


#anom = cal_anomaly(Base_clim, Futr_clim, rcp, g, var, input_dir, output_dir)
#g = 'MIROC'
#rcp = 'ssp585'

output_dir = 'pgw_test'
input_dir = 'Download_all'


gcms = [ g.split('/')[-1] for g in sorted(glob.glob('Download_all/*'))]

exps = ['historical','ssp585','ssp370','ssp245','ssp126']

mems =  ['r1i1p1f1']

varn = ['ta','ua','va','hur','zg','ts']




if False:
    index = pd.MultiIndex.from_product( [ gcms, exps, mems ] , names=["gcm", "exp", 'mem'] )
    
    df = pd.DataFrame(index=index, columns=varn)
    
    
    for g in gcms:    
        for exp in ['historical','ssp585','ssp370','ssp245','ssp126'][: ]:
            for mem in ['r1i1p1f1']:
                for var in ['ta','ua','va','hur','zg','ts'][:]:
                    ifile = glob.glob(input_dir + '/' + g + '/' + exp + '/'+var+'_*'+exp+'*'+mem+'*') 
                    print(len(ifile))
                    df.loc[ (g, exp, mem), var ] = len(ifile)
                    
                    
                    
    df.to_csv('check_gcms.csv')
    
    
df = pd.read_csv('check_gcms.csv', index_col = [0,1,2])


for exp in exps[1:][:0]:
    
    print(exp)
    d = df[df.index.get_level_values(1) == exp]
    d0 = df[df.index.get_level_values(1) == 'historical' ]
    for ig, g in enumerate(gcms):
        x1, x0 = d.loc[g].astype(bool).values.all(), d0.loc[g].astype(bool).values.all()
        if not x1 & x0: continue
        print(ig, g, x1, x0)
        
        
        rcp = exp
        anom = OrderedDict()
        err = False
        for v in var[:]:
            try:
                if err: continue
                print('Baseclim: ', Base_clim, '\nFutureclim: ', Futr_clim, '\ngcm:', g, '\nVariable: ',v )
                tbs, tbe = Base_clim 
                tts, tte = Futr_clim   
                
                odir = output_dir  + '/diff/' +rcp+'/' +fstr(Base_clim, Futr_clim)+'/'+v+'/'
                ofile = v+'_'+g+'.nc' 
                if not os.path.exists(odir): os.makedirs(odir)
            
                print(input_dir, output_dir)
                # find files
                hfile = glob.glob(input_dir+ '/'+g+ '/historical/'+v+'_*'+g+'*')[0]
                rfile = glob.glob(input_dir+'/'+ g + '/'+ rcp+'/'+v+'_*'+g+'*')[0]
    
                rd = xr.open_dataset(rfile)[v].isel(member_id=0) # rcp data
                hd = xr.open_dataset(hfile)[v].isel(member_id=0) # historical data
                
                
                hr_base = hd.loc[str(Base_clim[0]):str(Base_clim[1])]
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
    
        wps_info = get_wps_info('eraI')
        
        
    
        dc = iris.cube.CubeList() 
        err = False
        for v in var[:]:
            try:
                if err: continue
                x,y, lev = wps_info['x'], wps_info['y'], wps_info['lev']
                
                idir = output_dir  + '/diff/' +rcp+'/' +fstr(Base_clim, Futr_clim)+'/'+v+'/'
                ifile =idir+v+'_'+g+'.nc' 
                
                print(ifile)
                
                if not os.path.exists(ifile): continue
                #rd    = iris.load_cube(ifile)    
                rd = xr.open_dataset(ifile)[v].to_iris()
                ny, nx = len(y), len(x)
                latitude = DimCoord(y, standard_name='latitude', units='degrees')
                longitude = DimCoord(x, standard_name='longitude', units='degrees')
                target_grid = Cube(np.zeros((ny, nx), np.float32),dim_coords_and_dims=[(latitude, 0),(longitude, 1)])
                                    
                if v == 'ts':
                    rd_reg  = rd.regrid(target_grid,iris.analysis.Nearest(),) 
                else:
                    xdim, ydim = 3,2
                    rd_fill = rd.copy()
                    if ma.is_masked(rd.data): 
                        filled, converged = fill(rd.data, xdim, ydim, **kw)
                        rd_fill.data[:] = filled
            
                    sample_points = [('air_pressure', lev)]
                    rd_vert = rd_fill.interpolate(sample_points,iris.analysis.Linear())
                    rd_reg  = rd_vert.regrid(target_grid,iris.analysis.Nearest(),)
            
                dc.append(rd_reg)    
            except:
                print('\nError')
                err = True
    
        #==============================================
        if len(dc) == 6: 
            print('co du 6 variables')
            pgw_dir       = output_dir + '/pgw_all/'
            odir = pgw_dir + '/' + rcp+'/'+fstr(Base_clim, Futr_clim)+'/'
            if not os.path.exists(odir): os.makedirs(odir)
            iris.save(dc,odir+g+'.nc',netcdf_format='NETCDF3_CLASSIC')    
        else: print('error: ko du 6 variables', g)  
    #==============================================================================
    


for exp in exps[1:][:1]:
    
    print(exp)  
    rcp = exp
    pgw_dir       = output_dir + '/pgw_all/'
    idir = pgw_dir + '/' + rcp+'/'+fstr(Base_clim, Futr_clim)+'/'       
    print(idir)
    files = glob.glob(idir+'/*nc')
        
    
    gg = [f.split('/')[-1].split('.')[0] for f in files]
    ds = xr.open_mfdataset(files, combine='nested',concat_dim='g')
    do = ds.mean(dim='g')
    do.attrs['list_of_gcms'] = ','.join(gg)
    
    odir = output_dir + '/pgw_avg/' + rcp+'/'+fstr(Base_clim, Futr_clim)+'/'            
    ofile = odir + 'ensemble_mean.nc'
    if not os.path.exists(odir): os.makedirs(odir)
    do.to_netcdf(ofile, format = 'NETCDF3_CLASSIC')  

    
        
        
    
    
    
    
        
        
        
        
    











if False:
    anom = OrderedDict()
    err = False
    for v in var[:0]:
        try:
            if err: continue
            print('Baseclim: ', Base_clim, '\nFutureclim: ', Futr_clim, '\ngcm:', g, '\nVariable: ',v )
            tbs, tbe = Base_clim 
            tts, tte = Futr_clim   
            
            odir = output_dir  + '/diff/' +rcp+'/' +fstr(Base_clim, Futr_clim)+'/'+v+'/'
            ofile = v+'_'+g+'.nc' 
            if not os.path.exists(odir): os.makedirs(odir)
        
            print(input_dir, output_dir)
            # find files
            hfile = glob.glob(input_dir+ '/'+g+ '/historical/'+v+'_*'+g+'*')[0]
            rfile = glob.glob(input_dir+'/'+ g + '/'+ rcp+'/'+v+'_*'+g+'*')[0]

            rd = xr.open_dataset(rfile)[v].isel(member_id=0) # rcp data
            hd = xr.open_dataset(hfile)[v].isel(member_id=0) # historical data
            
            
            hr_base = hd.loc[str(Base_clim[0]):str(Base_clim[1])]
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

    wps_info = get_wps_info('eraI')
    
    

    dc = iris.cube.CubeList() 
    err = False
    for v in var[:]:
        try:
            if err: continue
            x,y, lev = wps_info['x'], wps_info['y'], wps_info['lev']
            
            idir = output_dir  + '/diff/' +rcp+'/' +fstr(Base_clim, Futr_clim)+'/'+v+'/'
            ifile =idir+v+'_'+g+'.nc' 
            
            print(ifile)
            
            if not os.path.exists(ifile): continue
            #rd    = iris.load_cube(ifile)    
            rd = xr.open_dataset(ifile)[v].to_iris()
            ny, nx = len(y), len(x)
            latitude = DimCoord(y, standard_name='latitude', units='degrees')
            longitude = DimCoord(x, standard_name='longitude', units='degrees')
            target_grid = Cube(np.zeros((ny, nx), np.float32),dim_coords_and_dims=[(latitude, 0),(longitude, 1)])
                                
            if v == 'ts':
                rd_reg  = rd.regrid(target_grid,iris.analysis.Nearest(),) 
            else:
                xdim, ydim = 3,2
                rd_fill = rd.copy()
                if ma.is_masked(rd.data): 
                    filled, converged = fill(rd.data, xdim, ydim, **kw)
                    rd_fill.data[:] = filled
        
                sample_points = [('air_pressure', lev)]
                rd_vert = rd_fill.interpolate(sample_points,iris.analysis.Linear())
                rd_reg  = rd_vert.regrid(target_grid,iris.analysis.Nearest(),)
        
            dc.append(rd_reg)    
        except:
            print('\nError')
            err = True

    #==============================================
    if len(dc) == 6: 
        print('co du 6 variables')
        pgw_dir       = output_dir + '/pgw_all/'
        odir = pgw_dir + '/' + rcp+'/'+fstr(Base_clim, Futr_clim)+'/'
        if not os.path.exists(odir): os.makedirs(odir)
        iris.save(dc,odir+g+'.nc',netcdf_format='NETCDF3_CLASSIC')    
    else: print('error: ko du 6 variables', g)  
#==============================================================================














