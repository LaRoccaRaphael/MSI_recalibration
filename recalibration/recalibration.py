#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from sklearn import linear_model
from scipy.stats import gaussian_kde

def peak_selection(ms_intensities):
    # return the 300 mot intense centroid of a mass spectrum
    intensities_arr = np.array(ms_intensities)
    return(intensities_arr.argsort()[::-1][:300])

def compute_masserror(experimental_mass, database_mass, tolerance):
    # mass error in Dalton
    if database_mass != 0:
        return abs(experimental_mass - database_mass) <= tolerance
       
def binarySearch_tol(arr, l, r, x, tolerance): 
    # binary with a tolerance in Da search from an ordered list 
    while l <= r: 
        mid = l + (r - l)//2; 
        if compute_masserror(x,arr[mid],tolerance): 
            itpos = mid +1
            itneg = mid -1
            index = []
            index.append(mid)
            if( itpos < len(arr)):
                while compute_masserror(x,arr[itpos],tolerance) and itpos < len(arr):
                    index.append(itpos)
                    itpos += 1 
            if( itneg > 0): 
                while compute_masserror(x,arr[itneg],tolerance) and itneg > 0:
                    index.append(itneg)
                    itneg -= 1     
            return index 
        elif arr[mid] < x: 
            l = mid + 1
        else: 
            r = mid - 1
    return -1

def hits_generation(peaks_mz,database_exactmass, tolerance): 
    # for each detected mz return its index in of the hits in the database
    hit_errors = list()
    hit_exp = list()
    for i in range(0,np.size(peaks_mz,0)):
        exp_peak = peaks_mz[i]
        db_ind = binarySearch_tol(np.append(database_exactmass,np.max(database_exactmass)+1),
                                  0, len(database_exactmass)-1, exp_peak,tolerance)
        if db_ind != -1:
            for j in range(0,len(db_ind)):
                true_peak = database_exactmass[db_ind[j]]
                da_error = (exp_peak - true_peak)
                hit_errors.append(da_error)
                hit_exp.append(exp_peak)
    return(np.asarray(hit_exp),np.asarray(hit_errors))


def kde_scipy(x, x_grid, bandwidth=0.002, **kwargs):
    # kernel density estimation of the hit errors 
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def hits_selection(hit_errors, step, tolerance, da_limit):
    # return the indexes of the hits of the most populated error region 
    x = np.asarray(hit_errors)
    x_grid = np.arange(-tolerance,tolerance+0.0001,0.0001)
    pdf = kde_scipy(x, x_grid, bandwidth=step)
    max_da_value = x_grid[np.argmax(pdf,axis=0)]
    roi = (x <= (max_da_value + da_limit)) & (x >= (max_da_value -da_limit ))
    return(roi) 


def create_lm(hit_exp,hit_errors,tolerance=30,da_limit=2.5,step=0.001):
    # estimate a linear model of the mz error according to the mz with RANSAC algorithm
    X = np.vander(hit_exp, 2) # 2d array for ransac algorithm, we add only ones in the second column 
    roi = hits_selection(hit_errors,step,tolerance=tolerance,da_limit=da_limit)
    y = hit_errors[roi]
    X = X[roi,]
    try:
        model = linear_model.RANSACRegressor(max_trials=300, min_samples=10)
        mz_error_model = model.fit(X, y)
    except ValueError:
        print("error")
        mz_error_model = []
    return(mz_error_model)

def correct_mz_lm(ms_mzs,mz_error_model):
    # predict the Da errors for each detected mz and correct them
    X = np.vander(ms_mzs, 2)
    predicted_mz_errors = mz_error_model.predict(X)
    estimated_mz = ms_mzs - predicted_mz_errors
    return(estimated_mz)
    

def write_corrected_msi(msi,output_file,tolerance,database_exactmass,step,dalim):
    # iterate throug each pixel of an MSI 
    with ImzMLWriter(output_file) as w:
        p = ImzMLParser(msi, parse_lib='ElementTree')
        for idx, (x,y,z) in enumerate(p.coordinates):
            
            ms_mzs, ms_intensities = p.getspectrum(idx)
            peaks_ind = peak_selection(ms_intensities)
            peaks_mz = ms_mzs[peaks_ind]
            
            if len(peaks_mz) >30 :
                hit_exp, hit_errors = hits_generation(peaks_mz,database_exactmass, tolerance)
                if len(hit_errors) > 10:
                    roi = hits_selection(hit_errors, step, tolerance , da_limit=dalim)
                    if np.sum(roi) > 10:
                        mz_error_model = create_lm(hit_exp,hit_errors,tolerance=tolerance,da_limit=dalim,step=step)
                        if mz_error_model:
                            corrected_mzs = correct_mz_lm(ms_mzs, mz_error_model)
                            w.addSpectrum(corrected_mzs, ms_intensities, (x,y,z))

                            
my_parser = argparse.ArgumentParser(allow_abbrev=False)
my_parser.add_argument('-i','--input', action='store', type=str, required=True,help='file path to an imzML')
my_parser.add_argument('-i2','--input2', action='store', type=str, required=True,help='file containing the calibrating ion mass values')
my_parser.add_argument('-o','--output', action='store', type=str, required=True,help='file path for the recalibrated MSI in imzML format')
my_parser.add_argument('-st','--step', action='store', type=float, required=True,help='bandwidth for the density estimation function')
my_parser.add_argument('-tl','--tol', action='store', type=float, required=True,help='Da tolerance for the identifications')
my_parser.add_argument('-lm','--dalim', action='store', type=float, required=True,help='limit in Da for hits selection')


args = my_parser.parse_args()

msi = args.input
step = args.step
tolerance = args.tol
database_name = args.input2
dalim = args.dalim


exact_mass_full = np.genfromtxt(database_name)
# order the list of masses for the binary search 
database_exactmass = exact_mass_full[exact_mass_full.argsort()]


output_file = args.output
write_corrected_msi(msi,output_file,tolerance,database_exactmass,step,dalim)
