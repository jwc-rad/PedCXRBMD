from __future__ import annotations

import numpy as np
import os, pickle

# Function to evaluate F and G piecewise polynomials for given ages
def evaluate_fitted_model(ages, F_coeffs_list, G_coeffs_list, age_bound):
    F_values = np.zeros_like(ages)
    G_values = np.zeros_like(ages)
    
    n_intervals = len(age_bound)
    
    # Loop over each piecewise range defined by age_bound
    for i in range(n_intervals):
        age_min = age_bound[i]
        if i < n_intervals - 1:
            age_max = age_bound[i + 1]
            mask = (ages >= age_min) & (ages < age_max)
        else:
            mask = (ages >= age_min)
        
        if np.any(mask):
            F_values[mask] = np.polyval(F_coeffs_list[i][::-1], ages[mask])
            G_values[mask] = np.polyval(G_coeffs_list[i][::-1], ages[mask])

    return F_values, G_values

class BMDZCalc(object):
    def __init__(self):
        super().__init__()
        mod_filepath = os.path.abspath(__file__)
        mod_dirpath = os.path.dirname(mod_filepath)
        
        with open(os.path.join(mod_dirpath, 'BMDZ_spine_M.pkl'), 'rb') as f:
            params_spine_M = pickle.load(f)
        with open(os.path.join(mod_dirpath, 'BMDZ_spine_F.pkl'), 'rb') as f:
            params_spine_F = pickle.load(f)
            
        self.params_spine = {
            'M': params_spine_M,
            'F': params_spine_F,
        }
        
    def calc_z_spine(self, x, age, sex):
        tF_age, tG_age = evaluate_fitted_model(
            age, 
            self.params_spine[sex]['Mean'], self.params_spine[sex]['SD'], 
            self.params_spine[sex]['age_bound'],
        )
        
        Z_pred = (x - tF_age) / tG_age
        return Z_pred