#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 12:45:53 2026

@author: asher
"""

# IMPORT LIBARARIES
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
import seaborn as sns  # ◄ ADD THIS LINE TO REGISTER 'mako'
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
from matplotlib import cm

#FUNCTIONS 
def calculate_piecewise_linear(y, x, xe=None, breakpoint=2022, maxlags=12):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    x_target = np.asarray(xe).flatten() if xe is not None else x
    
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]

    pre_mask = x_clean < breakpoint
    x_pre = x_clean[pre_mask]
    y_pre = y_clean[pre_mask]
    N = len(x_pre)

    pre_slope = 0
    if N > 0:
        y_at_breakpoint = np.mean(y_pre)
    else:
        y_at_breakpoint = y_clean[0] if len(y_clean) > 0 else 0

    stats_dict = {
        'slope_raw': np.nan,
        'p_val_raw': np.nan,
        'p_val_adj': np.nan,
        'std_err_adj': np.nan,
        'pre_trend_applied': False
    }

    #Pre-Breakpoint Trend calculation - Appendix Table 1 (HAC / Newey-West)
    if N > 2:
        #Add a constant column to x vector for the intercept term in statsmodels
        X_pre_sm = sm.add_constant(x_pre)
        
        #Fit OLS model
        ols_model = sm.OLS(y_pre, X_pre_sm).fit()
        
        #Extract baseline OLS properties
        slope_raw = ols_model.params[1]
        intercept_raw = ols_model.params[0]
        p_val_raw = ols_model.pvalues[1]
        
        #Newey-West (HAC) correction fpr autocorrelation and heteroskedasticity
        #'maxlags' sets the bandwidth window size (12 lags is typical for monthly anomalies)
        hac_results = ols_model.get_robustcov_results(cov_type='HAC', maxlags=maxlags)
        p_val_adj = hac_results.pvalues[1]
        std_err_adj = hac_results.bse[1]
        
        #Update stats
        stats_dict['slope_raw'] = slope_raw
        stats_dict['p_val_raw'] = p_val_raw
        stats_dict['p_val_adj'] = p_val_adj
        stats_dict['std_err_adj'] = std_err_adj

        #trend if HAC-adjusted p-value is significant
        if p_val_adj < 0.05:
            pre_slope = slope_raw
            y_at_breakpoint = (slope_raw * breakpoint) + intercept_raw
            stats_dict['pre_trend_applied'] = True

    #Post-Breakpoint Slope
    post_mask = x_clean >= breakpoint
    if np.any(post_mask):
        X_post = (x_clean[post_mask] - breakpoint).reshape(-1, 1)
        Y_post = (y_clean[post_mask] - y_at_breakpoint)
        
        model = LinearRegression(fit_intercept=False)
        model.fit(X_post, Y_post)
        slope_post = model.coef_[0]
    else:
        slope_post = 0

    #Construct Trend
    x_shifted = x_target - breakpoint
    trend = np.where(
        x_target < breakpoint,
        y_at_breakpoint + (pre_slope * x_shifted),
        y_at_breakpoint + (slope_post * x_shifted)
    )
    
    return trend, stats_dict


def get_season_info(index):
    """Helper to map months to DJF, MAM, JJA, SON and adjust December's year."""
    seasons = {12: 'DJF', 1: 'DJF', 2: 'DJF',
               3: 'MAM', 4: 'MAM', 5: 'MAM',
               6: 'JJA', 7: 'JJA', 8: 'JJA',
               9: 'SON', 10: 'SON', 11: 'SON'}
    
    season_names = index.month.map(seasons)
    # Move December to the 'Year' of the following January/February
    season_years = index.year
    season_years = np.where(index.month == 12, season_years + 1, season_years)
    
    return season_names, season_years

def get_processed_series1(df_main, loc, press, remseascyl, interpOrNot, varname):
    column_map = {
        'H2O': 'H2O_Residuals',
        'logH2O': 'log_H2O_Residuals',
        'Temp': 'Temp_Residuals',
        'H2SO4': '%Wt_Residuals',
        'H2SO4_abs': 'H2SO4_%wt_amb',
        'logPress': 'log_Press_Residuals',
        'Press': 'Press_Residuals'
    }
    
    target_col = column_map.get(varname, varname)
    
    #Filter and set index 
    df_sub = df_main[(df_main['Location'] == loc) & (df_main['Pressure [hPa]'] == press)].copy()
    if df_sub.empty: return None

    df_sub1 = df_sub.set_index('Datetime_M').sort_index()
    
    
    #remove seasonal cycle
    if ((remseascyl == 1)):
        #Climatology
        if varname == 'H2O':
            #through Dec 2021
            df_clim_source = df_sub1[df_sub1.index.year <= 2021]
        else:
            df_clim_source = df_sub1
        
        if df_clim_source.empty: return None

        #Monthly Climatology
        climatology = df_clim_source.groupby(df_clim_source.index.month)[target_col].mean()
        
        #Subtract Climatology from original time-series
        mapped_climatology = df_sub1.index.month.map(climatology)
        data_to_resample = df_sub1[target_col] - mapped_climatology  
        
        
    else:
        data_to_resample = df_sub1[target_col] # for absolute timeseries 
    
        #HAC/Newey-West Trend Audit on Monthly Deviations
    if remseascyl == 1 and varname == 'H2O':
            # Check for NaNs
            clean_dev = data_to_resample.dropna()
            
            if len(clean_dev) > 2:
                # Convert to decimal fractional years
                x_years = clean_dev.index.year + (clean_dev.index.month - 0.5) / 12.0
                y_values = clean_dev.values
                
                #OLS Fit & Newey-West standard errors
                X_matrix = sm.add_constant(x_years)
                ols_model = sm.OLS(y_values, X_matrix).fit()
                hac_results = ols_model.get_robustcov_results(cov_type='HAC', maxlags=12)
                
                print(f"\n [HAC Monthly Deviation Trend Audit] {loc} at {press} hPa ({varname})")
                print(f"   • Trend per Decade: {hac_results.params[1] * 10:.5f} ppmv/dec")
                print(f"   • Raw p-value:     {ols_model.pvalues[1]:.4f}")
                print(f"   • HAC Adj p-value: {hac_results.pvalues[1]:.4f}")
                print(f"   • HAC Std Error:   {hac_results.bse[1]:.6f}")
                print("-" * 65)

        
    
    #resample monthly deviations -> seasonal deviations
    season_names, season_years = get_season_info(data_to_resample.index)
    
    #Groupby year & season
    temp_df = data_to_resample.to_frame(name='val')
    temp_df['s_name'] = season_names
    temp_df['s_year'] = season_years
    seasonal_grp = temp_df.groupby(['s_year', 's_name'])['val'].mean()
    
    #Seasonal map 
    month_map = {'DJF': 1, 'MAM': 4, 'JJA': 7, 'SON': 10}
    new_index = [pd.Timestamp(year=y, month=month_map[s], day=1) for y, s in seasonal_grp.index]
    series = pd.Series(seasonal_grp.values, index=new_index).sort_index()
    series.index = series.index.to_period('M').to_timestamp() 
    
    #Interpoalte option (not selected)
    if interpOrNot ==1 and series.isna().any():
        series = series.interpolate(method='time', limit=2, limit_direction='both')
        
    return series.dropna()

def get_monthly_climatology(df_main, loc, press, remseascyl, varname):
    #options to calculate monthly climatologies
    column_map = {
        'H2O': 'H2O_Residuals',
        'logH2O': 'log_H2O_Residuals',
        'Temp': 'Temp_Residuals',
        'H2SO4': '%Wt_Residuals',
        'logPress': 'log_Press_Residuals',
        'Press': 'Press_Residuals'
    }
    
    target_col = column_map.get(varname, varname)
    
    #Filter & set index
    df_sub = df_main[(df_main['Location'] == loc) & (df_main['Pressure [hPa]'] == press)].copy()
    if df_sub.empty: return None

    df_sub1 = df_sub.set_index('Datetime_M').sort_index()
    
    #remove seasonal cycle
    if remseascyl == 1:
        # using Climatology
        if varname == 'H2O':
            # Use data through Dec 2021
            df_clim_source = df_sub1[df_sub1.index.year <= 2021]
        else:
            df_clim_source = df_sub1
        
        if df_clim_source.empty: return None

        #Calculate Monthly Climatology
        climatology = df_clim_source.groupby(df_clim_source.index.month)[target_col].mean()

                                  
    return climatology
                                 

# Optimized H2SO4 Weight Percent Calculation
def calc_h2so4_wt_vectorized(T_k, Pw_hpa):
    """
    Vectorized calculation of H2SO4 aerosol %wt. Tabazadeh 1997
    """
    #Constants
    Wi = np.linspace(0.1, 0.8, 15)
    ai = np.array([19.726, 19.747, 19.761, 19.794, 19.883, 20.078, 20.379, 20.637, 20.682, 20.55, 20.405, 20.383, 20.585, 21.169, 21.808])
    bi = np.array([-4364.8, -4390.9, -4414.7, -4451.1, -4519.2, -4644.0, -4828.5, -5011.5, -5121.3, -5177.6, -5252.1, -5422.4, -5743.8, -6310.6, -6985.9])
    ci = np.array([-147620, -144690, -142940, -140870, -136500, -127240, -112550, -98811, -94033, -96984, -100840, -97966, -83701, -48396, -12170])

    # Rounding for matching
    Pw_rounded = np.round(Pw_hpa, decimals=5)

    #Interpolation grid
    W_grid = np.linspace(0.1, 0.8, 701)
    a_interp = np.interp(W_grid, Wi, ai)
    b_interp = np.interp(W_grid, Wi, bi)
    c_interp = np.interp(W_grid, Wi, ci)

    #Calculate lookup grid for current Temperature
    pw_calc_grid = np.exp(a_interp + b_interp/T_k + c_interp/T_k**2)
    
    #Find closest match
    idx = np.abs(pw_calc_grid - Pw_rounded).argmin()
    wt = W_grid[idx]

    # 5. Gmitro & Vermeulen Threshold 
    if wt >= 0.8:
        W_GV = np.array([0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98])
        R = 1.98726
        alpha = np.array([0.0114, 0.0568, 0.1233, 0.0666, -0.0120, -0.0346, -0.0398, -0.0427, -0.0436, -0.0428, -0.0405, -0.0368, -0.0314, -0.024])
        Cp_298 = np.array([9.77, 10.36, 13.78, 18.96, 22.13, 22.76, 22.30, 21.48, 20.44, 19.32, 18.06, 16.64, 15.05, 13.25])
        L_298 = np.array([-3475, -4015, -4656, -5319, -5938, -6419, -6627, -6816, -6983, -7139, -7286, -7433, -7574, -7712])
        F_F0 = np.array([-3090, -3427, -3789, -4167, -4557, -4960, -5165, -5375, -5595, -5830, -6090, -6390, -6741, -7204])
        
        A = -3.67340 + 1/R * (Cp_298 - 298.15 * alpha)
        B = -4143.5 + 1/R * (L_298 - 298.15 * Cp_298 + 298.15**2/2 * alpha)
        C = 10.24353 + 1/R * (Cp_298 + (F_F0 - L_298) * 1/298.15)
        D = 0.618943E-3 - alpha/(2 * R)
        
        pw_gv = np.exp(A + B/T_k + C*np.log(T_k) + D*T_k)
        idx_gv = np.abs(pw_gv - Pw_rounded).argmin()
        wt = W_GV[idx_gv]
    return wt

def goff_gratch_ice(T):
    """Standard Goff-Gratch formulation over ice (T in Kelvin, returns e_i in hPa)."""
    val = (-9.09718 * (273.16 / T - 1) 
           - 3.56654 * np.log10(273.16 / T) 
           + 0.876793 * (1 - T / 273.16) 
           + np.log10(6.1071))
    return 10**val

def calculate_t_fph_vectorized(e_hpa):
    """Vectorized Newton-Raphson solver find T_fph from water vapor partial pressure."""
    t_guess = np.full_like(e_hpa, 220.0, dtype=float)
    for _ in range(6):  
        e_calc = goff_gratch_ice(t_guess)
        dF_dT = np.log(10) * e_calc * (
            (9.09718 * 273.16 / t_guess**2) + 
            (3.56654 / (t_guess * np.log(10))) - 
            (0.876793 / 273.16)
        )
        t_guess -= (e_calc - e_hpa) / dF_dT
    return t_guess



#Variables
trend_levels = {
        'Boulder': [26, 38, 46, 56, 68, 82, 100], 
        'Lauder': [26, 38, 46, 56, 68, 82, 100],
        'Hilo': [26, 38, 46, 56, 68, 82, 100]
    }
plot_press_levels = [26, 38, 46, 56, 68, 82, 100] #all pressure levels 

model_press_levels = [38,  56, 82]
#press_levels = [26, 46, 68, 100]
    
#READ IN DATA & assign datetime & Year values
root_dir = '/Users/asher/Documents/'
df_p0_i = pd.read_csv(root_dir + 'AGUAnalysisFiles/balloon_w_press_2026.csv', sep=',', dtype=None, engine='python') #root_dir = '/Users/asher/Documents/'
df_p0_i['Datetime'] = pd.to_datetime(df_p0_i['Launch'])

#Getting datetime
df_p0_i['Datetime_M'] = df_p0_i['Datetime'].dt.to_period('M').dt.to_timestamp() 
# Create a datetime object for the start of each year
start_of_year = df_p0_i['Datetime'].dt.year

# Calculate the fractional year
df_p0_i['Year'] = df_p0_i['Datetime_M'].dt.year + \
           (df_p0_i['Datetime_M'].dt.dayofyear - 1) / \
           (365 + df_p0_i['Datetime_M'].dt.is_leap_year.astype(int))
           

#Calculate H2SO4 % Wt & remove NAs
#H2SO4 calculation vectorized (FAST)    
df_p0_i['Pw_frostpoint']  = df_p0_i['H2O_ppmv']*1e-6*df_p0_i['Press [hPa]']/(1+ df_p0_i['H2O_ppmv']*1e-6) #algebraicly calculate Pw_frostpoint
v_lookup = np.vectorize(calc_h2so4_wt_vectorized)
df_p0_i['H2SO4_%wt_amb'] = v_lookup(df_p0_i['Tcorr_degC'] + 273.15, df_p0_i['Pw_frostpoint']) #e_i in hPa
#clean data
df_p0 = df_p0_i[(df_p0_i['H2O_ppmv'].notna()) & (df_p0_i['Year'] >= 2004)].copy()

mdt_results = []
mdt_temp_results = []
mdt_h2o_results = []

# Historical baseline window prior to Hunga Tonga Volcanic Eruption
df_baseline = df_p0[df_p0['Year'] < 2022]


for loc in ['Lauder', 'Hilo', 'Boulder']:
    for press in plot_press_levels:
        mask = (df_baseline['Location'] == loc) & (df_baseline['Pressure [hPa]'] == press)
        
        #Ambient Acid %wt Processing
        subset = df_baseline[mask].dropna(subset=['Year', 'H2SO4_%wt_amb'])
        
        if len(subset) < 15:
            continue
            
        x = subset['Year'].values
        y = subset['H2SO4_%wt_amb'].values
        
        #OLS fit with HAC robust standard errors
        X = sm.add_constant(x)
        ols_model = sm.OLS(y, X).fit()
        hac_results = ols_model.get_robustcov_results(cov_type='HAC', maxlags=12)
        
        # Extract adjusted standard error of the slope and residuals
        std_err_adj = hac_results.bse[1]
        resid = hac_results.resid
        
        # Calculate sample lag-1 rho f\
        rho = np.corrcoef(resid[:-1], resid[1:])[0, 1] if len(resid) > 2 else 0
        rho = 0 if np.isnan(rho) else max(-0.95, min(rho, 0.95))
        
        # Compute the Minimum Detectable Trend slope at 80% Power per decade
        mdt_slope_per_decade = (2.801 * std_err_adj) * 10
        
        mdt_results.append({
            'Site': loc,
            'Pressure_hPa': press,
            'MDT_per_Decade': mdt_slope_per_decade,
            'Residual_Noise_sigma': np.std(resid),
            'Lag1_Autocorr_rho': rho
        })

        #Ambient Temperature MDT Processing
        subset_temp = df_baseline[mask].dropna(subset=['Year', 'Tcorr_degC'])
        if len(subset_temp) >= 15:
            x = subset_temp['Year'].values
            y = subset_temp['Tcorr_degC'].values
            
            X = sm.add_constant(x)
            ols_model = sm.OLS(y, X).fit()
            hac_results = ols_model.get_robustcov_results(cov_type='HAC', maxlags=12)
            
            std_err_adj = hac_results.bse[1]
            resid = hac_results.resid
            
            rho = np.corrcoef(resid[:-1], resid[1:])[0, 1] if len(resid) > 2 else 0
            rho = 0 if np.isnan(rho) else max(-0.95, min(rho, 0.95))
            
            mdt_slope = (2.801 * std_err_adj) * 10
            
            mdt_temp_results.append({
                'Site': loc, 
                'Pressure_hPa': press, 
                'MDT_per_Decade': mdt_slope,
                'Residual_Noise_sigma': np.std(resid), 
                'Lag1_Autocorr_rho': rho
            })
            
        #Ambient Water Vapor MDT Processing
        subset_h2o = df_baseline[mask].dropna(subset=['Year', 'H2O_ppmv'])
        if len(subset_h2o) >= 15:
            x = subset_h2o['Year'].values
            y = subset_h2o['H2O_ppmv'].values
            
            X = sm.add_constant(x)
            ols_model = sm.OLS(y, X).fit()
            hac_results = ols_model.get_robustcov_results(cov_type='HAC', maxlags=12)
            
            std_err_adj = hac_results.bse[1]
            resid = hac_results.resid
            
            rho = np.corrcoef(resid[:-1], resid[1:])[0, 1] if len(resid) > 2 else 0
            rho = 0 if np.isnan(rho) else max(-0.95, min(rho, 0.95))
            
            mdt_slope = (2.801 * std_err_adj) * 10
            
            mdt_h2o_results.append({
                'Site': loc, 
                'Pressure_hPa': press, 
                'MDT_per_Decade': mdt_slope,
                'Residual_Noise_sigma': np.std(resid), 
                'Lag1_Autocorr_rho': rho
            })
            
#tracking dataframe
df_mdt = pd.DataFrame(mdt_results)
df_mdt_temp = pd.DataFrame(mdt_temp_results)
df_mdt_h2o = pd.DataFrame(mdt_h2o_results)

#pivot to grid display
mdt_table_final = df_mdt.pivot(index='Pressure_hPa', columns='Site', values='MDT_per_Decade')
mdt_table_temp_final = df_mdt_temp.pivot(index='Pressure_hPa', columns='Site', values='MDT_per_Decade')
mdt_table_h2o_final = df_mdt_h2o.pivot(index='Pressure_hPa', columns='Site', values='MDT_per_Decade')


#PLOT TIMESERIES 
#subtract mean and seasonal cycle to yeild deviations

new_cols = ['H2O_Residuals', 'log_H2O_Residuals', 'Temp_Residuals', '%Wt_Residuals', 'log_Press_Residuals']
df_p0.loc[:, new_cols] = np.nan

var_mapping = {
'H2O_ppmv':      'H2O_Residuals',
'log_H2O_ppmv':  'log_H2O_Residuals',
'Tcorr_degC':    'Temp_Residuals',
'H2SO4_%wt_amb': '%Wt_Residuals',
'log_Press [hPa]': 'log_Press_Residuals'}

df_p0['log_H2O_ppmv'] = np.log(df_p0['H2O_ppmv'])
df_p0['log_Press [hPa]'] = np.log(df_p0['Press [hPa]'])


mean_results = []
#Process data from each site and each variable of interest
for loc in ['Lauder', 'Hilo', 'Boulder']:
    for press in plot_press_levels:
        mask = (df_p0['Location'] == loc) & (df_p0['Pressure [hPa]'] == press)
        #process each atmospheric variable
        for raw_col, res_col in var_mapping.items():
            #drop NaNs 
            subset = df_p0[mask].dropna(subset=['Year', raw_col])
            if subset.empty:
                continue
            
            x = subset['Year'].values
            y = subset[raw_col].values
            
            # Check for mean trend - start of time-series to 2022
            if press in trend_levels[loc]:
                trend, stats_dict = calculate_piecewise_linear(y, x,  xe=None, breakpoint=2022) 
                residuals = y - trend
                label_type = "Piecewise LM Detrended"
            else:
                # Subtract time-series mean
                residuals = y - np.mean(y)
                label_type = "Mean Centered"
                
                mean_val = np.mean(y)
                mean_results.append({
                    'Location': loc,
                    'Pressure [hPa]': press,
                    'Variable': raw_col,
                    'Mean_Value': mean_val
                })
            
                
            # Assign back to the dataframe
            #df_p0.loc[subset.index, 'H2O_Residuals'] = residuals
            df_p0.loc[subset.index, res_col] = residuals
            df_p0.loc[subset.index, res_col.replace('Residuals', 'Trend')] = trend

df_means = pd.DataFrame(mean_results)

# target_var = 'H2SO4_%wt_amb' 

# sites = ['Lauder', 'Hilo', 'Boulder']
# fig, axes = plt.subplots(3, 1, figsize=(18, 6), sharey=True)

# for i, loc in enumerate(sites):
#     means = []
#     levels = []
    
#     #sort pressure levels 
#     sorted_press = sorted(plot_press_levels)
    
#     for press in sorted_press:
#         #mask data
#         mask = (df_p0['Location'] == loc) & (df_p0['Pressure [hPa]'] == press)
#         subset_y = df_p0[mask][target_var].dropna()
        
#         if not subset_y.empty:
#             means.append(np.mean(subset_y))
#             levels.append(press)
    
#     axes[i].hlines(y=levels, xmin=0, xmax=means, color='gray', alpha=0.5, linewidth=1.5)
#     axes[i].scatter(means, levels, s=100, color= '#D95F02', edgecolor='black', zorder=3)
    
#     #format
#     axes[i].set_title(f'Site: {loc}', fontsize=14, fontweight='bold')
#     axes[i].set_ylabel('Pressure Level [hPa]', fontsize=12)
#     axes[i].grid(axis='y', linestyle='--', alpha=0.4)
#     axes[i].invert_yaxis()
#     axes[i].set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ])
    
#     if i == 2:
#         axes[i].set_xlabel('Mean $H_2SO_4$ Weight Fraction', fontsize=12)
        
# axes[0].text(-0.02, 30, 'a.', fontsize=12, fontweight='bold')
# axes[1].text(-0.02, 30, 'b.', fontsize=12, fontweight='bold')
# axes[2].text(-0.02, 30, 'c.', fontsize=12, fontweight='bold')

# plt.tight_layout()
# plt.show()


target_var = 'H2SO4_%wt_amb' 
sites = ['Lauder', 'Hilo', 'Boulder']

# Configuration for styles: (Marker, Color)
site_styles = {
    'Hilo':    {'marker': 'o', 'color': '#FF8C00'},    # Dark Orange
    'Lauder':  {'marker': 'D', 'color': '#FFB347'},    # Pastel Orange
    'Boulder': {'marker': 's', 'color': '#CC5500'}     # Burnt Orange
}

fig, ax = plt.subplots(figsize=(10, 8))

for loc in sites:
    means = []
    levels = []
    sorted_press = sorted(plot_press_levels)
    
    for press in sorted_press:
        mask = (df_p0['Location'] == loc) & (df_p0['Pressure [hPa]'] == press)
        subset_y = df_p0[mask][target_var].dropna()
        
        if not subset_y.empty:
            means.append(np.mean(subset_y))
            levels.append(press)
    
    #plotting
    ax.plot(means, levels, 
            marker=site_styles[loc]['marker'], 
            color='black',              # Connection line color
            markerfacecolor=site_styles[loc]['color'], 
            markeredgecolor='black',
            markersize=8, 
            linestyle='-', 
            linewidth=1.5,
            label=loc)

#formatting
ax.set_title(f'Mean $H_2SO_4$ Weight Fraction by Pressure and Location', fontsize=14, fontweight='bold')
ax.set_ylabel('Pressure Level [hPa]', fontsize=12)
ax.set_xlabel('Mean $H_2SO_4$ Weight Fraction', fontsize=12)
ax.grid(axis='both', linestyle='--', alpha=0.4)
ax.invert_yaxis()
ax.legend(title="Location")

plt.tight_layout()
plt.show()

#plot not used in JGR-A submission
target_var = 'H2O_ppmv' 

sites = ['Lauder', 'Hilo', 'Boulder']
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for i, loc in enumerate(sites):
    means = []
    levels = []
    
    #sort pressure levels
    sorted_press = sorted(plot_press_levels)
    
    for press in sorted_press:
        #mask data
        mask = (df_p0['Location'] == loc) & (df_p0['Pressure [hPa]'] == press)
        subset_y = df_p0[mask][target_var].dropna()
        
        if not subset_y.empty:
            means.append(np.mean(subset_y))
            levels.append(press)
    
    #plotting
    axes[i].hlines(y=levels, xmin=0, xmax=means, color='gray', alpha=0.5, linewidth=1.5)
    axes[i].scatter(means, levels, s=100, color='royalblue', edgecolor='black', zorder=3)
    
    # Formatting
    axes[i].set_title(f'Site: {loc}', fontsize=14, fontweight='bold')
    axes[i].set_ylabel('Pressure Level [hPa]', fontsize=12)
    axes[i].grid(axis='y', linestyle='--', alpha=0.4)
    axes[i].invert_yaxis()
    
    if i == 0:
        axes[i].set_xlabel('Mean $H_2O$ (ppmv)', fontsize=12)

plt.tight_layout()
plt.show()



#cividis rocket  mako
colors = mpl.colormaps['cividis'].resampled(len(plot_press_levels))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,  figsize=(30, 9), sharey=True)
ax_map = {'Lauder': ax1, 'Hilo': ax2, 'Boulder': ax3}

for loc in [ 'Lauder', 'Boulder', 'Hilo']: 
            ax = ax_map[loc]    
            count = 0
            
            for press in model_press_levels:
            
                df_p0_s = df_p0[df_p0['Year'] < 2027] #previously 2022
                h2o_ser  = get_processed_series1(df_p0_s , loc, press, 1, 0, varname='H2O')
                temp_ser = get_processed_series1(df_p0_s , loc, press, 1, 0, varname='Temp')
                h2so4_ser= get_processed_series1(df_p0_s , loc, press, 1, 0, varname='H2SO4')                
                x1_std = temp_ser
                x2_std = h2o_ser
                y_std = h2so4_ser
                X = np.column_stack((x1_std, x2_std))
                df_pairs = pd.DataFrame({'H2SO4': h2so4_ser ,'Temp': temp_ser, 'H2O': h2o_ser})

                #OLS regression
                X = sm.add_constant(X)
                X1 = sm.add_constant(x1_std)
                X2 = sm.add_constant(x2_std)
                
                model_ols = sm.OLS(y_std, X).fit()
                model_ols1 = sm.OLS(y_std, X1).fit()

                #extract AIC 
                aic_1 = model_ols.aic
                aic_2 = model_ols1.aic

                delta_aic = aic_1 - aic_2 #positive means Model 2 is better

                color1 = colors(count*3)
                
                b0, b1 = model_ols1.params
                se_b0, se_b1 = model_ols1.bse
                
                #eq string
                sign = "+" if b0 >= 0 else "-"
                eq_str = f"y={b1:.6f}x{sign}{abs(b0):.6f}"

             
                ax.scatter(x1_std, y_std, alpha=0.4, s=15, color=color1, edgecolors='k')
                #plot regression Line
                x_range = np.linspace(x1_std.min(), x1_std.max(), 100)
                y_line = b0 + b1 * x_range

                #updated plot 
                ax.plot(x_range, y_line, color=color1, lw=2.5, 
                        label=f"{press} hPa: {eq_str} (Adj-R²={model_ols1.rsquared_adj:.2f})")#label=f"{press} hPa: {eq_str} (R²={results1.rsquared:.2f})")
                
                ax.set_xlabel("Temperature Deviations ($\sigma$ C)", fontsize=16)
                if loc == 'Lauder':
                    ax.set_ylabel("$H_2SO_4$ Fractional Wt Deviations ($\sigma$)", fontsize=16)
                ax.set_title(loc)
                ax.set_ylim([-0.075, 0.075])
                
                ax.legend(title="Pressure Level", fontsize='small', loc='upper left', frameon=True)
                plt.savefig('/Users/asher/Temp_v_H2SO4_STAT5010.png', dpi=400, bbox_inches='tight')
            
                print(press)
                print(loc)
                count = count + 1

                
                if count >5:
                    print(model_ols1.summary())
plt.show()

colors = mpl.colormaps['cividis'].resampled(len(plot_press_levels))
#seasonal climatology H2SO4 acid weidht fraction
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(30, 10), sharey=True)
ax_map = {'Boulder': ax1, 'Hilo': ax2, 'Lauder': ax3}

for loc in ['Boulder', 'Hilo', 'Lauder']: 
    ax = ax_map[loc]
    
    for i, press in enumerate(plot_press_levels):
        #monthly mean anomalies (mean level at each pressure subtracted)
        h2so4_cycle = get_monthly_climatology(df_p0, loc, press,1, varname='H2SO4')
        
        if h2so4_cycle is not None:
            #plot seasonal anomalies
            ax.plot(h2so4_cycle.index, h2so4_cycle.values, 
                    marker='o', 
                    markersize=8,
                    linewidth=2.5, 
                    color=colors(i), 
                    label=f'{press} hPa')

    # formatting
    ax.set_title(f'{loc}: $H_2SO_4$ Seasonal Climatology', fontsize=16, pad=20)
    ax.set_xlabel('Month', fontsize=16)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontsize=15)
    ax.set_yticks([-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06])
    ax.tick_params(axis='y', labelsize=15) # This controls the font size
    ax.grid(True, linestyle=':', alpha=0.7)

#y-axis label
ax2.set_ylabel('$H_2SO_4$ Weight Fraction', fontsize=15)
ax1.text(1, 0.04, 'a.', fontsize=15, fontweight='bold')
ax2.text(1, 0.04, 'b.', fontsize=15, fontweight='bold')
ax3.text(1, 0.04, 'c.', fontsize=15, fontweight='bold')
#legend
ax3.legend(title='Pressure Level', title_fontsize=16, fontsize=16, 
           bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

#seasonal climatology water vapor (not used in JGR-A submission)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10), sharey=True)
ax_map = {'Lauder': ax1, 'Hilo': ax2, 'Boulder': ax3}

for loc in ['Lauder', 'Hilo', 'Boulder']: 
    ax = ax_map[loc]
    
    for i, press in enumerate(plot_press_levels):
        #monthly mean anomalies
        h2so4_cycle = get_monthly_climatology(df_p0, loc, press,1, varname='H2O')
        
        if h2so4_cycle is not None:
            #plot seasonal anomalies
            ax.plot(h2so4_cycle.index, h2so4_cycle.values, 
                    marker='o', 
                    markersize=8,
                    linewidth=2.5, 
                    color=colors(i), 
                    label=f'{press} hPa')

    #formatting
    ax.set_title(f'{loc}: $H_2O$ Seasonal Climatology', fontsize=14, pad=20)
    ax.set_xlabel('Month', fontsize=16)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)

ax1.set_ylabel('$H_2SO_4$ Weight Fraction', fontsize=18)
ax3.legend(title='Pressure Level', title_fontsize=15, fontsize=12, 
           bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


#calculate ANOVA to compare models
anova_results = anova_lm(model_ols1, model_ols) #aruments = (full model, reduced model)
print(anova_results)
#create ANOVA table
anova_table = []
fig, (ax1, ax2, ax3) = plt.subplots(1, 3,  figsize=(30, 9), sharey=True)
ax_map = {'Lauder': ax1, 'Hilo': ax2, 'Boulder': ax3}

#MSPE Comparison including both 2004 - 2022 and 2022 - 2026 periods
fig_era, (ax_e1, ax_e2, ax_e3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
ax_era_map = {'Lauder': ax_e1, 'Hilo': ax_e2, 'Boulder': ax_e3}

fig_ma_era, (ax_m1, ax_m2, ax_m3) = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
ax_ma_plot_map = {'Lauder': ax_m1, 'Hilo': ax_m2, 'Boulder': ax_m3}

for loc in [ 'Lauder', 'Boulder', 'Hilo']: 
            loc_data = []
            ax = ax_map[loc]    
            count = 0
            
            for press in model_press_levels:
                

            
                df_p0_s = df_p0[df_p0['Year'] < 2027]
                h2o_ser  = get_processed_series1(df_p0_s , loc, press, 1, 0, varname='logH2O')
                temp_ser = get_processed_series1(df_p0_s , loc, press, 1, 0, varname='Temp')
                temp_abs_ser = get_processed_series1(df_p0_s , loc, press, 1, 0, varname='Tcorr_degC')
                h2so4_ser= get_processed_series1(df_p0_s , loc, press, 1, 0, varname='H2SO4')  
                h2so4_abs_ser = get_processed_series1(df_p0_s , loc, press, 0, 0, varname='H2SO4_abs') #does not remove mean or seasonal cycle with 0 instead of 1 (numeric argument after "press")
                press_ser= get_processed_series1(df_p0_s , loc, press, 1, 0, varname='logPress') 
                
                #extract pre-2022 pure historical background means
                baseline_idx = h2so4_abs_ser.index[h2so4_abs_ser.index < '2022-01-01']
                mean_h2so4_baseline = h2so4_abs_ser.loc[baseline_idx].mean()
                mean_temp_baseline = temp_abs_ser.loc[baseline_idx].mean()
                
                #add long-term baseline means back to the seasonal anomalies
                h2so4_mean_added = h2so4_ser + mean_h2so4_baseline
                temp_mean_added = temp_ser + mean_temp_baseline
                
                # Create the temporary DF for this specific level
                df_temp = pd.DataFrame({
                    'H2SO4_wt_abs': h2so4_abs_ser,
                    'H2SO4_wt': h2so4_ser,
                    'H2SO4_wt_mean_added': h2so4_mean_added,
                    'Temp': temp_ser,
                    'Temp_mean_added': temp_mean_added,
                    'logH2O': h2o_ser,
                    'logPress': press_ser, # The continuous variable
                    'Location': loc,
                    'Pressure_Lev': press,
                    'Date': h2so4_abs_ser.index
                    }).dropna()
    
                loc_data.append(df_temp)

            #contains the full 2004-2026 dataset
            df_master = pd.concat(loc_data, ignore_index=True)
            
            #split before and after Hunga eruption
            df_master_pre = df_master[df_master['Date'] < '2022-01-01'].copy()
            df_master_post = df_master[(df_master['Date'] >= '2022-01-01') & (df_master['Date'] <= '2026-12-31')].copy()
            
            #cross-validation folds only on the pre-2022 baseline data
            np.random.seed(10) 
            k = 5 
            n_pre = len(df_master_pre)
            cv_folds = np.random.permutation(np.tile(np.arange(k), int(np.ceil(n_pre/k))))[:n_pre]
            
            mspe2, mspe1, mspe0, mspe_a, mspe_b = [], [], [], [], []
            mspe2_pre, mspe0_pre, mspe1_pre, mspe_a_pre, mspe_b_pre = [], [], [], [], []
            mspe2_post, mspe0_post, mspe1_post, mspe_a_post, mspe_b_post = [], [], [], [], []
            mspe2_pre_ma, mspe1_pre_ma, mspe0_pre_ma = [], [], []
            mspe2_post_ma, mspe1_post_ma, mspe0_post_ma = [], [], []
            
            for i in range(k):
                # 80% train / 20% test split from the PRE-2022 baseline data 
                test_ind = np.where(cv_folds == i)[0]
                train_ind = np.where(cv_folds != i)[0]
                
                df_train = df_master_pre.iloc[train_ind]
                df_test_pre = df_master_pre.iloc[test_ind].copy() # This is your 20% baseline validation
                
                #train simple statistical models only on baseline data
                model_ols2 = smf.ols(formula="H2SO4_wt ~ Temp * C(Pressure_Lev, Treatment(1))", data=df_train).fit()
                model_ols1 = smf.ols(formula="H2SO4_wt ~ Temp + C(Pressure_Lev, Treatment(1))", data=df_train).fit()
                model_ols0 = smf.ols(formula="H2SO4_wt ~ Temp", data=df_train).fit()
                
                #models Trained on baseline anomaly + baseline mean (at each pressure level)
                model_ols2_ma = smf.ols(formula="H2SO4_wt_mean_added ~ Temp_mean_added * C(Pressure_Lev, Treatment(1))", data=df_train).fit()
                model_ols1_ma = smf.ols(formula="H2SO4_wt_mean_added ~ Temp_mean_added + C(Pressure_Lev, Treatment(1))", data=df_train).fit()
                model_ols0_ma = smf.ols(formula="H2SO4_wt_mean_added ~ Temp_mean_added", data=df_train).fit()
                
                #evaluate on the 20% Pre-Eruption test (partitioned remaining 20%) data
                if not df_test_pre.empty:
                    pred2_pre = model_ols2.predict(df_test_pre)
                    pred1_pre = model_ols1.predict(df_test_pre)
                    pred0_pre = model_ols0.predict(df_test_pre)
                    df_test_pre["H2SO4_wt_mss"] = df_test_pre["H2SO4_wt_abs"] - df_test_pre["H2SO4_wt"]
                    
                    mspe2_pre.append(np.mean((df_test_pre["H2SO4_wt_abs"] - (df_test_pre["H2SO4_wt_mss"] + pred2_pre))**2))
                    mspe1_pre.append(np.mean((df_test_pre["H2SO4_wt_abs"] - (df_test_pre["H2SO4_wt_mss"] + pred1_pre))**2))
                    mspe0_pre.append(np.mean((df_test_pre["H2SO4_wt_abs"] - (df_test_pre["H2SO4_wt_mss"] + pred0_pre))**2))
                    mspe_a_pre.append(np.mean((df_test_pre["H2SO4_wt_abs"] - 0.75)**2))
                    mspe_b_pre.append(np.mean((df_test_pre["H2SO4_wt_abs"] - df_test_pre["H2SO4_wt_mss"])**2))
                
                    # ─── NEW CODE: Reconstruct Absolute field from Mean-Added Prediction ───
                    pred2_pre_ma = model_ols2_ma.predict(df_test_pre)
                    pred1_pre_ma = model_ols1_ma.predict(df_test_pre)
                    pred0_pre_ma = model_ols0_ma.predict(df_test_pre)
                    
                    mspe2_pre_ma.append(np.mean((df_test_pre["H2SO4_wt_mean_added"] - pred2_pre_ma)**2))
                    mspe1_pre_ma.append(np.mean((df_test_pre["H2SO4_wt_mean_added"] - pred1_pre_ma)**2))
                    mspe0_pre_ma.append(np.mean((df_test_pre["H2SO4_wt_mean_added"] - pred0_pre_ma)**2))
                    
                # evaluate on 100% of the Post-Eruption volcanic Hunga-Tonga era data 
                if not df_master_post.empty:
                    df_test_post = df_master_post.copy() 
                    
                    pred2_post = model_ols2.predict(df_test_post)
                    pred1_post = model_ols1.predict(df_test_post)
                    pred0_post = model_ols0.predict(df_test_post)
                    df_test_post["H2SO4_wt_mss"] = df_test_post["H2SO4_wt_abs"] - df_test_post["H2SO4_wt"]
                    
                    mspe2_post.append(np.mean((df_test_post["H2SO4_wt_abs"] - (df_test_post["H2SO4_wt_mss"] + pred2_post))**2))
                    mspe1_post.append(np.mean((df_test_post["H2SO4_wt_abs"] - (df_test_post["H2SO4_wt_mss"] + pred1_post))**2))
                    mspe0_post.append(np.mean((df_test_post["H2SO4_wt_abs"] - (df_test_post["H2SO4_wt_mss"] + pred0_post))**2))
                    mspe_a_post.append(np.mean((df_test_post["H2SO4_wt_abs"] - 0.75)**2))
                    mspe_b_post.append(np.mean((df_test_post["H2SO4_wt_abs"] - df_test_post["H2SO4_wt_mss"])**2))
                    
                    #Post-Eruption 
                    pred2_post_ma = model_ols2_ma.predict(df_test_post)
                    pred1_post_ma = model_ols1_ma.predict(df_test_post) # Track Additive Post
                    pred0_post_ma = model_ols0_ma.predict(df_test_post)
                    
                    mspe2_post_ma.append(np.mean((df_test_post["H2SO4_wt_mean_added"] - pred2_post_ma)**2))
                    mspe1_post_ma.append(np.mean((df_test_post["H2SO4_wt_mean_added"] - pred1_post_ma)**2))
                    mspe0_post_ma.append(np.mean((df_test_post["H2SO4_wt_mean_added"] - pred0_post_ma)**2))
                    

            #ANOVA table (on full dataset)
            model_ols2a = smf.ols(formula="H2SO4_wt ~ Temp * C(Pressure_Lev, Treatment(1))", data=df_master_pre).fit() #C(Pressure_Lev, Treatment(1))"
            model_ols1a = smf.ols(formula="H2SO4_wt ~ Temp + C(Pressure_Lev, Treatment(1))", data=df_master_pre).fit()
            model_ols0a = smf.ols(formula="H2SO4_wt ~ Temp", data=df_master_pre).fit()
            
            #Full Dataset Mean-Added ANOVA Setup ───
            model_ols2a_ma = smf.ols(formula="H2SO4_wt_mean_added ~ Temp_mean_added * C(Pressure_Lev, Treatment(1))", data=df_master_pre).fit() 
            model_ols1a_ma = smf.ols(formula="H2SO4_wt_mean_added ~ Temp_mean_added + C(Pressure_Lev, Treatment(1))", data=df_master_pre).fit()
            model_ols0a_ma = smf.ols(formula="H2SO4_wt_mean_added ~ Temp_mean_added", data=df_master_pre).fit()
        
           
            #clean up memory
            del df_train, df_test_pre, df_test_post, df_master_pre, df_master_post, df_master
                            
            
            #MSPE FIGURE
            print(loc)
            adj_r2 = model_ols2a.rsquared_adj 
            #MSPE plot
            means = [np.mean(mspe2_pre), np.mean(mspe1_pre), np.mean(mspe0_pre), np.mean(mspe_a_pre), np.mean(mspe_b_pre)]
            stderrs = [np.std(mspe2_pre)/np.sqrt(k), np.std(mspe1_pre)/np.sqrt(k), np.std(mspe0_pre)/np.sqrt(k), np.std(mspe_a_pre)/np.sqrt(k), np.std(mspe_b_pre)/np.sqrt(k)]
            models = ["Interaction MLR", "Additive MLR", "Temp SLR", "Fixed 0.75", "Climatology"]
            bars = ax.bar(models, means, yerr = stderrs, capsize = 6, color = ['blue', 'firebrick', 'black', 'darkgray', 'purple'], alpha = 0.8)
            ax.set_yscale('log')
            if loc == 'Lauder':
                ax.set_ylabel('Mean Squared Prediction Error')
            count = count + 1
            ax.set_title(f"MSPE: {loc}", fontsize = 16)
            ax.text(0.3, 0.96, f"Interaction (P&T); Adj-R² = {adj_r2: 0.2f}", transform=ax.transAxes, ha='center', fontsize = 15, color = 'gray')
            ax1.text(1, 0.04, 'a.', fontsize=15, fontweight='bold')
            ax2.text(1, 0.04, 'b.', fontsize=15, fontweight='bold')
            ax3.text(1, 0.04, 'c.', fontsize=15, fontweight='bold')
            
            #legend based on location
            ax_era = ax_era_map[loc]
            
            #means
            means_pre = [np.mean(mspe2_pre), np.mean(mspe1_pre), np.mean(mspe0_pre),  np.mean(mspe_a_pre), np.mean(mspe_b_pre)]
            means_post = [np.mean(mspe2_post),np.mean(mspe1_post),  np.mean(mspe0_post), np.mean(mspe_a_post), np.mean(mspe_b_post)]
            
            #standard errors
            stderrs_pre = [
                np.std(mspe2_pre) / np.sqrt(k), 
                np.std(mspe1_pre) / np.sqrt(k), 
                np.std(mspe0_pre) / np.sqrt(k), 
                np.std(mspe_a_pre) / np.sqrt(k), 
                np.std(mspe_b_pre) / np.sqrt(k)
            ]
            stderrs_post = [
                np.std(mspe2_post) / np.sqrt(k), 
                np.std(mspe1_post) / np.sqrt(k), 
                np.std(mspe0_post) / np.sqrt(k), 
                np.std(mspe_a_post) / np.sqrt(k), 
                np.std(mspe_b_post) / np.sqrt(k)
            ]
            
            era_models = ["Interaction MLR",  "Additive MLR", "Temp SLR", "Fixed 0.75", "Climatology"]
            x_locs = np.arange(len(era_models))
            b_width = 0.35
            
            #set label dynamically based 
            label_pre = '2010 - 2022' if loc == 'Hilo' else '2004 - 2022'
            
            
            
            #Figure MSPE ERA
            ax_era.bar(x_locs - b_width/2, means_pre, b_width, yerr=stderrs_pre, capsize=6,
                       label=label_pre, color='gray', edgecolor='black', alpha=0.85)
            ax_era.bar(x_locs + b_width/2, means_post, b_width, yerr=stderrs_post, capsize=6,
                       label='2022 - 2026', color='#D95F02', edgecolor='black', alpha=0.85)
            
            ax_era.set_yscale('log')
            ax_era.set_xticks(x_locs)
            ax_era.set_xticklabels(era_models, fontsize=15)
            ax_era.set_title(f"MSPE Era Comparison: {loc}", fontsize=16, fontweight='bold')
            ax_era.grid(True, which="both", linestyle=":", alpha=0.5)
            ax_era.legend(loc='upper right')
            ax_era.text(0.01, 0.95, f"Interaction (P&T); Adj-R² = {adj_r2: 0.2f}", transform=ax.transAxes, ha='center', fontsize = 16, color = 'gray')
            
            ax_e1.text(0.2, 0.01, 'a.', fontsize=15, fontweight='bold')
            ax_e2.text(0.2, 0.01, 'b.', fontsize=15, fontweight='bold')
            ax_e3.text(0.2, 0.01, 'c.', fontsize=15, fontweight='bold')
            if loc == 'Lauder':
                ax_era.set_ylabel('MSPE', fontsize=16)
            
            

            ax_era_ma = ax_ma_plot_map[loc]
            
            #calculate means 
            means_pre_ma = [np.mean(mspe2_pre_ma), np.mean(mspe1_pre_ma), np.mean(mspe0_pre_ma), np.mean(mspe_a_pre), np.mean(mspe_b_pre)]
            means_post_ma = [np.mean(mspe2_post_ma), np.mean(mspe1_post_ma), np.mean(mspe0_post_ma), np.mean(mspe_a_post), np.mean(mspe_b_post)]
            
            #calculate standard errors (divided by sqrt of folds k)
            stderrs_pre_ma = [np.std(mspe2_pre_ma)/np.sqrt(k), np.std(mspe1_pre_ma)/np.sqrt(k), np.std(mspe0_pre_ma)/np.sqrt(k), np.std(mspe_a_pre) / np.sqrt(k), np.std(mspe_b_pre) / np.sqrt(k)]
            stderrs_post_ma = [np.std(mspe2_post_ma)/np.sqrt(k), np.std(mspe1_post_ma)/np.sqrt(k), np.std(mspe0_post_ma)/np.sqrt(k), np.std(mspe_a_post) / np.sqrt(k), np.std(mspe_b_post) / np.sqrt(k)]
            
            #define models 
            ma_models = ["Interaction MLR", "Additive MLR", "Temp SLR", "Fixed 0.75", "Climatology"]
            x_locs = np.arange(len(ma_models))
            b_width = 0.35
            
            #labels
            label_pre = '2010 - 2022' if loc == 'Hilo' else '2004 - 2022'
            
            #grouped bars
            ax_era_ma.bar(x_locs - b_width/2, means_pre_ma, b_width, yerr=stderrs_pre_ma, capsize=6,
                       label=f'Baseline ({label_pre})', color='darkgray', edgecolor='black', alpha=0.85)
            
            ax_era_ma.bar(x_locs + b_width/2, means_post_ma, b_width, yerr=stderrs_post_ma, capsize=6,
                       label='Post-HT Era (2022 - 2026)', color='#D95F02', edgecolor='black', alpha=0.85)
            
            #axes & other formatting
            ax_era_ma.set_yscale('log')
            ax_era_ma.set_xticks(x_locs)
            ax_era_ma.set_xticklabels(ma_models, fontsize=15)

            ax_era_ma.set_title(f"Mean-Added MSPE: {loc}", fontsize=16, fontweight='bold', pad=15)
            ax_era_ma.grid(True, which="both", linestyle="--", alpha=0.4)
            ax_era_ma.legend(loc='upper right', fontsize=16, frameon=True)
            ax_m1.text(0.2, 0.01, 'a.', fontsize=15, fontweight='bold')
            ax_m2.text(0.2, 0.01, 'b.', fontsize=15, fontweight='bold')
            ax_m3.text(0.2, 0.01, 'c.', fontsize=15, fontweight='bold')
            
            
            if loc == 'Lauder':
                ax_era_ma.set_ylabel('MSPE', fontsize=16)
            #ANOVA table for deviations
            anova_results0 = anova_lm(model_ols0a, model_ols1a)
            anova_results1 = anova_lm(model_ols0a, model_ols2a)
            
            res0 = anova_results0.iloc[1]
            res1 = anova_results1.iloc[1]
            anova_table.append({'Site': loc, 'Comparison': 'Temp SLR vs. Additive T&P ANCOVA', 'DF Difference': res0['df_diff'], 'SSE Difference': res0['ss_diff'], 'F-stat': res0['F'], 'P-value': res0['Pr(>F)']})
            anova_table.append({'Site': loc, 'Comparison': 'Temp SLR vs. Interaction T&P MLR', 'DF Difference': res1['df_diff'], 'SSE Difference': res1['ss_diff'], 'F-stat': res1['F'], 'P-value': res1['Pr(>F)']})
            anova_table_final = pd.DataFrame(anova_table) #this is table 1 in the paper
            print(model_ols2a.summary()) #this summary goes into a table in the appendix
            
            #NEW ANOVA table for added means
            anova_results0_ma = anova_lm(model_ols0a_ma, model_ols1a_ma)
            anova_results1_ma = anova_lm(model_ols0a_ma, model_ols2a_ma)
            
            print(f"\n================= MEAN-ADDED MODEL AUDIT ({loc}) =================")
            print("Comparison: Temp SLR vs. Additive Mean-Added ANCOVA")
            print(anova_results0_ma)
            print("\nComparison: Temp SLR vs. Interaction Mean-Added MLR")
            print(anova_results1_ma)
            print("==========================================================================")
            
            #Get residuals, fitted values and standardize Residuals
            resids = model_ols2a.resid
            fitted = model_ols2a.fittedvalues
            std_resid = (resids - np.mean(resids)) / np.std(resids)
            
            #Durbin Watson test to print on Diagnostics plots 
            stat = durbin_watson(resids) #printed on the residuals vs. fitted plot...looking if model assumptions are met
                
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            #Diagnostic Q-Q Plot
            stats.probplot(std_resid, dist="norm", plot=ax1)
            ax1.set_title(f"Q-Q Plot Full Model {loc}")
            ax1.grid(True, alpha=0.3)
            ax2.scatter(fitted, std_resid, alpha = 0.5)
            ax2.axhline(0, color = 'red')
            ax2.set_title(f"Fitted vs. Residuals Plot {loc}")
            ax2.set_xlabel('Fitted Values')
            ax2.set_ylabel('Standardized Residuals')
            ax2.grid(True, alpha=0.3)
            ax2.text(0.6, 0.075, f" Durbin Watson Statistic = {stat: .1f}", transform=ax2.transAxes, ha='center', fontsize = 15, color = 'gray')

plt.tight_layout()
plt.show()


mean_results = []
audit_records = []

for loc in ['Lauder', 'Hilo', 'Boulder']:
    for press in plot_press_levels:
        mask = (df_p0['Location'] == loc) & (df_p0['Pressure [hPa]'] == press)
        
        for raw_col, res_col in var_mapping.items():
            subset = df_p0[mask].dropna(subset=['Year', raw_col])
            if subset.empty:
                continue
            
            x = subset['Year'].values
            y = subset[raw_col].values
            
            trend_col = res_col.replace('Residuals', 'Trend')
            method_col = res_col.replace('Residuals', 'Method')
            
            if press in trend_levels[loc]:
                trend, stats_dict = calculate_piecewise_linear(y, x, xe=None, breakpoint=2022)
                residuals = y - trend
                label_type = "Piecewise LM Detrended"
                
                stats_dict.update({
                    'Location': loc,
                    'Pressure [hPa]': press,
                    'Variable': raw_col
                })
                audit_records.append(stats_dict)
                
            else:
                mean_val = np.mean(y)
                residuals = y - mean_val
                trend = mean_val
                label_type = "Mean Centered"
                
                mean_results.append({
                    'Location': loc,
                    'Pressure [hPa]': press,
                    'Variable': raw_col,
                    'Mean_Value': mean_val
                })
            
            df_p0.loc[subset.index, res_col] = residuals
            df_p0.loc[subset.index, trend_col] = trend
            df_p0.loc[subset.index, method_col] = label_type

df_means = pd.DataFrame(mean_results)
df_audit_table = pd.DataFrame(audit_records)

print("\n" + "="*95)
print("                    PRE-BREAKPOINT (PRE-2022) TREND AUDIT SUMMARY")
print("="*95)
column_order = ['Location', 'Pressure [hPa]', 'Variable', 'slope_raw', 'p_val_raw', 'p_val_adj', 'std_err_adj', 'pre_trend_applied']
print(df_audit_table[column_order].to_string(index=False, formatters={
    'slope_raw': '{:,.6f}'.format,
    'p_val_raw': '{:,.4f}'.format,
    'p_val_adj': '{:,.4f}'.format,
    'std_err_adj': '{:,.6f}'.format
}))
print("="*95)

#uncertainty propagation/ contour generation 
#back-calculate frost point temperature and analytical derivative
df_p0['T_fph_K'] = calculate_t_fph_vectorized(df_p0['Pw_frostpoint'].values)
df_p0['dF_dT'] = np.log(10) * df_p0['Pw_frostpoint'] * (
    (9.09718 * 273.16 / df_p0['T_fph_K']**2) + 
    (3.56654 / (df_p0['T_fph_K'] * np.log(10))) - 
    (0.876793 / 273.16)
)

#independent FPH and radiosonde errors (Hall et al., 2016 and SkySonde Processor)
sigma_T_fph = 0.511   #uncertainty in frost point temperature
df_p0['sigma_e'] = np.abs(df_p0['dF_dT']) * sigma_T_fph
df_p0['sigma_Tamb'] = 0.5  #maximum uncertainty in radiosonde temperature

#bounds of errors
T_actual_K = df_p0['Tcorr_degC'].values + 273.15
e_actual = df_p0['Pw_frostpoint'].values

T_plus  = T_actual_K + df_p0['sigma_Tamb'].values
T_minus = T_actual_K - df_p0['sigma_Tamb'].values
e_plus  = np.maximum(1e-6, e_actual + df_p0['sigma_e'].values)  
e_minus = np.maximum(1e-6, e_actual - df_p0['sigma_e'].values)

w1 = v_lookup(T_plus,  e_plus)
w2 = v_lookup(T_plus,  e_minus)
w3 = v_lookup(T_minus, e_plus)
w4 = v_lookup(T_minus, e_minus)

w_combos = np.column_stack([w1, w2, w3, w4])
df_p0['W_max'] = np.max(w_combos, axis=1)
df_p0['W_min'] = np.min(w_combos, axis=1)

#values for figures
df_p0['Unc_Positive'] = df_p0['W_max'] - df_p0['H2SO4_%wt_amb']
df_p0['Unc_Negative'] = df_p0['H2SO4_%wt_amb'] - df_p0['W_min']


df_p0.loc[df_p0['Pressure [hPa]'] < 26, 'Unc_Positive'] = np.nan
df_p0.loc[df_p0['Pressure [hPa]'] < 26, 'Unc_Negative'] = np.nan

#2D Contours
sites = ['Lauder', 'Hilo', 'Boulder']
fig_unc, axes_unc = plt.subplots(2, 3, figsize=(24, 10), sharex=True)

pos_levels = np.linspace(0.0, 0.06, 25)  
neg_levels = np.linspace(0.0, 0.08, 25)  

cmap_pos = plt.colormaps['magma']
cmap_neg = plt.colormaps['inferno']

for j, loc in enumerate(sites):
    
    #positive uncertainty (+ΔW) ───
    ax_pos = axes_unc[0, j]
    df_sub_pos = df_p0[(df_p0['Location'] == loc)].copy()
    
    if loc == 'Hilo':
        df_sub_pos = df_sub_pos[df_sub_pos['Year'] >= 2010]
        
    df_sub_pos = df_sub_pos[
        (df_sub_pos['Unc_Positive'] >= 0) & (df_sub_pos['Unc_Positive'] < 0.5)
    ].dropna(subset=['Year', 'Pressure [hPa]', 'Unc_Positive'])
    
    cf_pos = ax_pos.tricontourf(df_sub_pos['Year'], df_sub_pos['Pressure [hPa]'], df_sub_pos['Unc_Positive'], 
                                 levels=pos_levels, cmap=cmap_pos, extend='both')
    
    cbar_pos = fig_unc.colorbar(cf_pos, ax=ax_pos, pad=0.02, aspect=22)
    cbar_pos.set_label('Positive Uncertainty ($+\Delta$W)', fontsize=22)
    cbar_pos.ax.tick_params(labelsize=18)
    ax_pos.set_title(loc, fontsize=25, fontweight='bold', pad=10)
    
    
    #negative uncertainty (-ΔW) ───
    ax_neg = axes_unc[1, j]
    df_sub_neg = df_p0[(df_p0['Location'] == loc)].copy()
    
    if loc == 'Hilo':
        df_sub_neg = df_sub_neg[df_sub_neg['Year'] >= 2010]
        
    df_sub_neg = df_sub_neg[
        (df_sub_neg['Unc_Negative'] >= 0) & (df_sub_neg['Unc_Negative'] < 0.5)
    ].dropna(subset=['Year', 'Pressure [hPa]', 'Unc_Negative'])
    
    cf_neg = ax_neg.tricontourf(df_sub_neg['Year'], df_sub_neg['Pressure [hPa]'], df_sub_neg['Unc_Negative'], 
                                 levels=neg_levels, cmap=cmap_neg, extend='both')
    
    cbar_neg = fig_unc.colorbar(cf_neg, ax=ax_neg, pad=0.02, aspect=22)
    cbar_neg.set_label('Negative Uncertainty ($-\Delta$W)', fontsize=22)
    cbar_neg.ax.tick_params(labelsize=18)
    ax_neg.set_xlabel('Year', fontsize=22)


#formatting (log pressure and 5-yr intervals) 
target_ticks = [20, 40, 70, 100, 150, 250]
target_xticks = [2005, 2010, 2015, 2020, 2025] 

for row_idx in range(2):
    for col_idx in range(3):
        ax = axes_unc[row_idx, col_idx]
        
        #log scale profiles
        ax.set_yscale('log')
        ax.set_ylim([250, 20])  
        ax.set_yticks(target_ticks)
        
        #vertical axes
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())  
        ax.set_yticklabels([str(p) for p in target_ticks], fontsize=20)
        
        #5-year axes ticks
        ax.set_xlim([2004, 2026.5]) 
        ax.set_xticks(target_xticks)
        ax.set_xticklabels([str(year) for year in target_xticks], fontsize=20)
        
        ax.grid(True, which='both', linestyle=':', alpha=0.4, color='gray')
        
        if col_idx == 0:
            ax.set_ylabel('Pressure [hPa]', fontsize=20)
            ax.text(2005, 25, 'a.', fontsize=22, fontweight='bold')
        elif col_idx == 1:
            ax.text(2005, 25, 'b.', fontsize=22, fontweight='bold')
        else:
            ax.text(2005, 25, 'c.', fontsize=22, fontweight='bold')

fig_unc.tight_layout()
fig_unc.savefig('/Users/asher/H2SO4_Asymmetric_Uncertainty_Profiles.png', dpi=400, bbox_inches='tight')
plt.show()