import numpy as np
import scipy.io as sio
import scipy.special as sp
import matplotlib.pyplot as plt
from glob import glob

'''
Douna objects to check SFR IMFs:
>SBS 0335-052 Prestwich 2013, uses Hunter 2010 for SFR, which assumes Salpeter
>Mrk 71, Hopkins 2002, which assumes Salpeter
>Mrk 209, Kaaret 2011, uses Calzetti 2010 for SFR, which assumes Kroupa
>UM 461 as above, assumes Kroupa
>IIZw 40, Sage 1992 , uses Thronson & Telesco 1986, which assumes Salpeter
'''

# use LaTeX for plotting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

### Constraints on Lx per SFR at various metallicities for short-lived population (<100myr) ###

IMF = 'KroupaWeidner' # target IMF used for SFR estimates in all datasets (corrections applied as needed for conversion)
salp_to_kr_SFR = 0.67 # from Madau & Dickinson 2014
kr_to_salp_SFR = 1 / salp_to_kr_SFR
salp_to_chab_SFR = 0.63 # from Madau & Dickinson 2014
chab_to_salp_SFR = 1 / salp_to_chab_SFR
salp_to_kw_SFR = 1.45
kw_to_salp_SFR = 1 / salp_to_kw_SFR
kr_to_kw_SFR = 2.16
kw_to_kr_SFR = 1 / kr_to_kw_SFR

_sfr_imf_corrections = {
    ('Salpeter', 'Kroupa'): salp_to_kr_SFR,
    ('Kroupa', 'Salpeter'): kr_to_salp_SFR,
    ('Salpeter', 'Chabrier'): salp_to_chab_SFR,
    ('Chabrier', 'Salpeter'): chab_to_salp_SFR,
    ('Kroupa', 'Chabrier'): 1.0, # difference between Kroupa and Chabrier is negligible here
    ('Chabrier', 'Kroupa'): 1.0,
    ('Salpeter', 'KroupaWeidner'): salp_to_kw_SFR,
    ('KroupaWeidner', 'Salpeter'): kw_to_salp_SFR,
    ('Kroupa', 'KroupaWeidner'): kr_to_kw_SFR,
    ('KroupaWeidner', 'Kroupa'): kw_to_kr_SFR,
    ('Chabrier', 'KroupaWeidner'): kr_to_kw_SFR,
    ('KroupaWeidner', 'Chabrier'): kw_to_kr_SFR,
}

valid_imfs = {'Salpeter', 'Kroupa', 'Chabrier', 'KroupaWeidner'}
if IMF not in valid_imfs:
    raise ValueError(f'Unknown IMF {IMF}. Valid options are {sorted(valid_imfs)}')

def sfr_imf_correction(source_imf, target_imf=IMF):
    """Multiplicative SFR correction to convert source_imf SFRs to target_imf."""
    if source_imf == target_imf:
        return 1.0
    return _sfr_imf_corrections[(source_imf, target_imf)]

def convert_sfr_imf(sfr, source_imf, target_imf=IMF):
    return sfr * sfr_imf_correction(source_imf, target_imf)

def convert_log_sfr_imf(log_sfr, source_imf, target_imf=IMF):
    return log_sfr + np.log10(sfr_imf_correction(source_imf, target_imf))

def convert_log_lx_per_sfr_imf(log_lx_per_sfr, source_imf, target_imf=IMF):
    return log_lx_per_sfr - np.log10(sfr_imf_correction(source_imf, target_imf))

# rows will be different observations, columns will be logLx_per_SFR_mean, logLx_per_SFR_max, logLx_per_SFR_min, Z_12_plus_log_O_H_mean, dataset_classifier
all_100myr_constraints = [] 

##--Lehmer 2024--##

lehmer2024_xlfs = np.genfromtxt('./lehmer2024_xlf_fits.txt',dtype=str)
logLx = lehmer2024_xlfs[:,6]
# each logLx is a string of the form 'mean+upper_error-lower_error', extract the mean and errors
logLx_mean = np.array([float(x.split('+')[0]) for x in logLx])
logLx_upper_error = np.array([float(x.split('+')[1].split('-')[0]) for x in logLx])
logLx_lower_error = np.array([float(x.split('+')[1].split('-')[1]) for x in logLx])
#logLx_max = logLx_mean + logLx_upper_error
#logLx_min = logLx_mean - logLx_lower_error

lehmer2024_galsample = np.genfromtxt('./lehmer2024_gal_sample.txt',dtype=str)
logSFR = lehmer2024_galsample[:,15] # Kroupa
# each logSFR is a string of the form 'mean+upper_error-lower_error', extract the mean and errors
logSFR_mean = np.array([float(x.split('+')[0]) for x in logSFR])
logSFR_upper_error = np.array([float(x.split('+')[1].split('-')[0]) for x in logSFR])
logSFR_lower_error = np.array([float(x.split('+')[1].split('-')[1]) for x in logSFR])
logSFR_mean = convert_log_sfr_imf(logSFR_mean, 'Kroupa')
#logSFR_max = logSFR_mean + logSFR_upper_error
#logSFR_min = logSFR_mean - logSFR_lower_error

Z_12_plus_log_O_H = (lehmer2024_galsample[:,16]).astype(float)

logMstar = lehmer2024_galsample[:,14]
# each logMstar is a string of the form 'mean+upper_error-lower_error', extract the mean and errors
logMstar_mean = np.array([float(x.split('+')[0]) for x in logMstar])
logMstar_upper_error = np.array([float(x.split('+')[1].split('-')[0]) for x in logMstar])
logMstar_lower_error = np.array([float(x.split('+')[1].split('-')[1]) for x in logMstar])
#logMstar_max = logMstar_mean + logMstar_upper_error
#logMstar_min = logMstar_mean - logMstar_lower_error

log_sSFR_mean = logSFR_mean - logMstar_mean # sSFR in yr^-1
log_sSFR_mean_Gyr = log_sSFR_mean + 9 # sSFR in Gyr^-1

# remove galaxies that are considered giant elliptical by the threshold
# sSFR = 0.01 Gyr^-1
no_elliptical_mask = log_sSFR_mean_Gyr > -2

log_Lx_per_SFR_mean = logLx_mean - logSFR_mean
log_Lx_per_SFR_upper_error = np.sqrt(logLx_upper_error**2 + logSFR_upper_error**2)
log_Lx_per_SFR_lower_error = np.sqrt(logLx_lower_error**2 + logSFR_lower_error**2)
log_Lx_per_SFR_max = log_Lx_per_SFR_mean + log_Lx_per_SFR_upper_error
log_Lx_per_SFR_min = log_Lx_per_SFR_mean - log_Lx_per_SFR_lower_error
#log_Lx_per_SFR_max = logLx_max - logSFR_min
#log_Lx_per_SFR_min = logLx_min - logSFR_max
# mask
log_Lx_per_SFR_mean = log_Lx_per_SFR_mean[no_elliptical_mask]
log_Lx_per_SFR_max = log_Lx_per_SFR_max[no_elliptical_mask]
log_Lx_per_SFR_min = log_Lx_per_SFR_min[no_elliptical_mask]
Z_12_plus_log_O_H = Z_12_plus_log_O_H[no_elliptical_mask]

# add to constraints list with data_classifier = 1 for Lehmer 2024
data_classifiers = np.ones_like(Z_12_plus_log_O_H) * 1
all_100myr_constraints.append(np.column_stack((log_Lx_per_SFR_mean, log_Lx_per_SFR_max, log_Lx_per_SFR_min, Z_12_plus_log_O_H, data_classifiers)))

##--Lehmer 2022 (stacked constraint at single metallicity)--##

lehmer2022_galsample = np.genfromtxt('./lehmer2022_gal_sample.txt',dtype=str)
Z_12_plus_log_O_H = (lehmer2022_galsample[:,-1]).astype(float)
mean_Z_12_plus_log_O_H = np.mean(Z_12_plus_log_O_H) # since they're all very similar
log_Lx_per_SFR = 40.19 # beta; equation 9 in Lehmer 2022, Kroupa
log_Lx_per_SFR_error = 0.06 # Kroupa
log_Lx_per_SFR = convert_log_lx_per_sfr_imf(log_Lx_per_SFR, 'Kroupa')
log_Lx_per_SFR_max = log_Lx_per_SFR + log_Lx_per_SFR_error
log_Lx_per_SFR_min = log_Lx_per_SFR - log_Lx_per_SFR_error

# add to constraints list with data_classifier = 2 for Lehmer 2022
data_classifier = 2
all_100myr_constraints.append(np.array([[log_Lx_per_SFR, log_Lx_per_SFR_max, log_Lx_per_SFR_min, mean_Z_12_plus_log_O_H, data_classifier]]))

##--Lehmer 2021 Supplemental Sample Upper Limits (ultralow metallicity, shallow data, limited to bright point sources, uncertain whether XRB or SMBH dominated, use with caution)--##
## Pre-corrected to Kroupa IMF I think

lehmer2021_supp_galsample_upperlims = np.genfromtxt('./lehmer2021_gal_sample_supp_upperlimits.txt',dtype=str)
Z_12_plus_log_O_H = (lehmer2021_supp_galsample_upperlims[:,14]).astype(float)
logLx = (lehmer2021_supp_galsample_upperlims[:,17]).astype(float)
starSFR = (lehmer2021_supp_galsample_upperlims[:,13]).astype(float) # Kroupa
starSFR = convert_sfr_imf(starSFR, 'Kroupa')
log_Lx_per_SFR = logLx - np.log10(starSFR)

# these are 1sigma upper limits derived from non-detections of X-ray point sources, so the uncertainty is just the upper limit value itself
# BUT we plot these without error bars for clarity

# add to constraints list with data_classifier = 3 for Lehmer 2021 supplemental upper lims
data_classifiers = np.ones_like(Z_12_plus_log_O_H) * 3
all_100myr_constraints.append(np.column_stack((log_Lx_per_SFR, log_Lx_per_SFR, log_Lx_per_SFR, Z_12_plus_log_O_H, data_classifiers)))

##--Lehmer 2021 Supplemental Sample Detections--##
## Pre-corrected to Kroupa IMF I think

lehmer2021_supp_galsample_detections = np.genfromtxt('./lehmer2021_gal_sample_supp_detections.txt',dtype=str)
Z_12_plus_log_O_H = (lehmer2021_supp_galsample_detections[:,14]).astype(float)
logLx = (lehmer2021_supp_galsample_detections[:,21]).astype(float)
logLth = (lehmer2021_supp_galsample_detections[:,17]).astype(float)
starSFR = (lehmer2021_supp_galsample_detections[:,13]).astype(float)
starSFR = convert_sfr_imf(starSFR, 'Kroupa')

log_Lx_per_SFR = logLx - np.log10(starSFR)
log_Lth_per_SFR = logLth - np.log10(starSFR)

# reconstruct 1sigma error as sqrt(Lx * Lth) per SFR (Poisson statistics)
Lx = 10**logLx
Lth = 10**logLth
Lx_per_SFR = 10**log_Lx_per_SFR
Lx_err = np.sqrt(Lx * Lth)

# add flat 25% error in star-formation rates (untabulated)
starSFR_err = 0.25 * starSFR

Lx_per_SFR_max = (Lx + Lx_err) / (starSFR - starSFR_err)
Lx_per_SFR_min = (Lx - Lx_err) / (starSFR + starSFR_err)
log_Lx_per_SFR_max = np.log10(Lx_per_SFR_max)
log_Lx_per_SFR_min = np.log10(Lx_per_SFR_min)

# add to constraints list with data_classifier = 4 for Lehmer 2021 supplemental detections
data_classifiers = np.ones_like(Z_12_plus_log_O_H) * 4
all_100myr_constraints.append(np.column_stack((log_Lx_per_SFR, log_Lx_per_SFR_max, log_Lx_per_SFR_min, Z_12_plus_log_O_H, data_classifiers)))

##--Fornasini 2020--##
## Chabrier IMF

fornasini2020_data = np.genfromtxt('./fornasini2020.txt')
Z_12_plus_log_O_H = fornasini2020_data[:,0]
log_Lx_per_SFR_mean_wrongband = fornasini2020_data[:,1] # 2-10 keV, not 0.5-8 keV, so we must convert using spectral model
log_Lx_per_SFR_error_wrongband = fornasini2020_data[:,2]
log_Lx_per_SFR_max_wrongband = log_Lx_per_SFR_mean_wrongband + log_Lx_per_SFR_error_wrongband
log_Lx_per_SFR_min_wrongband = log_Lx_per_SFR_mean_wrongband - log_Lx_per_SFR_error_wrongband
# convert to 0.5-8 keV using absorbed power law spectral model with photon index 2 and column density 10^21 cm^-2
# conversion factor is 1.56 from WebPIMMS
log_Lx_per_SFR_mean = log_Lx_per_SFR_mean_wrongband + np.log10(1.56)
log_Lx_per_SFR_max = log_Lx_per_SFR_max_wrongband + np.log10(1.56)
log_Lx_per_SFR_min = log_Lx_per_SFR_min_wrongband + np.log10(1.56)
log_Lx_per_SFR_mean = convert_log_lx_per_sfr_imf(log_Lx_per_SFR_mean, 'Chabrier')
log_Lx_per_SFR_max = convert_log_lx_per_sfr_imf(log_Lx_per_SFR_max, 'Chabrier')
log_Lx_per_SFR_min = convert_log_lx_per_sfr_imf(log_Lx_per_SFR_min, 'Chabrier')

# add to constraints list with data_classifier = 5 for Fornasini 2020
data_classifiers = np.ones_like(Z_12_plus_log_O_H) * 5
all_100myr_constraints.append(np.column_stack((log_Lx_per_SFR_mean, log_Lx_per_SFR_max, log_Lx_per_SFR_min, Z_12_plus_log_O_H, data_classifiers)))

##--Brorby 2016--##
## M12 SFRs assume Salpeter IMF
# section 8.4: error in SFR is negligible compared to error in Lx

brorby2016_data = np.genfromtxt('./brorby2016.txt')
Z_12_plus_log_O_H = brorby2016_data[:,1]
Lx_mean = brorby2016_data[:,4] # in 10^41 erg/s
Lx_error = brorby2016_data[:,5] # in 10^41 erg/s
Lx_max = Lx_mean + Lx_error
Lx_min = Lx_mean - Lx_error
log_Lx_mean = np.log10(Lx_mean * 1e41) # convert to erg/s and take log
log_Lx_max = np.log10(Lx_max * 1e41)
log_Lx_min = np.log10(Lx_min * 1e41)
v1_SFR = brorby2016_data[:,2] # in M_sun/yr
v2_SFR = brorby2016_data[:,3] # in M_sun/yr (M12 SFRs, adopted by rest of paper so we stick with these)
log_Lx_per_SFR_mean = log_Lx_mean - np.log10(v2_SFR)
log_Lx_per_SFR_max = log_Lx_max - np.log10(v2_SFR)
log_Lx_per_SFR_min = log_Lx_min - np.log10(v2_SFR)
log_Lx_per_SFR_mean = convert_log_lx_per_sfr_imf(log_Lx_per_SFR_mean, 'Salpeter')
log_Lx_per_SFR_max = convert_log_lx_per_sfr_imf(log_Lx_per_SFR_max, 'Salpeter')
log_Lx_per_SFR_min = convert_log_lx_per_sfr_imf(log_Lx_per_SFR_min, 'Salpeter')

# add to constraints list with data_classifier = 6 for Brorby 2016
data_classifiers = np.ones_like(Z_12_plus_log_O_H) * 6
all_100myr_constraints.append(np.column_stack((log_Lx_per_SFR_mean, log_Lx_per_SFR_max, log_Lx_per_SFR_min, Z_12_plus_log_O_H, data_classifiers)))

##--Garofali 2020--##
## Kroupa IMF for SFR

logMstar = 10.65 # https://iopscience.iop.org/article/10.1088/0004-637X/774/2/152
star_SFR = 38.0 # in M_sun/yr
star_SFR = convert_sfr_imf(star_SFR, 'Kroupa')
Z_12_plus_log_O_H = 8.4
log_Lx_mean = 41.50
log_Lx_error = 0.02
log_Lx_max = log_Lx_mean + log_Lx_error
log_Lx_min = log_Lx_mean - log_Lx_error
log_Lx_per_SFR_mean = log_Lx_mean - np.log10(star_SFR)
log_Lx_per_SFR_max = log_Lx_max - np.log10(star_SFR)
log_Lx_per_SFR_min = log_Lx_min - np.log10(star_SFR)

# add to constraints list with data_classifier = 7 for Garofali 2020
data_classifier = 7
all_100myr_constraints.append(np.array([[log_Lx_per_SFR_mean, log_Lx_per_SFR_max, log_Lx_per_SFR_min, Z_12_plus_log_O_H, data_classifier]]))

##--Douna 2015 upper limits (remaining samples not in any of the other datasets)--##
## These are all Salpeter

douna2015_data_upperlims = np.genfromtxt('./douna2015_upperlimits.txt')
Z_12_plus_log_O_H = douna2015_data_upperlims[:,1].astype(float)
star_SFR = douna2015_data_upperlims[:,2].astype(float)
star_SFR = convert_sfr_imf(star_SFR, 'Salpeter')
logLx = douna2015_data_upperlims[:,3].astype(float)
log_Lx_per_SFR = logLx - np.log10(star_SFR)

# add to constraints list with data_classifier = 8 for Douna 2015 upper limits
data_classifiers = np.ones_like(Z_12_plus_log_O_H) * 8
all_100myr_constraints.append(np.column_stack((log_Lx_per_SFR, log_Lx_per_SFR, log_Lx_per_SFR, Z_12_plus_log_O_H, data_classifiers)))

# these are 1sigma upper limits derived from non-detections of X-ray point sources, so the uncertainty is just the upper limit value itself
# BUT we plot these without error bars for clarity

##--Douna 2015 detections (remaining samples not in any of the other datasets)--##
## all SFRs pre-corrected to Salpeter (some I had to do manually)

douna2015_data_detections = np.genfromtxt('./douna2015_detections.txt')
Z_12_plus_log_O_H = douna2015_data_detections[:,1].astype(float)
star_SFR = douna2015_data_detections[:,2].astype(float)
star_SFR = convert_sfr_imf(star_SFR, 'Salpeter')
logLx = douna2015_data_detections[:,3].astype(float)
logLth = douna2015_data_detections[:,4].astype(float)
log_Lx_per_SFR = logLx - np.log10(star_SFR)
log_Lth_per_SFR = logLth - np.log10(star_SFR)

# reconstruct 1sigma error as sqrt(Lx * Lth) per SFR (Poisson statistics)
Lx = 10**logLx
Lth = 10**logLth
Lx_per_SFR = 10**log_Lx_per_SFR
Lx_err = np.sqrt(Lx * Lth)

# add flat 25% error in star-formation rates (untabulated)
star_SFR_err = 0.25 * star_SFR

Lx_per_SFR_max = (Lx + Lx_err) / (star_SFR - star_SFR_err)
Lx_per_SFR_min = (Lx - Lx_err) / (star_SFR + star_SFR_err)
log_Lx_per_SFR_max = np.log10(Lx_per_SFR_max)
log_Lx_per_SFR_min = np.log10(Lx_per_SFR_min)

# add to constraints list with data_classifier = 9 for Douna 2015 detections
data_classifiers = np.ones_like(Z_12_plus_log_O_H) * 9
all_100myr_constraints.append(np.column_stack((log_Lx_per_SFR, log_Lx_per_SFR_max, log_Lx_per_SFR_min, Z_12_plus_log_O_H, data_classifiers)))

##--Plot of all constraints (with error bars)--##

all_log_Lx_per_SFR_mean = np.concatenate([c[:,0] for c in all_100myr_constraints])
all_log_Lx_per_SFR_max = np.concatenate([c[:,1] for c in all_100myr_constraints])
all_log_Lx_per_SFR_min = np.concatenate([c[:,2] for c in all_100myr_constraints])
all_Z_12_plus_log_O_H = np.concatenate([c[:,3] for c in all_100myr_constraints])
all_data_classifiers = np.concatenate([c[:,4] for c in all_100myr_constraints])

# marker styles for different datasets
marker_styles = {
    1: 'o', # Lehmer 2024
    2: '*', # Lehmer 2022
    3: 'v', # Lehmer 2021 supplemental upper lims
    4: 'p',  # Lehmer 2021 supplemental detections
    5: 's', # Fornasini 2020
    6: 'D', # Brorby 2016
    7: '*',  # Garofali 2020
    8: 'v',  # Douna 2015 upper limits
    9: 'p'   # Douna 2015 detections
}
marker_sizes = {
    1: 5, # Lehmer 2024
    2: 10, # Lehmer 2022
    3: 5, # Lehmer 2021 supplemental upper lims
    4: 5, # Lehmer 2021 supplemental detections
    5: 5, # Fornasini 2020
    6: 5, # Brorby 2016
    7: 10, # Garofali 2020
    8: 5,  # Douna 2015 upper limits
    9: 5   # Douna 2015 detections
}
marker_colors = {
    1: 'black', # Lehmer 2024
    2: 'red', # Lehmer 2022
    3: 'gray', # Lehmer 2021 supplemental upper lims
    4: 'gray', # Lehmer 2021 supplemental detections
    5: 'green', # Fornasini 2020
    6: 'blue', # Brorby 2016
    7: 'orange', # Garofali 2020
    8: 'purple',  # Douna 2015 upper limits
    9: 'purple'   # Douna 2015 detections
}
plot_errorbars = {
    1: 1, # Lehmer 2024
    2: 1, # Lehmer 2022
    3: 0, # Lehmer 2021 supplemental upper limits (no error bars)
    4: 1, # Lehmer 2021 supplemental detections
    5: 1, # Fornasini 2020
    6: 1, # Brorby 2016
    7: 1, # Garofali 2020
    8: 0,  # Douna 2015 upper limits (no error bars)
    9: 1   # Douna 2015 detections
}
labels = {
    1: 'L24', # Lehmer 2024
    2: 'L22', # Lehmer 2022
    3: 'L21(ND)', # Lehmer 2021 supplemental upper lims
    4: 'L21(D)',  # Lehmer 2021 supplemental detections
    5: 'F20', # Fornasini 2020
    6: 'B16', # Brorby 2016
    7: 'G20', # Garofali 2020
    8: 'D15(ND)',  # Douna 2015 upper limits
    9: 'D15(D)'   # Douna 2015 detections
}

# Convert metallicity in 12 + log(O/H) to log(Z)
all_log_Z_over_Zsun = all_Z_12_plus_log_O_H - 8.69 # solar metallicity in 12 + log(O/H) is 8.69
all_log_Z = all_log_Z_over_Zsun + np.log10(0.0142) # solar metallicity in mass fraction is 0.0142

for data_classifier in np.unique(all_data_classifiers):
    mask = all_data_classifiers == data_classifier
    plt.errorbar(all_log_Z[mask], 
                 all_log_Lx_per_SFR_mean[mask], 
                 yerr=[all_log_Lx_per_SFR_mean[mask] - all_log_Lx_per_SFR_min[mask], 
                 all_log_Lx_per_SFR_max[mask] - all_log_Lx_per_SFR_mean[mask]], 
                 fmt=marker_styles[data_classifier], 
                 ecolor=marker_colors[data_classifier],
                 c=marker_colors[data_classifier],
                 markersize=marker_sizes[data_classifier], 
                 capsize=5*plot_errorbars[data_classifier],
                 label=labels[data_classifier])

plt.ylabel(r'$\log_{10}(L_X/\mathrm{SFR})$ [erg s$^{-1}$ / $M_\odot$ yr$^{-1}$]')
#plt.xlabel(r'$12 + \log_{10}(\mathrm{O/H})$')
plt.xlabel(r'$\log_{10}Z$')
plt.ylim(37,43)
plt.xlim(-3.5,-1.25)

# Place legend outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig(f'./Lx_per_SFR_constraints_{IMF}.png', dpi=300, bbox_inches='tight')
plt.show()

### Now we have all the observational constraints compiled, extract the z=0 values from my simulations for each model and calculate
### likelihood for each. for Lx/SFR we use Jiten's asymmetric gaussian likelihood in log10 of Lx/SFR (eq 34 in Dhandha et al. 2024a)
### later we will introduce a likelihood for the predicted/measured metallicity of the galaxies in the sample as well

# import Lx/SFR simulations for all models and extract z=0 values. interpolate over metallicity to get Lx/SFR against metallicity for each model.
# plot these against metallicity over the observations to have a look
imfstring = 'PL-0.1-100-2.35'
#imfstring = 'Kr3-canonical-2.35-100'
KeV_to_Hz = 2.417990504024e17
freq_lower_cutoff = 0.5*KeV_to_Hz
freq_upper_cutoff = 8.0*KeV_to_Hz
yr_to_s = 3.16e7
J_to_erg = 1e7

freq_bin_indices = [100, 200, 300, 350, 360, 364, 367, 370, 373, 376,
                    379, 382, 385, 388, 391, 394, 397, 400, 403, 406,
                    409, 412, 415, 418, 421, 424, 427, 430, 433, 436,
                    439, 442, 445, 448, 451, 454, 457, 460, 463, 466,
                    469, 472, 475, 478, 481, 485, 490, 500, 600, 700]

energy_data = np.loadtxt('./fine_freqs.txt') * 4.135665538536e-18  # Convert Hz to keV
energy_data = energy_data[freq_bin_indices]
freq_data = energy_data * KeV_to_Hz

integration_band = np.bitwise_and(freq_data >= freq_lower_cutoff, freq_data <= freq_upper_cutoff)

model_metallicities = np.array([0.00056,0.0018,0.005,0.008,0.016]) # absolute Z
#model_metallicities = np.array([0.005,0.008,0.016])
#model_metallicities = np.array([0.00056,0.0018]) # absolute Z
all_log_Z_model = np.log10(model_metallicities)
model_bh_prescriptions = np.array([2])
model_wind_prescriptions = np.array([1,3])
model_wind_multipliers_MS = np.array([1])
model_wind_multipliers_GB = np.array([1])
model_wind_multipliers_AGB = np.array([0.1,1,10])
model_wind_multipliers_WR = np.array([1])
model_wind_multipliers_LBV = np.array([0.1,1,10])
model_twin_fractions = np.array([0,0.1,0.2,0.3])
#model_twin_fractions = np.array([0])
#model_twin_fractions = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) # not used in modelstring but used later to scale for plotting
model_alphas_ce = np.array([0.2,1])
#model_bfs = np.linspace(0.05,0.95,200)
#model_bfs = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]) # not used in modelstring but used later to scale for plotting
model_bfs = np.array([1]) # not used in modelstring but used later to scale for plotting
modelstrings = []
modelstrings_no_bf = []


for bh_prescription in model_bh_prescriptions:
    for wind_prescription in model_wind_prescriptions:
        for wind_multiplier_MS in model_wind_multipliers_MS:
            for wind_multiplier_GB in model_wind_multipliers_GB:
                for wind_multiplier_AGB in model_wind_multipliers_AGB:
                    for wind_multiplier_WR in model_wind_multipliers_WR:
                        for wind_multiplier_LBV in model_wind_multipliers_LBV:
                            for alpha_ce in model_alphas_ce:
                                for twin_fraction in model_twin_fractions:
                                    for bf in model_bfs:
                                    
                                        if wind_multiplier_MS.is_integer():
                                            wind_multiplier_MS = int(wind_multiplier_MS)
                                        if wind_multiplier_GB.is_integer():
                                            wind_multiplier_GB = int(wind_multiplier_GB)
                                        if wind_multiplier_AGB.is_integer():
                                            wind_multiplier_AGB = int(wind_multiplier_AGB)
                                        if wind_multiplier_WR.is_integer():
                                            wind_multiplier_WR = int(wind_multiplier_WR)
                                        if wind_multiplier_LBV.is_integer():
                                            wind_multiplier_LBV = int(wind_multiplier_LBV)
                                        if twin_fraction == 0.0:
                                            twin_fraction = int(twin_fraction)
                                        if alpha_ce.is_integer():
                                            alpha_ce = int(alpha_ce)
                                        modelstring_no_bf = f'{bh_prescription}_{wind_prescription}_{wind_multiplier_MS}_{wind_multiplier_GB}_{wind_multiplier_AGB}_{wind_multiplier_WR}_{wind_multiplier_LBV}_{alpha_ce}_{twin_fraction}'
                                        modelstrings_no_bf.append(modelstring_no_bf) # will have duplicates since bf not included but this is by design
                                        modelstring = f'{modelstring_no_bf}_{bf}'
                                        modelstrings.append(modelstring)

all_models_Lx_per_SFR = np.zeros((len(modelstrings), len(model_metallicities))) # each row is a model, each column is a metallicity
print(f'Number of models: {len(modelstrings)}') # doesn't include metallicity in count

for modelstring_index, modelstring in enumerate(modelstrings):
    modelstring_no_bf = modelstrings_no_bf[modelstring_index] # corresponding modelstring without bf, for finding the file since data stored assuming bf=1
    bf = np.float64(modelstring.split('_')[-1])
    #print(f'Processing model {modelstring}...')
    this_model_Lx_per_SFR = np.zeros_like(model_metallicities)
    for Z_index, Z in enumerate(model_metallicities):
        # find the file for this model and metallicity
        try:
            redshift_evolution_data_file = f'../xrb_synthesis/Lx_per_SFR_generation/redshift_evolution_data_60_massbf/Lnu_per_SFR_attenuated_Z_{Z}_{modelstring_no_bf}_{imfstring}_100myr.mat' # bf not contained in filename because data stored assuming bf=1
            redshift_evolution_data = sio.loadmat(redshift_evolution_data_file)
        except:
            print(f'File not found for model {modelstring_no_bf} and Z {Z}, skipping...')
            this_model_Lx_per_SFR[Z_index] = np.nan
            all_models_Lx_per_SFR[modelstring_index, Z_index] = np.nan
        else:
            Lnu_per_SFR_data = 2*np.pi*redshift_evolution_data['Lnu_per_SFR_data_attenuated'].T*J_to_erg*bf # erg/s/Hz/(Msun/yr)
            # integrate over columns corresponding to 0.5-8 keV to get Lx/SFR in erg/s/(Msun/yr)
            Lx_per_SFR_data = np.trapezoid(Lnu_per_SFR_data[:,integration_band]*freq_data[integration_band], x=np.log(freq_data[integration_band])) # one value for each redshift
            # extract low z value and store
            Lx_per_SFR_zlow = Lx_per_SFR_data[-1] # runs from high z to low z
            #print(Lx_per_SFR_zlow)
            this_model_Lx_per_SFR[Z_index] = Lx_per_SFR_zlow
            all_models_Lx_per_SFR[modelstring_index, Z_index] = Lx_per_SFR_zlow

### inference time

data_classifiers_to_use_in_inference = [1,2,3,4,5,6,7,8,9] 

def asym_gauss_likelihood(log_Lx_per_SFR_pred, log_Lx_per_SFR_mean, log_Lx_per_SFR_max, log_Lx_per_SFR_min):
    # asymmetric gaussian likelihood in log10 of Lx/SFR (eq 34 in Dhandha et al. 2024a), used for the detections with error bars.
    sigma_model = 0.2 # ad-hoc model uncertainty of 0.2 dex
    sigma_upper = log_Lx_per_SFR_max - log_Lx_per_SFR_mean
    sigma_lower = log_Lx_per_SFR_mean - log_Lx_per_SFR_min
    sigma_total_upper = np.sqrt(sigma_model**2 + sigma_upper**2)
    sigma_total_lower = np.sqrt(sigma_model**2 + sigma_lower**2)
    prefactor = 1 / (sigma_total_upper + sigma_total_lower)
    if log_Lx_per_SFR_pred > log_Lx_per_SFR_mean:
        sigma_denom = sigma_total_upper
    else:
        sigma_denom = sigma_total_lower
    return prefactor * np.exp(-0.5 * ((log_Lx_per_SFR_pred - log_Lx_per_SFR_mean) / sigma_denom)**2)

'''
def upperlim_likelihood(log_Lx_per_SFR_pred, log_Lx_per_SFR_upperlim):
    # likelihood for upper limits using error function. L21 says D15 upper limits are 84% (1sigma) upper limits so we take the upper limit itself as the measurement part of the error.
    sigma_model = 0.2 # ad-hoc model uncertainty of 0.2 dex
    erf_numerator = log_Lx_per_SFR_upperlim - log_Lx_per_SFR_pred
    erf_denominator = np.sqrt(2 * (sigma_model**2 + log_Lx_per_SFR_upperlim**2))
    return 0.5 * (1 + sp.erf(erf_numerator / erf_denominator))
'''
    
def upperlim_likelihood(log_Lx_per_SFR_pred, log_Lx_per_SFR_upperlim):
    # same as previous but instead we interpret upper limits as non-detections (measured value = 0) with an error bar equal to the upper limit
    # use gaussian likelihood in log space
    sigma_model = 0.2 # ad-hoc model uncertainty of 0.2 dex
    sigma_total = np.sqrt(sigma_model**2 + log_Lx_per_SFR_upperlim**2)
    prefactor = 1 / sigma_total
    return prefactor * np.exp(-0.5 * ((log_Lx_per_SFR_pred - 0) / sigma_total)**2)

# calculate likelihood for each model against each data point
# model not calculated at the same metallicities of each data point so we interpolate the model predictions to the metallicity of each data point for likelihood calculation

# get the observational data points we want to use in inference
infmask = np.isin(all_data_classifiers, data_classifiers_to_use_in_inference)
all_log_Lx_per_SFR_mean_masked = all_log_Lx_per_SFR_mean[infmask]
all_log_Lx_per_SFR_max_masked = all_log_Lx_per_SFR_max[infmask]
all_log_Lx_per_SFR_min_masked = all_log_Lx_per_SFR_min[infmask]
all_log_Z_masked = all_log_Z[infmask]
all_data_classifiers_masked = all_data_classifiers[infmask]

#additional mask to remove data points with metallicity outside the range of the models, since we can't calculate likelihood for those
model_metallicity_min = np.min(all_log_Z_model)
model_metallicity_max = np.max(all_log_Z_model)
metallicity_mask = np.bitwise_and(all_log_Z_masked >= model_metallicity_min, all_log_Z_masked <= model_metallicity_max)
all_log_Lx_per_SFR_mean_masked_truncated = all_log_Lx_per_SFR_mean_masked[metallicity_mask]
all_log_Lx_per_SFR_max_masked_truncated = all_log_Lx_per_SFR_max_masked[metallicity_mask]
all_log_Lx_per_SFR_min_masked_truncated = all_log_Lx_per_SFR_min_masked[metallicity_mask]
all_log_Z_masked_truncated = all_log_Z_masked[metallicity_mask]
all_data_classifiers_masked_truncated = all_data_classifiers_masked[metallicity_mask]

model_likelihoods = np.zeros((len(modelstrings), len(all_log_Z_masked_truncated))) # each row is a model, each column is a data point

for modelstring_index, modelstring in enumerate(modelstrings):
    log_Lx_per_SFR_model = np.log10(all_models_Lx_per_SFR[modelstring_index])
    for data_index in range(len(all_log_Z_masked_truncated)):
        log_Lx_per_SFR_mean = all_log_Lx_per_SFR_mean_masked_truncated[data_index]
        log_Lx_per_SFR_max = all_log_Lx_per_SFR_max_masked_truncated[data_index]
        log_Lx_per_SFR_min = all_log_Lx_per_SFR_min_masked_truncated[data_index]
        log_Z = all_log_Z_masked_truncated[data_index]
        data_classifier = all_data_classifiers_masked_truncated[data_index]
        # interpolate model prediction to metallicity of this data point
        log_Lx_per_SFR_pred = np.interp(log_Z, all_log_Z_model, log_Lx_per_SFR_model)
        if data_classifier in [1,2,4,5,6,7,9]: # detections with error bars
            likelihood = asym_gauss_likelihood(log_Lx_per_SFR_pred, log_Lx_per_SFR_mean, log_Lx_per_SFR_max, log_Lx_per_SFR_min)
        elif data_classifier in [3,8]: # upper limits without error bars
            likelihood = upperlim_likelihood(log_Lx_per_SFR_pred, log_Lx_per_SFR_mean) # use mean as upper limit since no error bars
        else:
            raise ValueError(f'Unknown data classifier {data_classifier}')
        model_likelihoods[modelstring_index, data_index] = likelihood

# now we have likelihoods for each model against each data point, we can calculate total likelihood for each model by multiplying over data points (assuming independence)
total_model_likelihoods = np.prod(model_likelihoods, axis=1)

# rank models by likelihood and print the top 100 and bottom 100
log_L_over_Lmax = np.log10(total_model_likelihoods / np.max(total_model_likelihoods))
model_ranking_indices = np.argsort(total_model_likelihoods)[::-1] # sort in descending order
for rank, model_index in enumerate(model_ranking_indices):
    if rank < 100 or rank >= len(model_ranking_indices) - 100:
        print(f'Rank {rank+1}: Model {modelstrings[model_index]} with likelihood {log_L_over_Lmax[model_index]}')

'''
# print Lx/SFR against Z for top 5 models
print('Lx/SFR against Z for top 5 models:')
for rank in range(5):
    model_index = model_ranking_indices[rank]
    print(f'Rank {rank+1}: Model {modelstrings[model_index]}')
    log_Lx_per_SFR_model = np.log10(all_models_Lx_per_SFR[model_index])
    for Z, log_Lx_per_SFR in zip(all_log_Z_model, log_Lx_per_SFR_model):
        print(f'Z: {Z}, log_Lx/SFR: {log_Lx_per_SFR}')
'''

# get fragos data and calculate its likelihood with respect to the data
fragos_data = np.genfromtxt('./fragos_13.txt')
fragos_12_plus_log_O_H = fragos_data[:,0]
fragos_log_Lx_per_SFR = fragos_data[:,1]
fragos_log_Z = fragos_12_plus_log_O_H - 8.69 + np.log10(0.0142)
fragos_at_data_Z = np.interp(all_log_Z_masked, fragos_log_Z, fragos_log_Lx_per_SFR) # don't use truncated masked since fragos data covers full metallicity range, so we can calculate likelihood for all data points
fragos_likelihoods = np.zeros(len(all_log_Z_masked))
for data_index in range(len(all_log_Z_masked)):
    log_Lx_per_SFR_pred = fragos_at_data_Z[data_index]
    log_Lx_per_SFR_mean = all_log_Lx_per_SFR_mean_masked[data_index]
    log_Lx_per_SFR_max = all_log_Lx_per_SFR_max_masked[data_index]
    log_Lx_per_SFR_min = all_log_Lx_per_SFR_min_masked[data_index]
    data_classifier = all_data_classifiers_masked[data_index]
    if data_classifier in [1,2,4,5,6,7,9]: # detections with error bars
        likelihood = asym_gauss_likelihood(log_Lx_per_SFR_pred, log_Lx_per_SFR_mean, log_Lx_per_SFR_max, log_Lx_per_SFR_min)
    elif data_classifier in [3,8]: # upper limits without error bars
        likelihood = upperlim_likelihood(log_Lx_per_SFR_pred, log_Lx_per_SFR_mean) # use mean as upper limit since no error bars
    else:
        raise ValueError(f'Unknown data classifier {data_classifier}')
    fragos_likelihoods[data_index] = likelihood
total_fragos_likelihood = np.prod(fragos_likelihoods)
print(f'Fragos 2013 model likelihood: {total_fragos_likelihood}')

# plot distribution of likelihoods

'''
logL_over_Lmax = np.log10(total_model_likelihoods / np.max(total_model_likelihoods))
plt.figure()
plt.hist(logL_over_Lmax, bins=100)
# vertical line at likelihood of Fragos 2013 model
logL_fragos_over_Lmax = np.log10(total_fragos_likelihood / np.max(total_model_likelihoods))
plt.axvline(logL_fragos_over_Lmax, color='cyan', linestyle='--', label='F13(PS)')
plt.legend()
plt.xlabel(r'$\log_{10}(L/L_{max})$')
plt.ylabel('Number of models')
plt.savefig(f'./likelihood_distribution.pdf', dpi=300, bbox_inches='tight')
plt.show()
'''

# repeat constraints plot but now with model predictions overplotted
# for datasets not used in inference, add opacity to the points to indicate they were not used

plt.figure()
for data_classifier in np.unique(all_data_classifiers):
    opacity = 1.0 if data_classifier in data_classifiers_to_use_in_inference else 0.3
    mask = all_data_classifiers == data_classifier
    plt.errorbar(all_log_Z[mask], 
                all_log_Lx_per_SFR_mean[mask], 
                yerr=[all_log_Lx_per_SFR_mean[mask] - all_log_Lx_per_SFR_min[mask], 
                all_log_Lx_per_SFR_max[mask] - all_log_Lx_per_SFR_mean[mask]], 
                fmt=marker_styles[data_classifier], 
                ecolor=marker_colors[data_classifier],
                c=marker_colors[data_classifier],
                markersize=marker_sizes[data_classifier], 
                capsize=5*plot_errorbars[data_classifier],
                label=labels[data_classifier],
                zorder=10,
                alpha=opacity)

for modelstring_index, modelstring in enumerate(modelstrings):
    # if model doesn't exist (nan value) for all metallicities, skip plotting it
    if np.isnan(all_models_Lx_per_SFR[modelstring_index]).all():
        continue
    log_Lx_per_SFR_model = np.log10(all_models_Lx_per_SFR[modelstring_index])
    plt.plot(all_log_Z_model, log_Lx_per_SFR_model, color='#6ba4ff', alpha=0.05)

# take mean of the n highest likelihood models and plot as a thicker line
n_top_models = 1
top_model_indices = model_ranking_indices[:n_top_models]
top_models_Lx_per_SFR = all_models_Lx_per_SFR[top_model_indices]
top_models_log_Lx_per_SFR = np.log10(top_models_Lx_per_SFR)
mean_top_models_log_Lx_per_SFR = np.nanmean(top_models_log_Lx_per_SFR, axis=0)
if n_top_models > 1:
    plt.plot(all_log_Z_model, mean_top_models_log_Lx_per_SFR, color='C1', linewidth=3, label=f'Mean of top {n_top_models} models')
elif n_top_models == 1:
    plt.plot(all_log_Z_model, mean_top_models_log_Lx_per_SFR, color='C1', linewidth=3, label=f'Best-fit model')

# Plot PS constraint from Fragos 2013
plt.plot(fragos_log_Z, fragos_log_Lx_per_SFR, color='cyan', linestyle='--', label='F13(PS)')

plt.ylabel(r'$L_{X,0.5-8\mathrm{keV}}/\mathrm{SFR}$ [erg s$^{-1}$ Hz$^{-1}$ $M^{-1}_\odot$ yr]',fontsize=14)
#plt.xlabel(r'$12 + \log_{10}(\mathrm{O/H})$')
plt.xlabel(r'$\log_{10}Z$',fontsize=14)
plt.ylim(37,42)
plt.xlim(-3.5,-1.25)

# Enable ticks and grid
plt.tick_params(axis='both', which='both', direction='in', top=True)
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Place legend outside the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig(f'./Lx_per_SFR_constraints_with_models.pdf', dpi=300, bbox_inches='tight')
#plt.show()
