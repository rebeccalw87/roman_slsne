## looping through the slsn's phases and calculating the Roman magnitude at each phase for all 8 filters 
## at different chosen redshifts

import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from slsne.utils import get_params, get_lc, plot_colors
from astropy.cosmology import default_cosmology as cosmo

#name = '2019aamp'
name = '2020aup'

params = get_params(name)
lc = get_lc(name, lc_type = 'rest')

cosmo = cosmo.get()

peak_mjd = params.meta['Peak']

valid_lc = lc[np.isfinite(lc['Mean'])]

wavelengths = np.array(valid_lc['Cenwave'])
m_ab = np.array(valid_lc['Mean'])

## ab mag to flux density
f_jy = 3631 * 10 ** (-m_ab / 2.5)
fluxes = f_jy * 1e-23

## roman filters
roman_filters = {
    'F062': '/Users/rebeccawisenbaker/Desktop/Roman_WFI.F062.dat',
    'F087': '/Users/rebeccawisenbaker/Desktop/Roman_WFI.F087.dat',
    'F106': '/Users/rebeccawisenbaker/Desktop/Roman_WFI.F106.dat',
    'F129': '/Users/rebeccawisenbaker/Desktop/Roman_WFI.F129.dat',
    'F146': '/Users/rebeccawisenbaker/Desktop/Roman_WFI.F146.dat',
    'F158': '/Users/rebeccawisenbaker/Desktop/Roman_WFI.F158.dat',
    'F184': '/Users/rebeccawisenbaker/Desktop/Roman_WFI.F184.dat',
    'F213': '/Users/rebeccawisenbaker/Desktop/Roman_WFI.F213.dat'}

def simulate_roman(z):

    phases = np.array(valid_lc['MJD'] - peak_mjd) / (1 + z)

    ## calculating the simulated light curves
    roman_phases = {band: [] for band in roman_filters}
    roman_mags = {band: [] for band in roman_filters}

    for phase in np.sort(np.unique(phases)):

        ## choosing data within 0.1 days of this phase
        phase_mask = np.abs(phases - phase) < 0.1

        if np.sum(phase_mask) < 2:
            continue

        ## applying phase choice and adjusting wavelength to chosen z
        wavelength_choice = wavelengths[phase_mask] * (1 + z)
        flux_choice = fluxes[phase_mask]

        ## sorting SED
        sort = np.argsort(wavelength_choice)
        wavelength_choice = wavelength_choice[sort]
        flux_choice = flux_choice[sort]

        ## removing duplicate wavelengths
        wavelength_choice, unique_idx = np.unique(wavelength_choice, return_index = True)
        flux_choice = flux_choice[unique_idx]

        if len(wavelength_choice) < 2:
            continue

        ## interpolating SED
        sed_interp = interp1d(wavelength_choice, flux_choice, bounds_error = False, fill_value = 0)

        ## calculating magnitude in each roman band
        for band, path in roman_filters.items():

            wave_eff, area_eff = np.loadtxt(path, unpack = True)

            flux_band = sed_interp(wave_eff)

            num = np.trapezoid(flux_band * area_eff, wave_eff)
            den = np.trapezoid(area_eff, wave_eff)

            ## num = SLSN flux weighted by how much light the filter allows through
            ## den = total amount of light the filter allows through
            ## num / den = average SLSN flux measured through the filter
        
            ## SED drop-off diagnostics 
            #print('phase:', phase)
            #print('points:', np.sum(phase_mask))
            #print('wavelengths:', len(wavelength_choice), wavelength_choice.min(), wavelength_choice.max())
            #print('num:', num)
            #print('den:', den)

            if den <= 0 or num <= 0:
                continue

            ## calculating the filter-weighted average flux density of the slsn through the roman filter
            flux_roman = num / den

            mag_roman = -2.5 * np.log10(flux_roman) - 48.60

            ## converting to apparent magnitude at chosen redshift
            mag_roman = (mag_roman + cosmo.distmod(z).value - 2.5 * np.log10(1 + z))

            roman_phases[band].append(phase)
            roman_mags[band].append(mag_roman)

    ## plotting the simulated roman light curves
    fig, ax = plt.subplots(figsize = (10, 6))

    for band in roman_filters:
        ax.plot(roman_phases[band], roman_mags[band], marker = 'o', markersize = 3, color = plot_colors(band), label = band)

    ax.invert_yaxis()

    ax.set_xlabel('Phase (Rest Days)')
    ax.set_ylabel('Apparent Magnitude')
    ax.set_title(f'{name}: Simulated Roman Light Curves at z = {z}')

    ax.grid(True, alpha = 0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()

## running at the actual redshift, 1, 2, 3, 4, and 5
redshifts = [params.meta['Redshift'], 1, 2, 3, 4, 5]

for z in redshifts:
    simulate_roman(z)
