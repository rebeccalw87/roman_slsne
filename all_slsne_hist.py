## counts the number of SLSNe detectable at each redshift in at least one of the eight Roman filters with a magnitude ≤ 28.0
## checks all eight Roman filters, and a SLSN is counted if it is detectable in at least one of them

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from slsne.utils import get_params, get_lc
from astropy.cosmology import default_cosmology as cosmo

cosmo = cosmo.get()

## detection limit
mag_limit = 28.0

## redshift grid
redshifts = np.arange(0.1, 10.01, 0.1)

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

def simulate_roman(name, z):

    params = get_params(name)
    lc = get_lc(name, lc_type = 'rest')

    peak_mjd = params.meta['Peak']

    ## keeping only light curve points with valid magnitudes
    valid_lc = lc[np.isfinite(lc['Mean'])]

    ## converting light curve columns to usable arrays 
    phases = np.array(valid_lc['MJD'] - peak_mjd) / (1 + z)
    wavelengths = np.array(valid_lc['Cenwave'])
    m_ab = np.array(valid_lc['Mean'])

    ## ab mag to flux density
    f_jy = 3631 * 10 ** (-m_ab / 2.5)
    fluxes = f_jy * 1e-23

    ## calculating the simulated light curves
    roman_mags = {band: [] for band in roman_filters}

    for phase in np.sort(np.unique(phases)):

        ## choosing data within 0.1 days of this phase
        phase_mask = np.abs(phases - phase) < 0.1

        ## need at least 2 points to construct an SED
        if np.sum(phase_mask) < 2:
            continue

        ## applying phase choice and adjusting wavelength to chosen z
        wavelength_choice = wavelengths[phase_mask] * (1 + z)
        flux_choice = fluxes[phase_mask]

        ## sorting SED by wavelength for interpolation
        sort = np.argsort(wavelength_choice)
        wavelength_choice = wavelength_choice[sort]
        flux_choice = flux_choice[sort]

        ## removing duplicate wavelengths; interp has to be properly ordered 
        wavelength_choice, unique_idx = np.unique(wavelength_choice, return_index = True)
        flux_choice = flux_choice[unique_idx]

        ## need at least 2 wavelength/flux points to interpolate between them
        if len(wavelength_choice) < 2:
            continue

        ## interpolating SED
        sed_interp = interp1d(wavelength_choice, flux_choice, bounds_error = False, fill_value = 0)

        ## calculating magnitude in each roman band
        for band, path in roman_filters.items():

            wave_eff, area_eff = np.loadtxt(path, unpack = True)

            ## evaluating SED across roman filter transmission curve
            flux_band = sed_interp(wave_eff)

            num = np.trapezoid(flux_band * area_eff, wave_eff)
            den = np.trapezoid(area_eff, wave_eff)
            
            ## num = SLSN flux weighted by how much light the filter allows through
            ## den = total amount of light the filter allows through
            ## num / den = average SLSN flux measured through the filter

            if den <= 0 or num <= 0:
                continue

            ## calculating the filter-weighted average flux density of the slsn through the roman filter
            flux_roman = num / den

            ## converting the filter-weighted flux density to ab magnitude
            mag_roman = -2.5 * np.log10(flux_roman) - 48.6

            ## converting to apparent magnitude at chosen redshift
            mag_roman = mag_roman + cosmo.distmod(z).value - 2.5 * np.log10(1 + z)

            roman_mags[band].append(mag_roman)

    return roman_mags


## retrieving full catalog
catalog = get_params()

## counting detected slsne at each redshift
detected_counts = []

for z in redshifts:

    print(f'Processing z = {z:.1f}')

    count = 0

    for name in catalog['name']:
        
        try:
            roman_mags = simulate_roman(name, z)

            detected = False

            for band in roman_filters:

                if np.any(np.array(roman_mags[band]) <= mag_limit):
                    detected = True
                    break

            if detected:
                count += 1
                
        ## catch errors for individual slsne so the rest of the catalog can continue processing
        except Exception as e:
            print(f'Error processing {name} at z = {z:.1f}: {e}')

    detected_counts.append(count)

    print(f'Detected: {count} / {len(catalog)}')


## plotting the histogram
plt.figure(figsize = (10, 6))

plt.plot(redshifts, detected_counts, marker = 'o')

plt.xlabel('Redshift')
plt.ylabel('Number of SLSNe Detected')
plt.title('Number of SLSNe Detected by Roman vs Redshift')

plt.grid(True, alpha = 0.3)

plt.tight_layout()
plt.show()
