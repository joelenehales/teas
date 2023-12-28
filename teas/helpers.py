import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import UnivariateSpline


def find_nearest(array, value):
    """ Find closest value to the given value an array.
        Return the value and its index in the array. """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    closest_value = array[idx]

    return (closest_value, idx)


def find_overlap(array_one,array_two):
    """ Finds the overlapping range of two arrays. """

    # Define minimum and maximum values of each array
    min_one = min(array_one); max_one = max(array_one)
    min_two = min(array_two); max_two = max(array_two)

    # Determine overlapping range
    if np.logical_and(min_one <= min_two, min_two <= max_one):

        if np.logical_and(min_one <= max_two, max_two <= max_one):  # Array two entirely within array one
            overlap = (min_two, max_two)
        else:
            overlap = (min_two, max_one)

    else:

        if np.logical_and(min_two <= max_one, max_one <= max_two):  # Array two entirely within array one
            overlap = (min_one, max_one)
        else:
            overlap = (min_one, max_two)

    return overlap


def get_resolution_curve(file,plot=False):
    """ Opens the FITS file containing the tabulated data for an instrument's resolution as a function
    of wavelength and interpolates a smooth curve.

    NOTE: Resolution curve FITS files for NIRSpec can be downloaded here:
    https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters

    Parameters
    ----------
    file : str
        Directory of the FITS file with the tabulated data for the instrument's
        resolution as a function of wavelength
    plot : bool (default: False)
        To create a basic plot showing the curve fit

    Returns
    -------
    resolution_curve : class 'scipy.interpolate.UnivariateSpline'
        Interpolated resolution curve as a function of wavelength. Can be evaluated to find the
        instrument's resolution at any wavelength.

    NOTE: Currently only tested for some NIRSpec resolution curves    
    
    """

    # Open file with the tabulated resolution data
    table = fits.open(file)[1]  # In the form of a list of tuples: (wavelength,dlds,resolution)

    # Unpack data in the table
    wavelength = [row[0] for row in table.data]   # Wavelength is first column of table
    resolution = [row[2] for row in table.data]   # Resolution is third column of table
        
    # Interpolate function to the tabulated resolution data
    resolution_curve = UnivariateSpline(wavelength,resolution, s=0.5)  # Smaller smoothing factor (s) results in a closer fit to the tabulated data

    if plot:  # Create plot of curve fit
        matplotlib.rcParams.update({'font.size': 22})
        ax = plt.figure(figsize=(12, 10)).add_subplot()

        ax.scatter(wavelength,resolution,s=5,c="k",label="Tabulated Data",zorder=1)
        ax.plot(wavelength,resolution_curve(wavelength),c="red",label="Interpolated Function",zorder=2)
        ax.legend()

    return resolution_curve


def remove_lines(spectral_axis,fluxes,flux_err,line_list,spec_resolution_file):
    """ Removes narrow lines from the spectrum data.

    Parameters
    ----------
    spectral_axis : array-like
        Spectral axis (wavelengths) of spectrum
    fluxes : array-like
        Spectrum flux (each flux data point must have a corresponding wavelength
        in the spectral axis)
    flux_err : array-like (optional)
        Spectrum flux errors (each error value must have a corresponding data point in the flux array)
    line_list : one-dimensional array
        Array of wavelengths of narrow emission lines.
    spec_resolution_file : str
        Directory of the FITS file with the tabulated data for the
        instrument'S resolution as a function of wavelength

    Returns
    -------
    new_wave : array-like
        Spectral axis after removing narrow lines.
    new_flux : array-like
        Spectrum flux after removing narrow lines.
    new_flux_err : array-like
        Errors corresponding to the spectrum flux points after removing
        narrow lines

    """
    
    resolution_curve = get_resolution_curve(spec_resolution_file)  # Instrument's resolution curve as a function of wavelength

    line_inds = [] # Indices of data points corresponding to narrow lines in the spectrum
    for i in range(len(line_list)):

        line = line_list[i]
        spec_resolution = resolution_curve(line)  # Spectral resolution at the line's wavelength

        # Determine indices of narrow lines in the spectrum's data
        inds = [ind for ind,wave in enumerate(spectral_axis) if (np.logical_and((line - line/spec_resolution) <= wave, wave <= (line + line/spec_resolution)))]

        for i in inds:
            line_inds.append(i)

    # Create arrays of wavelengths/fluxes with lines removed
    new_wave = [wave for ind,wave in enumerate(spectral_axis) if ind not in line_inds]
    new_flux = [flux for ind,flux in enumerate(fluxes) if ind not in line_inds]
    new_flux_err = [err for ind,err in enumerate(flux_err) if ind not in line_inds]

    return new_wave,new_flux,new_flux_err