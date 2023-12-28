from .helpers import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import UnivariateSpline
from scipy import stats


class Spectrum():
    """ A class for 1D spectra.

    Attributes
    ----------
    spectral_axis : array-like
        Spectral axis (wavelengths) of spectrum
    flux : array-like
        Spectrum flux
    flux_err : array-like
        Errors on each spectrum flux data point
    continuum :  array-like
        Continuum flux
    continuum_subtracted_flux : array-like
        Continuum-subtracted spectrum flux
    aperture_ra : float
        Right ascension of the centre of the aperture the spectrum was
        extracted from, in degrees
    aperture_dec : float
        Declination of the centre of the aperture the spectrum was
        extracted from, in degrees
    aperture_radius : float
        Radius of the aperture the spectrum was extracted from, in arcseconds


    Methods
    -------
    __init__(self,spectral_axis,flux,flux_err=None,continuum_subtracted=False,aperture_ra=None,aperture_dec=None,aperture_radius=None):
        Constructor for all 1D spectrum objects.
    fit_continuum(anchors,method="spline",degree=3,find_mean=False,plot=False,xlim=(3,14),ylim=None,xtick_major_sep=1.0,xtick_minor_sep=0.2,title=None,fontsize=22,filename=None):
        Fits the continuum to the spectrum using an anchors-and-splines fitting.
    subtract_continuum():
        Subtracts the continuum flux from spectrum's flux.
    normalize_on(self,wave_min=None,wave_max=None,continuum_subtracted=False,override=False):
        Normalizes the spectrum's flux on the peak flux within the given
        wavelength range and returns a new spectrum. 
    stitch(other_spectrum):
        Simple function to stitch together the spectrum with an additional spectrum.    
    scale_to_match(match,continuum_subtracted=False,override=False)
        Scales the spectrum's flux to match another using a multiplication
        factor and returns a new spectrum.
    shift_to_match(match,continuum_subtracted=False,override=False):
        Shifts the spectrum's flux to match another by applying an offset and
        returns a new spectrum.
    scale_by(factor,continuum_subtracted=False,override=False):
        Scales the spectrum's flux by multiplying by the given scaling factor
         and returns a new spectrum.. 
    shift_by(offset,continuum_subtracted=False,override=False):
        Shifts the spectrum's flux by the given offset and returns a new spectrum..
    save(filename):
        Saves the spectrum to a file.

    """

    def __init__(self,spectral_axis,flux,flux_err=None,continuum_subtracted=False,aperture_ra=None,aperture_dec=None,aperture_radius=None):
        """ Constructor for all 1D spectrum objects.

        Parameters
        ----------
        spectral_axis : array-like
            Spectral axis (wavelengths) of spectrum
        flux : array-like
            Spectrum flux (each flux data point must have a corresponding wavelength
            in the spectral axis)
        flux_err (optional)
            Spectrum flux errors (each error value must have a corresponding
            data point in the flux array). If not given, an array with the same
            length as the flux array filled with NaN values is initialized.
        continuum_subtracted : boolean (default: False)
            If the given spectrum flux is continuum-subtracted (assumes false)
        aperture_ra : float, optional
            Right ascension of the centre of the aperture the spectrum was
            extracted from, in degrees
        aperture_dec : float, optional
            Declination of the centre of the aperture the spectrum was
            extracted from, in degrees
        aperture_radius : float, optional
            Radius of the aperture the spectrum was extracted from, in
            arcseconds

        """

        # Spectral data
        self.spectral_axis = spectral_axis   # Spectral axis (wavelength) array
        self.flux = flux                     # Spectrum flux array  
        
        # Errors on each spectrum flux data point
        if flux_err is not None:
            self.flux_err = flux_err
        else: # If not given, initialize an array of NaNs with the same length as the flux arra
            self.flux_err = np.empty(len(self.flux),dtype=float)
            self.flux_err.fill(np.nan)

        self.continuum = None  # Continuum flux

        # Continuum-subtracted spectrum flux
        if continuum_subtracted: # If the given flux is continuum-subtracted
            self.continuum_subtracted_flux = flux
        else:
            self.continuum_subtracted_flux = None

        # Check that the flux array is the same length as the spectral axis
        if len(self.spectral_axis) == len(self.flux):
            pass
        else:
            raise ValueError("Flux array must have the same length as the spectral axis.")

        # Check that the flux error array is the same length as the flux array
        if len(self.flux) == len(self.flux_err):
            pass
        else:
            raise ValueError("Flux error array must have the same length as the flux array.")

        # Aperture information
        self.aperture_ra = aperture_ra           # Right ascension of aperture centre, in degrees
        self.aperture_dec = aperture_dec         # Declination of aperture centre, in degrees
        self.aperture_radius = aperture_radius   # Aperture radius, in arcseconds


    def stitch(self,other_spectrum):
        """ Simple function to stitch together the spectrum with an additional spectrum.

        Function simply concatenates the two spectra and re-orders the arrays to
        sort the combined spectral axis in ascending order. If
        continuum-subtracted fluxes exist for both spectra, the
        continuum-subtracted fluxes are concatenated as well. Function assumes
        the two spectra are from the same region, and takes the aperture
        information from this spectrum.

        Parameters
        ----------
        other_spectrum : class 'Spectrum'
            Spectrum being stitched to the existing spectrum

        Returns
        -------
        stitched_spectrum : class 'Spectrum'
            The stitched spectrum

        """

        # Concatenate spectral axis/flux arrays of the two spectra
        combined_spec_axis = np.concatenate((self.spectral_axis,other_spectrum.spectral_axis))
        combined_flux_err = np.concatenate((self.flux_err,other_spectrum.flux_err))

        # If continuum-subtracted flux exists for both spectra, stitch them
        if self.continuum_subtracted_flux is not None and other_spectrum.continuum_subtracted_flux is not None:
            combined_flux = np.concatenate((self.continuum_subtracted_flux,other_spectrum.continuum_subtracted_flux))
            cont_subtracted = True

        else:
            combined_flux = np.concatenate((self.flux,other_spectrum.flux))
            cont_subtracted = False

        # Re-order the stitched spectral data to put the spectral axs in ascending order
        combined_spec_axis,combined_flux,combined_flux_err = zip(*sorted(zip(combined_spec_axis,combined_flux,combined_flux_err)))

        # Create new spectrum object
        stitched_spectrum = Spectrum(spectral_axis=combined_spec_axis,flux=combined_flux,flux_err=combined_flux_err,continuum_subtracted=cont_subtracted,aperture_ra=self.aperture_ra,aperture_dec=self.aperture_dec,aperture_radius=self.aperture_radius)

        return stitched_spectrum


    def fit_continuum(self,anchors,method="spline",degree=3,find_mean=False,plot=False,xlim=(3,14),ylim=None,xtick_major_sep=1.0,xtick_minor_sep=0.2,title=None,fontsize=22,filename=None):
        """ Fit the continuum flux.

        The continuum flux is determined by fitting either a linear or spline
        function with the specified degree through a list of given anchor points
        on the spectrum. Anchor points must lie within the range of the spectral
        axis, and should be chosen to not lie on any emission or absoroption features. 
        Continuum flux array will have the same length as the spectral axis. Can
        optionally create a plot of the continuum fit.

        Parameters
        ----------
        anchors : array-like
            Anchor point wavelengths, in the same units as the spectral axis.
            Wavelelengths must lie within the range of the spectrum's spectral
            axis.
        find_mean : bool (default: False)
            Find anchor point fluxes by taking the mean of spectrum flux of
            wavelengths surrounding the anchor points' wavelengths. If False,
            anchor point fluxes are obtained by evaluating the spectrum at the
            anchor points' wavelengths.
        method : "spline" or "linear" (default: "spline")
            Continuum fitting method to use
                "spline": Fits a spline function with the specified degree
                "linear": Fits a linear function using a linear regression
        degree : int (default: 3)
            Degree of spline function. Default is 3, for a cubic spline.
        plot : bool
            Create plot of the continuum fit (default is False)
        x_lim : tuple (float,float) (default: (3, 14))
            Plot x-axis limit
        y_lim : tuple (float,float) (optional)
            Plot y-axis limit
        xtick_major_sep : float (default: 1.0)
            Separation between major x-axis tick marks
        xtick_minor_sep : float (default: 0.2)
            Separation between minor x-axis tick marks
        title : str (optional)
            Plot title (if not passed, plot will have no title)
        fontsize : int (default: 22)
            Fontsize to use in plots
        filename : str, optional
            Filepath to save plot to (if not passed, plot is not saved)

        """

        # Check that all anchor points are within the range of the spectrum's spectral axis
        if max(anchors) < min(self.spectral_axis) or max(self.spectral_axis) < min(anchors):
            raise IndexError("One or more anchor points are outside of the wavelength range of the spectrum's spectral axis.")
        else:
            pass


        # Find anchor points on the spectrum
        anchor_wavelengths = []  # Initialize empty lists to hold anchor point dataanchor point
        anchor_fluxes = []

        for anchor in anchors: # Iterate over list of given anchor points

            # Find closest wavelength on spectrum to the given anchor point
            closest_wavelength,index = find_nearest(self.spectral_axis,anchor)
            anchor_wavelengths.append(closest_wavelength)

            # Find corresponding flux on spectrum
            if find_mean: # Find the mean of 3 points to the left and right of the anchor point
                surrounding_fluxes = [self.flux[index],self.flux[index-3],self.flux[index-2],self.flux[index-1],self.flux[index+1],self.flux[index+2],self.flux[index+3]]
                anchor_fluxes.append(np.mean(surrounding_fluxes))
            else: # Use the flux at the anchor point
                anchor_fluxes.append(self.flux[index])

        # Fit the continuum flux
        if method == "spline":

            # Fit spline function to the anchor points
            spline = UnivariateSpline(anchor_wavelengths, anchor_fluxes, k=degree,s=0.5) # Set smoothing factor to be small, so the spline fits the anchor points closely

            continuum = spline(self.spectral_axis)   # Evaluate function at each wavelength on the spectral axis
        
        elif method == "linear":

            # Fit a linear function to the anchor points using a least-squares regression
            regression = stats.linregress(x=anchor_wavelengths,y=anchor_fluxes)

            # Continuum fit function parameters
            m_fit = regression.slope
            b_fit = regression.intercept

            # Find continuum and continuum-subtracted fluxes in the 3um region
            continuum = [m_fit*wave+b_fit for wave in self.spectral_axis]   # Evaluate function at each wavelength on the spectral axis

        else:
            raise ValueError("Unrecognized fitting method: {}\n Valid fitting methods: \"spline\", \"linear\"".format(method))
        

        # Update attribute
        self.__dict__["continuum"] = continuum 


        # Create plot continuum fit
        if plot:

            matplotlib.rcParams.update({'font.size': fontsize})  # Set up plot
            fig = plt.figure(figsize=(20, 8))
            ax = fig.add_subplot(111)
            ax.set_title(title,fontsize=fontsize)

            # Plot data
            ax.plot(self.spectral_axis,self.flux,lw=0.75,c="k",label="Data",zorder=0)
            ax.plot(self.spectral_axis,continuum,lw=1,c="red",label="Continuum Fit",zorder=1)
            ax.scatter(anchor_wavelengths,anchor_fluxes,c="blue",label="Anchor Points",zorder=2)

            # Add x-ticks on spectral axis
            ax.xaxis.set_major_locator(ticker.MultipleLocator(xtick_major_sep))   # Major tick marks
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(xtick_minor_sep)) # Minor tick marks
            ax.tick_params(axis="both",which="major",length=6.5)  # Make ticks larger
            ax.tick_params(axis="x",which="minor",length=5)

             # Set axis limits
            if xlim is not None:
                ax.set_xlim(xlim[0],xlim[1])
            else:
                pass

            if ylim is not None:
                ax.set_ylim(ylim[0],ylim[1])
            else: # Calculate y-axis limits using lowest and highest anchor point fluxes
                median = np.median(self.flux)
                ymin = min(anchor_fluxes) - 0.2*median
                ymax = max(anchor_fluxes) + 0.2*median
                ax.set_ylim(ymin,ymax)

            # Format axis labels and legend
            ax.set_ylabel("Flux Density (MJy/sr)",fontsize=fontsize)
            ax.set_xlabel(r'Wavelength ($\mathrm{{\mu}m}$)',fontsize=fontsize)
            ax.legend(fontsize=fontsize-4,fancybox=True,facecolor="white",loc="upper left")

            if filename:  # Save figure
                fig.savefig(filename, bbox_inches="tight")

        return self


    def subtract_continuum(self):
        """ Subtract the continuum flux from spectrum's flux and update the
        object's attribute. """

        # Check that continuum exists
        if self.continuum is None:
            raise AttributeError("Spectrum continuum is not defined. Please run fit_continuum() first.")
        else:
            pass

        # Define continuum-subtracted flux array
        continuum_subtracted_flux = []
        for flux_point,continuum_point in zip(self.flux,self.continuum):  # For each flux datapoint, subtract corresponding flux on the continuum
            continuum_subtracted_flux.append(flux_point - continuum_point)

        self.__dict__["continuum_subtracted_flux"] = continuum_subtracted_flux  # Update attribute

        return self


    def normalize_on(self,wave_min=None,wave_max=None,continuum_subtracted=False,override=False):
        """ Normalizes the spectrum's flux on the peak flux within the given
        wavelength range and returns a new spectrum. 
        
        Maximum flux within the given wavelength range becomes 1 in the
        normalized flux array. If no minimum/maximum wavelength is given, the
        minimum/maximum wavelength in the spectral axis is used. If there are no
        data points within the given wavelength range, an IndexError is raised.
        The object's flux/continuum-subtracted flux attributes are not updated.
        
        Parameters
        ----------
        wave_min : float (optional)
            Minimum wavelength of range to normalize around. If not given, the
            minimum wavelength in the spectral axis is used.
        wave_max : float (optional)
            Maximum wavelength of range to normalize around. If not given, the
            maximum wavelength in the spectral axis is used.
        continuum_subtracted : boolean (default: False)
            If True, the continuum-subtracted spectrum flux is normalized.
            Otherwise, the original spectrum flux is normalized.
        override : bool (default: False)
            If true, this spectrum's flux attribute is overridden and replaced
            with the normalized flux

        Returns
        -------
        normalized_spectrum : class 'Spectrum'
            Normalized spectrum. Flux attribute is the normalized flux.
        peak : float
            Peak flux the spectrum was normalized by.

        TODO: Need to test this

        """

        # If no minimum wavelength is given, use the minimum wavelength in the spectral axis
        if wave_min is None:
            wave_min = min(self.spectral_axis)
        else:
            pass

        # If no maximum wavelength is given, use the maximum wavelength in the spectral axis
        if wave_max is None:
            wave_max = max(self.spectral_axis)
        else:
            pass

        # Define indices in data corresponding to the given range
        ind_peak = [ind for ind,wave in enumerate(self.spectral_axis) if np.logical_and(wave_min<wave,wave<wave_max)]
        
        if not ind_peak: # If no wavelengths in the spectral axis are between 11.2 - 11.3 um
            raise IndexError("No spectral axis values found between {} - {} u00B5m".format(wave_min, wave_max))
        else:
            pass
        
        # Set which flux to normalize
        if continuum_subtracted:  # Use continuum-subtracted flux

            if self.continuum_subtracted_flux is None: # Check that the continuum-subtracted flux for the object exists
                raise AttributeError("Flag to normalize the continuum-subtracted flux was set to true, but no continuum-subtracted flux exists. Please run .fit_continuum() and .subtract_continuum() first.")
            else:
                flux = self.continuum_subtracted_flux
                attribute = "continuum_subtracted_flux"  # Corresponding attribute name
        
        else:  # Use original spectrum flux
            flux = self.flux
            attribute = "flux"  # Corresponding attribute name
        
        # Determine the peak flux within the given range
        peak = max([flux[i] for i in ind_peak])
        
        # Normalize flux by dividing by the peak value
        normalized_flux = [flux_val/peak for flux_val in flux]

        if override:  # Replace the object's flux attribute with the normalized flux
            self.__dict__[attribute] = normalized_flux
        else:
            pass

        # Create new spectrum object
        normalized_spectrum = Spectrum(spectral_axis=self.spectral_axis,flux=normalized_flux,continuum_subtracted=continuum_subtracted,aperture_ra=self.aperture_ra,aperture_dec=self.aperture_dec,aperture_radius=self.aperture_radius)

        return normalized_spectrum,peak


    def scale_to_match(self,match,continuum_subtracted=False,override=False):
        """ Scales the spectrum's flux to match another using a multiplication
        factor and returns a new spectrum.

        Multiplication factor is determined by taking ratio of the medians of
        the fluxes in the region where the two spectra overlap. NaN flux values
        are removed prior to the mean calculation.

        Parameters
        ----------
        match : class 'Spectrum'
            Spectrum to scale this spectrum to match
        continuum_subtracted : bool (default: False)
            If True, the continuum-subtracted flux is scaled to match the
            continuum-subtracted flux of the other spectrum. Otherwise, the
            original spectrum flux is scaled to match the original flux of the
            other spectrum.
        override : bool (default: False)
            If true, this spectrum's flux attribute is overridden and replaced
            with the new scaled flux

        Returns
        -------
        scaled_spectrum : class 'Spectrum'
            Spectrum after being scaled by the multiplication factor. Flux
            attribute is the scaled flux.
        ratio : float
            Scaling factor used

        """

        # Determine overlapping wavelength range of the two spectra
        overlap_min,overlap_max = find_overlap(self.spectral_axis,match.spectral_axis)
        
        # Set which flux to scale
        if continuum_subtracted:  # Use continuum-subtracted flux

            # Check that the continuum-subtracted flux for both objects exists
            if self.continuum_subtracted_flux is None: 
                raise AttributeError("Flag to scale the continuum-subtracted flux was set to true, but no continuum-subtracted flux for this spectrum exists. Please run .fit_continuum() and .subtract_continuum() first.")
            elif match.continuum_subtracted_flux is None:
                raise AttributeError("Flag to scale the continuum-subtracted flux was set to true, but no continuum-subtracted flux for the given spectrum exists. Please run .fit_continuum() and .subtract_continuum() first.")
            else:
                flux = self.continuum_subtracted_flux
                flux_to_match = match.continuum_subtracted_flux
                attribute = "continuum_subtracted_flux"  # Corresponding attribute name
        
        else:  # Use original spectrum flux
            flux = self.flux
            flux_to_match = match.flux
            attribute = "flux"  # Corresponding attribute name

        # Define non-nan fluxes within the overlapping region
        self_flux_overlap = [f for wave,f in zip(self.spectral_axis,flux)
                             if (np.logical_and(overlap_min <= wave, wave <= overlap_max) and ~np.isnan(f))]
        scale_to_flux_overlap = [f for wave,f in zip(match.spectral_axis,flux_to_match)
                                 if (np.logical_and(overlap_min <= wave, wave <= overlap_max) and ~np.isnan(f))]

        # Find medians of the fluxes within the overlapping region
        mean_self = np.median(self_flux_overlap)
        mean_scale_to = np.median(scale_to_flux_overlap)

        # Take the ratio of the means and apply it as a scaling factor
        ratio = mean_scale_to/mean_self
        scaled_flux = [val*ratio for val in flux]

        if override:  # Replace the object's flux attribute with the scaled flux
            self.__dict__[attribute] = scaled_flux
        else:
            pass

        # Create new spectrum object
        scaled_spectrum = Spectrum(spectral_axis=self.spectral_axis,flux=scaled_flux,continuum_subtracted=continuum_subtracted,aperture_ra=self.aperture_ra,aperture_dec=self.aperture_dec,aperture_radius=self.aperture_radius)

        return scaled_spectrum,ratio


    def shift_to_match(self,match,continuum_subtracted=False,override=False):
        """ Shifts the spectrum's flux to match another by applying an offset
        and returns a new spectrum.

        Offset is determined by taking difference of the medians of the fluxes
        in the region where the two spectra overlap. NaN flux values are removed
        prior to the mean calculation.

        Parameters
        ----------
        match : class 'Spectrum'
            Spectrum to shift this spectrum to match
        continuum_subtracted : bool (default: False)
            If True, the continuum-subtracted flux is scaled to match the
            continuum-subtracted flux of the other spectrum. Otherwise, the
            original spectrum flux is scaled to match the original flux of the
            other spectrum.
        override : bool (default: False)
            If true, this spectrum's flux attribute is overridden and replaced
            with the new scaled flux

        Returns
        -------
        shifted_spectrum : class 'Spectrum'
            This spectrum after being shifted by the offset. Flux attribute is
            the shifted flux.
        offset : float
            Offset used

        """

        # Determine overlapping wavelength range of the two spectra
        overlap_min,overlap_max = find_overlap(self.spectral_axis,match.spectral_axis)

        # Set which flux to scale
        if continuum_subtracted:  # Use continuum-subtracted flux

            # Check that the continuum-subtracted flux for both objects exists
            if self.continuum_subtracted_flux is None: 
                raise AttributeError("Flag to scale the continuum-subtracted flux was set to true, but no continuum-subtracted flux for this spectrum exists. Please run .fit_continuum() and .subtract_continuum() first.")
            elif match.continuum_subtracted_flux is None:
                raise AttributeError("Flag to scale the continuum-subtracted flux was set to true, but no continuum-subtracted flux for the given spectrum exists. Please run .fit_continuum() and .subtract_continuum() first.")
            else:
                flux = self.continuum_subtracted_flux
                flux_to_match = match.continuum_subtracted_flux
                attribute = "continuum_subtracted_flux"  # Corresponding attribute name
        
        else:  # Use original spectrum flux
            flux = self.flux
            flux_to_match = match.flux
            attribute = "flux"  # Corresponding attribute name

        # Define non-nan fluxes within the overlapping region
        self_flux_overlap = [f for wave,f in zip(self.spectral_axis,flux)
                             if (np.logical_and(overlap_min <= wave, wave <= overlap_max) and ~np.isnan(f))]
        shift_to_flux_overlap = [f for wave,f in zip(match.spectral_axis,flux_to_match)
                                 if (np.logical_and(overlap_min <= wave, wave <= overlap_max) and ~np.isnan(f))]

        # Find medians of the fluxes within the overlapping region
        mean_self = np.median(self_flux_overlap)
        mean_shift_to = np.median(shift_to_flux_overlap)

        # Take the difference of the means and apply it as an offset
        offset = mean_shift_to - mean_self
        shifted_flux = [val + offset for val in flux]

        if override:  # Replace the object's flux attribute with the shifted flux
            self.__dict__[attribute] = shifted_flux
        else:
            pass

        # Create new spectrum object
        shifted_spectrum = Spectrum(spectral_axis=self.spectral_axis,flux=shifted_flux,continuum_subtracted=continuum_subtracted,aperture_ra=self.aperture_ra,aperture_dec=self.aperture_dec,aperture_radius=self.aperture_radius)

        return shifted_spectrum,offset


    def scale_by(self,factor,continuum_subtracted=False,override=False):
        """ Scales the spectrum's flux by multiplying by the given scaling
        factor and returns a new spectrum. 
        
        Parameters
        ----------
        factor : float
            Value to multiply the spectrum by
        continuum_subtracted : boolean (default: False)
            If True, the offset is applied to the continuum-subtracted spectrum.
            Otherwise, the offset is applied to the original spectrum flux.
        override : bool (default: False)
            If true, this spectrum's flux attribute is overridden and replaced
            with the new scaled flux

        Returns
        -------
        scaled_spectrum : class 'Spectrum'
            This spectrum after being scaled by the given factor. Flux attribute is
            the scaled flux.
        
        """

        if continuum_subtracted: # Scale the continuum-subtracted flux

            if self.continuum_subtracted_flux is None: # Check that the continuum-subtracted flux for the object exists
                raise AttributeError("Flag to offset the continuum-subtracted flux was set to true, but no continuum-subtracted flux exists. Please run .fit_continuum() and .subtract_continuum() first.")
            else:
                flux = self.continuum_subtracted_flux
                attribute = "continuum_subtracted_flux"  # Corresponding attribue name
        
        else: # Shift the original flux
            flux = self.flux
            attribute = "flux"  # Corresponding attribute name

        scaled_flux = [val * factor for val in flux]  # Scale flux

        if override:  # Replace the object's flux attribute with new scaled flux
            self.__dict__[attribute] = scaled_flux
        else:
            pass

        # Create new spectrum object
        scaled_spectrum = Spectrum(spectral_axis=self.spectral_axis,flux=scaled_flux,continuum_subtracted=continuum_subtracted,aperture_ra=self.aperture_ra,aperture_dec=self.aperture_dec,aperture_radius=self.aperture_radius)

        return scaled_spectrum


    def shift_by(self,offset,continuum_subtracted=False,override=False):
        """ Shifts the spectrum's flux by the given offset and returns a new spectrum.

        Parameters
        ----------
        offset : float
            Value to shift the spectrum by
        continuum_subtracted : boolean (default: False)
            If True, the offset is applied to the continuum-subtracted spectrum.
            Otherwise, the offset is applied to the original spectrum flux.
        override : bool (default: False)
            If true, this spectrum's flux attribute is overridden and replaced
            with the new shifted flux

        Returns
        -------
        shifted_spectrum : class 'Spectrum'
            This spectrum after being shifted by the given offset. Flux attribute is
            the shifted flux.

         """

        if continuum_subtracted: # Shift the continuum-subtracted flux

            if self.continuum_subtracted_flux is None: # Check that the continuum-subtracted flux for the object exists
                raise AttributeError("Flag to offset the continuum-subtracted flux was set to true, but no continuum-subtracted flux exists. Please run .fit_continuum() and .subtract_continuum() first.")
            else:
                flux = self.continuum_subtracted_flux
                attribute = "continuum_subtracted_flux" # Corresponding attribute name
        
        else: # Shift the original flux
            flux = self.flux
            attribute = "flux" # Corresponding attribute name

        shifted_flux = [val + offset for val in flux]  # Shift flux
        
        if override:  # Replace the object's flux attribute with new scaled flux
            self.__dict__[attribute] = shifted_flux
        else:
            pass

        # Create new spectrum object
        shifted_spectrum = Spectrum(spectral_axis=self.spectral_axis,flux=shifted_flux,continuum_subtracted=continuum_subtracted,aperture_ra=self.aperture_ra,aperture_dec=self.aperture_dec,aperture_radius=self.aperture_radius)

        return shifted_spectrum


    def save(self,filename):
        """ Saves the spectrum to a file.
        
        Parameters
        ----------
        filename: str
            Filepath to save the spectrum to. File extention must be
            either .txt or .csv.

        TODO: Has not been tested
        
         """

        if filename[-4:] == ".txt" or filename[-4:] == ".csv":  # Check that a valid filename was given
            pass
        else:
            raise IOError("Must give the filename of a text file (ending in .txt) or a CSV file (ending in .csv).")


        # If continuum or continuum-subtracted flux do not exist, create an an
        # array of NaNs with the same length as the spectral_axis
        nan_array = np.empty(len(self.spectral_axis),dtype=float).fill(np.nan)

        if self.continuum is None:
            continuum = nan_array
        else:
            continuum = self.continuum
        
        if self.continuum_subtracted_flux is None:
            continuum_subtracted_flux = nan_array
        else:
            continuum_subtracted_flux = self.continuum_subtracted_flux

        # Define function that returns the value if it is not NaN, or an empty string otherwise
        convert = lambda x : (x if ~np.isnan(x) else "")

        with open(filename, "w+") as file:
            
            # Write header
            file.write("Wavelength,Flux,FluxError,Continuum,ContinuumSubtractedFlux\n")

            for i in range(len(self.spectral_axis)):
                
                wavelength = self.spectral_axis[i]
                flux = self.flux[i]
                flux_err = self.flux_err[i]

                file.write("{},{},{},{},{}\n".format(convert(wavelength),
                                                     convert(flux),
                                                     convert(flux_err),
                                                     convert(continuum[i]),
                                                     convert(continuum_subtracted_flux[i]))) 


