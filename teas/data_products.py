from .helpers import *
from .spectra import *

import numpy as np
import pandas as pd
import copy

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates.angles import Angle
from reproject import reproject_interp

from photutils.aperture import CircularAperture
from specutils import Spectrum1D
from jdaviz import Cubeviz

import warnings

class JWST_DataProduct():
    """ The parent class for all JWST data products. Contains attributes and
    methods common to all JWST data product formats (spectal data cubes and images).

    Attributes
    ----------
    filepath : str
        Filepath to FITS file with data cube/image data.
    hdu_list : class 'astropy.io.fits.HDUList'
        List of all HDUs in the FITS file, if data was given in the form of a
        filepath.
    data_hdu : class 'astropy.io.fits.hdu.image.ImageHDU'
        Cube/image data in the form of an astropy ImageHDU object
    err_hdu : class 'astropy.io.fits.hdu.image.ImageHDU'
        Cube/image data errors in the form of an astropy ImageHDU object
    data : numpy.ndarray
        Cube/image data in the form of a numpy array
    err : numpy.ndarray
        Cube/image data errors in the form of a numpy array
    fits_header : class 'astropy.io.fits.Header'
        FITS header for the data
    wcs : class 'astropy.wcs.WCS'
        World Coordinate System (WCS) of the data
    celestial_wcs : class 'astropy.wcs.WCS'
        Celestial WCS (WCS without the spectral axis, if the data has one)
    fov_angle : float
        Position angle of the V3 reference axis eastward relative to north when projected onto the sky.
        For more information, see:
        https://jwst-docs.stsci.edu/jwst-observatory-characteristics/jwst-position-angles-ranges-and-offsets]
    data_product_dimensions : int
        Dimensions of the data product (2 for an image, 3 for a data cube)
    ra_num : int
        Number of values in the R.A. dimension of the array, from NAXIS1 in the FITS header
    dec_num : int
        Number of values in the Dec. dimension of the array, from NAXIS2 in the FITS header
    wave_num : int or None
        Number of values in the spectral axis, from NAXIS3 in the FITS header
    instrument : str
        JWST instrument that collected the data
    filter_chan : str or None
        Instrument channel/filter
    pixel_size : str
        Spatial resolution (size in arcseconds of one pixel)
    label : str, optional
        Label given to data product (default is Instrument + Filter/Channel)

    Methods
    -------
    __init__(self,filepath,filter_chan,label=None):
        Constructor for all JWST data product objects.
    convert_pix_to_arcsec(pix):
        Converts distance in pixels to arcseconds.
    convert_arcsec_to_pix(arcseconds):
        Converts distance in arcseconds to pixels.
    coord_of_pixel(x,y):
        Finds the sky coordinate of a pixel (x, y).
    pixel_of_coord(ra,dec):
        Finds the pixel coordinate (x, y) of a given sky coordinate.
    reproject_celestial(other_data_product,label=None,plot=False,slice_index=0,vmin=-1.,vmax=250.,filename=None):
        Reprojects the celestial axes of the data product onto the field of view
        of another cube/image.
    find_fov(other_cube,return_sky_coord=False)
        Find the field of view (FOV) of another cube/image within the FOV of the
        cube/image the method is called on.
    align(true_coord=None,observed_coord=None,observed_pixel=None,separation=None,angle=None,plot=False,slice_index=0,vmin=-1.,vmax=250.,filename=None):
        Aligns the data by shifting the centre coordinate in the FITs header.
        Can either directly pass an offset separation (distance) and angle
        (direction) to apply, or calculate the separation between an object's
        observed location in the data and it's true coordinate. 
    save_fits(self,filename,include_errors=True):
        Saves the data product to a FITS file.

    NOTE: Currently lacks MIRI image functionality. Would need to update the
    lists of accepted channels/filters and add their pixel sizes.

    """

    def __init__(self,filepath,filter_chan,label=None):
        """ Constructor for all JWST data product objects.

        Parameters
        ----------
        filepath : str
            Filepath to FITS file with data cube/image data.
        filter_chan : str or None
            Instrument channel/filter (not case sensitive)
        label : str
            Label given to the data product (if None is passed, defaults to the Instrument + Channel/Filter)

        """

        # Initialize attributes
        self.filepath = filepath   # Filepath to the original FITs file
        self.hdu_list = None       # All HDUs in the FITS file, if data given as a filepath
        self.data_hdu = None       # Cube/image data, as an astropy ImageHDU
        self.data = None           # Cube/image data, as a numpy array
        self.err_hdu = None        # Cube/image data errors, as an astropy ImageHDU
        self.err = None            # Cube/image data errors, as a numpy array
        self.fits_header = None    # Astropy FITS header 


        if filepath[-5:] != ".fits":  # Catch invalid FITs filename
            raise IOError("Invalid filename: {}. Must give the filename of a FITS file (ending in .fits).".format(filepath))

        else:  # If acceptable input, continue
            pass

        # Define attributes
        self.hdu_list = fits.open(filepath)
        self.data_hdu = self.hdu_list[1]
        self.data = self.data_hdu.data
        self.err_hdu = self.hdu_list[2]
        self.err = self.err_hdu.data
        self.fits_header = self.data_hdu.header

        if "MJD-OBS" not in self.fits_header:   # Observation time under this heading required for WCS
                self.fits_header["MJD-OBS"] = self.fits_header["EPH_TIME"]
        else:
            pass


        # Define World Coordinate System (WCS) attributes
        self.wcs = WCS(self.fits_header)
        self.celestial_wcs = self.wcs.celestial  # WCS not including spectral axis, if it has one


        # Check that the data array has the same shape as the error shape
        if self.data.shape != self.err.shape:
            raise ValueError("Data array and error array must be the same shape. \n  Data array shape: {}. Error array shape: {}.".format(self.data.shape,self.err.shape))
        else:
            pass

        # Define dimensions of the data cube/image
        self.data_product_dimensions = len(self.data.shape) # Number of axes (2 for an image, 3 for a cube)

        self.ra_num = self.fits_header["NAXIS1"]  # Number of values in each axis' array
        self.dec_num = self.fits_header["NAXIS2"]
        
        if self.data_product_dimensions == 3:
            self.wave_num = self.fits_header["NAXIS3"]
        elif self.data_product_dimensions == 2:
            self.wave_num = None
        else:
            raise ValueError("Unsupported data product dimensions: {}".format(self.data_product_dimensions))


        # Lists of acceptable channel/filter inputs (not case sensitive)
        nircam_shortwave_filters = ["F070W","F090W","F115W","F140M","F150W","F162M","F164N","F150W2","F187N","F182M","F020W","F210M","F212N"]  # NIRCam short wavelength filters
        nircam_longwave_filters = ["F250M","F277W","F300M","F322W2","F323N","F335M","F356W","F360M","F405N","F410M","F430M","F444W","F460M","F466N","F470N","F480M"]           # NIRCam long wavelength filters
        miri_ifu_channels = ["CHANNEL 1","CHANNEL 2","CHANNEL 3","CHANNEL 4"]  # MIRI IFUs


       # Define instrument, channel/filter, and pixel size (in arcsec) as instance variables
        if not filter_chan:  # NIRSpec has no channel/filter
            self.instrument = "NIRSpec"
            self.filter_chan = None
            self.pixel_size = 0.1

        else: # If filter_chan passed, validate input

            filter_chan_upper = filter_chan.upper() # Capitalized version of given channel/filter, for validating input

            # Check if channel/filter given is an acceptable input
            if filter_chan_upper not in nircam_shortwave_filters + nircam_longwave_filters + miri_ifu_channels:
                raise ValueError("Channel/filter entered is not recognized. Please enter one of the following:\n\nNIRCam filters: {}\n\nMIRI MRS IFUs: [\"Channel 1\", \"Channel 2\", \"Channel 3\", \"Channel 4\"]\n\n Or None for NIRSpec".format(nircam_shortwave_filters+nircam_longwave_filters))
            else:
                pass


            if filter_chan_upper in nircam_shortwave_filters + nircam_longwave_filters:
                self.instrument = "NIRCam"
                self.filter_chan = filter_chan_upper  # Keep NIRCam filters all capitalized

                if filter_chan_upper in nircam_shortwave_filters:  # NIRCam short wavelength filter
                    self.pixel_size = 0.031
                else:  # NIRCam long wavelength filter
                    self.pixel_size = 0.061

            else:  # MIRI IFU data
                self.instrument = "MIRI MRS"
                self.filter_chan = filter_chan.capitalize() # Capitalize only first letter in MIRI channels

                # Dictionary of MIRI pixel size (in arcseconds) by channel/filter
                miri_pixel_sizes_dict = {
                    "Channel 1" : 0.196,
                    "Channel 2" : 0.196,
                    "Channel 3" : 0.245,
                    "Channel 4" : 0.273}

                self.pixel_size = miri_pixel_sizes_dict[self.filter_chan]


        # Define label for data product
        if label:  # Use given label, if given
            self.label = label
        else:  # Define default label if none given (instrument + channel/filter)
            if self.filter_chan == None:
                self.label = self.instrument
            else:
                self.label = self.instrument+" "+self.filter_chan


        # Define position angle the reference axis (east of north) using instrument
        if "PA_V3" in self.fits_header:
            self.fov_angle = self.fits_header["PA_V3"]
        else:  # If it does not exist in the header, set it to 0
            self.fov_angle = 0


    # Distance conversion functions between pixels and arcseconds
    def convert_pix_to_arcsec(self,pix):
        """ Convert distance in pixels to arcseconds. """

        return pix*self.pixel_size

    def convert_arcsec_to_pix(self,arcseconds):
        """ Convert distance in arcseconds to pixels. """

        return arcseconds/self.pixel_size


    # Coordinate conversion functions between pixel coordinates and sky
    # coordinates in degrees
    def coord_of_pixel(self,x,y):
        """ Find the sky coordinate of a pixel (x, y).

        Parameters
        ----------
        x : float
            x-axis coordinate of pixel in the data cube/image
        y : float
            y-axis coordinate of pixel in the data cube/image

        Returns
        -------
        (ra,dec) : tuple of (float,float)
            Right ascension and declination of pixel, in degrees

        """

        coord = self.celestial_wcs.pixel_to_world(x,y) # Find sky coordinates of the pixel using the celestial world coordinate system from the data cube
        ra = coord.ra.to(u.deg).value  # Convert values in degrees to decimal format
        dec = coord.dec.to(u.deg).value

        return (ra,dec)


    def pixel_of_coord(self,ra,dec):
        """ Find the pixel coordinate (x, y) of a given sky coordinate.

        Parameters
        ----------
        ra : float
            Right ascension, in degrees
        dec : float
            Declination, in degrees

        Returns
        -------
        (x,y) : tuple (float,float)
            Pixel coordinate of sky coordinate

        """

        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg) # Create skycoord object from given R.A. and Dec.
        x,y = self.celestial_wcs.world_to_pixel(coord) # Find pixel of the coordinate using the celestial world coordinate system from the data cube

        return (x,y)


    def reproject_celestial(self,other_data_product,label=None,plot=False,slice_index=0,vmin=-1.,vmax=250.,filename=None):
        """ Reprojects the celestial axes of the cube/image the method is called
        on (+ its errors) onto the FOV of another cube/image.

        Only the celestial axes are reprojected. If the
        data has a spectral axis, it remains unchanged. Can optionally create a
        plot comparing the cube/image before and after the reprojection. Returns
        the data as either an SpectralDataCube()/Image() according to the
        original data's format.


        Parameters
        ----------
        other_data_product : class 'JWST_DataProduct'
            Data cube/image whose FOV to reproject onto
        label : str
            Label to use for reprojected data (if None is passed, defaults to
            Instrument + Channel/Filter + "Reprojected")
        plot : bool (default: False)
            To create a comparison plot showing the data cube before vs. after
            being reprojected and the field of view of the data product
            projected onto
        slice_index : ind (default: 0)
            Specified the index of the slice to plot, if any data products are a
            data cube
        vmin : float
            Data value to map the minimumn colourmap value to (default is -1.0)
        vmax : float
            Data value to map the maximum colourmap value (default is 250.0)
        filename : str (optional)
            Filepath to save the comparison plot to (if not passed, plot is not saved)

        Returns
        -------
        reprojected_object : class 'SpectralDataCube' or class 'Image()'
            Data with reprojected celestial axes

        """

        if label is None: # If label for reprojected data not given, create default label
            label = self.label+", "+"Reprojected"
        else:
            pass


        # Check dimensions of data product being projected
        if self.data_product_dimensions == 2:  # Image

            # Create copy of object
            reprojected_object = Image(filepath=self.filepath,filt=self.filter_chan,label=self.label)

            # Define WCS to reproject onto as the celestial WCS of the other data product
            reproject_header = other_data_product.fits_header
            reproject_wcs = other_data_product.celestial_wcs

            # Reproject image data onto this WCS
            reprojected_data = reproject_interp(self.data_hdu,reproject_wcs,reproject_wcs.array_shape,return_footprint=False)


        else:  # Data cube

            # Create copy of object
            reprojected_object = SpectralDataCube(filepath=self.filepath,channel=self.filter_chan,wavelengths=self.spectral_axis,label=self.label)

            # Need to define a WCS with celestial axes of cube/image being projected onto and spectral axis of cube being projected. Project onto this WCS.

            # Check dimensions of data product being projected onto
            if other_data_product.data_product_dimensions == 2: # Data cube onto image

                # Copy header of data cube being projected
                reproject_header = copy.deepcopy(self.fits_header)

                # Replace celestial (R.A. and Dec.) axes (1 and 2) with those of the image
                # being projected onto
                for ax in [1,2]:
                    headings_replace = ["CTYPE{}".format(ax),"NAXIS{}".format(ax),"CDELT{}".format(ax),"CRVAL{}".format(ax)]

                    for heading in headings_replace:
                        reproject_header[heading] = other_data_product.fits_header[heading]

            else: # Data cube onto data cube

                # Copy header of data cube projecting onto
                reproject_header = copy.deepcopy(other_data_product.fits_header)

                # Replace spectral axis with that of the data cube being projected
                for heading in ["CTYPE3","NAXIS3","CDELT3","CRVAL3"]:
                    reproject_header[heading] = self.fits_header[heading]


            # Define this header as a WCS
            reproject_wcs = WCS(reproject_header)

            # Reproject data cube onto this WCS
            reprojected_data = reproject_interp(self.data_hdu,reproject_wcs,reproject_wcs.array_shape, return_footprint=False)


        # Set attributes to reprojected data
        reprojected_object.data_hdu.data = reprojected_data
        reprojected_object.data = reprojected_data
        reprojected_object.fits_header = reproject_header
        reprojected_object.wcs = reproject_wcs
        reprojected_object.celestial_wcs = reproject_wcs.celestial

        # Update header in each HDU to use thereprojected WCS
        hdu_list = reprojected_object.hdu_list

        for i in range(len(hdu_list)):
            hdu = hdu_list[i]

            for heading in ["CTYPE1","NAXIS1","CDELT1","CRVAL1", # List of headings to change
                            "CTYPE2","NAXIS2","CDELT2","CRVAL2",
                            "CTYPE3","NAXIS3","CDELT3","CRVAL3"]:
                if heading in hdu.header:
                    hdu.header[heading] = reproject_header[heading]


        if plot:

            fig = plt.figure(figsize=(21,8))  # Set up plot
            gs = fig.add_gridspec(nrows=1,ncols=3,width_ratios=[1,1,1],wspace=0.45)
            ax_1 = fig.add_subplot(gs[0,0],projection=self.celestial_wcs)  # Original data
            ax_2 = fig.add_subplot(gs[0,1],projection=reprojected_object.celestial_wcs) # Reprojected 
            ax_3 = fig.add_subplot(gs[0,2],projection=other_data_product.celestial_wcs) # FOV projected onto
            fontsize = 20
            cmap = "viridis"

            # Define data to be plotted
            if self.data_product_dimensions == 2:  # Image
                before = self.image
                after = reprojected_object.image
            else:  # Data cube
                before = self.data[slice_index,:,:]
                after = reprojected_object.data[slice_index,:,:]
            
            if other_data_product.data_product_dimensions == 2:  # Image
                fov_onto = other_data_product.image
            else:  # Data cube
                fov_onto = other_data_product[slice_index,:,:]
                

            # Plot data
            ax_1.set_title("Original Data",fontsize=fontsize) # Original data
            ax_1.imshow(before, origin='lower', cmap=cmap,vmin=vmin, vmax=vmax)

            ax_2.set_title("Reprojected Data",fontsize=fontsize) # Reprojected data
            ax_2.imshow(after, origin='lower', cmap=cmap,vmin=vmin, vmax=vmax)

            ax_3.set_title("FOV Reprojected Onto",fontsize=fontsize) # FOV projected onto
            ax_3.imshow(fov_onto, origin='lower', cmap=cmap,vmin=vmin, vmax=vmax)


            # Set axis labels
            for ax in [ax_1,ax_2,ax_3]:
                ax.set_xlabel("R.A.", fontsize=fontsize)
                ax.set_ylabel("Dec.",fontsize=fontsize)

                ra = ax.coords[0]
                dec = ax.coords[1]

                ra.set_format_unit(u.deg,decimal=True,show_decimal_unit=True)
                ra.set_ticklabel(exclude_overlapping=True,size=fontsize-2)
                ra.set_major_formatter('d.ddd')
                ra.set_ticks_position('b')   # Set ticks on only bottom, not top

                dec.set_format_unit(u.deg,decimal=True,show_decimal_unit=True)
                dec.set_ticklabel(exclude_overlapping=True,size=fontsize-2)
                dec.set_major_formatter('d.dddd')
                dec.set_ticks_position('l')  # Set ticks on only left, not right


            if filename:  # Save figure
                fig.savefig(filename, bbox_inches="tight")

        return reprojected_object


    def find_fov(self,other_cube,return_sky_coord=False):
            """ Find the field of view (FOV) of another cube/image within the FOV of
            the cube/image the method is called on.

            Method returns the locations of the four corners of the other cube's FOV, in
            either pixel or sky coordinates. Field of view can then be plotted
            by passing the corner locations to matplotlib.patches.Polygon

            Parameters
            ----------
            other_cube : class 'JWST_DataProduct'
                Data cube/image whose FOV to find
            return_sky_coord : bool (default: False)
                If True, the sky coordinates of the corners are 

            Returns
            -------
            corners: np.ndarray with shape (4, 2)
                Locations of the four corners of the data cube/image's FOV

            NOTE: This has only been tested with finding the FOV of MIRI MRS
            channels within NIRCam images

            """

            # Find coordinate of the four corners of the other data cube/image's
            # field of view
            corners_sky = other_cube.celestial_wcs.calc_footprint()

            # Determine which coordinates to return the corners' location in
            # (sky or pixel)
            if return_sky_coord:  # Return sky coordinates
                corners = corners_sky
            
            else: # Return pixel location

                corners_pixel = []  # Initialize array to hold the pixel locations of each corner
                
                # Find corresponding pixels in the cube/image the method is called on
                for corner in corners_sky:
                    ra,dec = corner
                    x,y = self.pixel_of_coord(ra,dec)
                    corners_pixel.append(np.array([x,y]))

                corners = np.array(corners_pixel)

            return corners
            

    def align(self,true_coord=None,observed_coord=None,observed_pixel=None,separation=None,angle=None,label=None,plot=False,slice_index=0,vmin=-1.,vmax=250.,filename=None):
        """ Aligns the data by shifting the centre coordinate in the FITs header.
        
        Amount to shift centre coordinate can be specified in one of two ways:
          1. By calculating the separation between an object in the data's
              observed location and it's true coordinate (ie. from a
              catalog like SIMBAD), then shifting the centre coordinate in the header
              by that amount. Observed coordinate may be passed as either a pixel
              coordinate or a sky coordinate.
          2. By applying a given offset, given by a separation (distance) and an
             angle (direction).

        Parameters
        ----------
        true_coord : tuple of (float,float)
            True location of the object in degrees (R.A., Dec.).
        observed_coord : tuple of (float,float)
            Observed location of the object in degrees (R.A., Dec.). If not passed, must pass the
            location as a pixel coordinate instead.
        observed_pixel : tuple of (float, float)
            Observed location of the object in pixel coordinates (x, y). If not
            passed, must pass the location as a sky coordinate instead.
        separation : class 'astropy.coordinates.angles.Angle'
            Distance to shift the centre of the data cube
        angle : class 'astropy.coordinates.angles.Angle'
            Direction in which to shift the centre of the data cube
        label : str
            Label to use for reprojected data (if None is passed, defaults to
            Instrument + Channel/Filter + "Aligned")
        plot : bool (default: False)
            To create a comparison plot showing the data cube before vs. after
            being aligned
        slice_index : ind (default: 0)
            Specified the index of the slice to plot, if the data product is a
            data cube
        vmin : float
            Data value to map the minimumn colourmap value to (default is -1.0)
        vmax : float
            Data value to map the maximum colourmap value (default is 250.0)
        filename : str (optional)
            Filepath to save the comparison plot to (if not passed, plot is not saved)

        Returns
        -------
        shifted_data : class 'SpectralDataCube' or class 'Image()'
            Data with shifted centre coordinate in the FITS header.
        
        """


        # If true and observed coordinates are given, calculate separation
        if true_coord is not None and (observed_coord is not None or observed_pixel is not None):

            if observed_pixel is not None and observed_coord is not None:
                raise ValueError("Can accept the observed location of the object in the data as either a pixel or sky coordinate, but not both.")
            else:
                pass


            # If a pixel coordinate is given, convert it to a sky coordinate
            if observed_pixel is not None:
                x,y = observed_pixel
                ra_obs,dec_obs = self.coord_of_pixel(x,y)
                
            else:
                ra_obs,dec_obs = observed_coord
            
            ra_true,dec_true = true_coord

            # Convert all coordinates to astropy sky coordinate objects
            true_coord = SkyCoord(ra=ra_true*u.deg, dec=dec_true*u.deg)
            observed_coord = SkyCoord(ra=ra_obs*u.deg, dec=dec_obs*u.deg)

            # Calculate separation between observed coordinate and true coordinate
            separation = observed_coord.separation(true_coord)
            angle = observed_coord.position_angle(true_coord)


        elif separation is not None and angle is not None:
            
            # Ensure separation and angle are given as astropy angle objects
            if isinstance(separation,Angle):
                pass
            else:
                raise ValueError("Separation must be given as an astropy Angle object.")

            if isinstance(angle,Angle):
                pass
            else:
                raise ValueError("Angle must be given as an astropy Angle object.")


        else: # Invalid combination of input
            raise ValueError("Must specify the amount to shift the centre data cube, either by passing a separation and angle, or by giving the true and observed locations of an object in the data.")


        # Define coordinate of the center of the data cube/image from FITS header
        centre_coord = SkyCoord(ra=self.fits_header["CRVAL1"],dec=self.fits_header["CRVAL2"],unit=u.deg,frame='icrs')
        
        # Calculate the center value shifted by the calculated offset
        centre_shifted = centre_coord.directional_offset_by(position_angle=angle,separation=separation)
        shifted_ra = centre_shifted.ra.value
        shifted_dec = centre_shifted.dec.value


        # Create a copy of the data product
        if self.data_product_dimensions == 2: # Image
            shifted_data = Image(filepath=self.filepath,filt=self.filter_chan,label=self.label)
        else: # Data cube
            shifted_data = SpectralDataCube(filepath=self.filepath,channel=self.filter_chan,wavelengths=self.spectral_axis,label=self.label)


        # Update header in each HDU to use shifted coordinate
        for i in range(len(shifted_data.hdu_list)):
            hdu = shifted_data.hdu_list[i]

            if "CRVAL1" in hdu.header:
                hdu.header["CRVAL1"] = shifted_ra
            else:
                pass
            if "CRVAL2" in hdu.header:
                hdu.header["CRVAL2"] = shifted_dec
            else:
                pass

        shifted_data.fits_header["CRVAL1"] = shifted_ra
        shifted_data.fits_header["CRVAL2"] = shifted_dec


        if plot:

            fig = plt.figure(figsize=(21,8))  # Set up plot
            gs = fig.add_gridspec(nrows=1,ncols=3,width_ratios=[1,1,1],wspace=0.45)
            ax_1 = fig.add_subplot(gs[0,0],projection=self.celestial_wcs)  # Original data
            ax_2 = fig.add_subplot(gs[0,1],projection=shifted_data.celestial_wcs) # Aligned data
            fontsize = 20
            cmap = "viridis"

            # Define data to be plotted
            if self.data_product_dimensions == 2:  # Image
                before = self.image
                after = shifted_data.image
            else:  # Data cube
                before = self.data[slice_index,:,:]
                after = shifted_data.data[slice_index,:,:]
            
            # Plot data
            ax_1.set_title("Original Data",fontsize=fontsize)
            ax_1.imshow(before, origin='lower', cmap=cmap,vmin=vmin, vmax=vmax) # Original data

            ax_2.set_title("Aligned Data",fontsize=fontsize)
            ax_2.imshow(after, origin='lower', cmap=cmap,vmin=vmin, vmax=vmax) # Aligned data

            # Set axis labels
            for ax in [ax_1,ax_2]:
                ax.set_xlabel("R.A.", fontsize=fontsize)
                ax.set_ylabel("Dec.",fontsize=fontsize)

                ra = ax.coords[0]
                dec = ax.coords[1]

                ra.set_format_unit(u.deg,decimal=True,show_decimal_unit=True)
                ra.set_ticklabel(exclude_overlapping=True,size=fontsize-2)
                ra.set_major_formatter('d.ddd')
                ra.set_ticks_position('b')   # Set ticks on only bottom, not top

                dec.set_format_unit(u.deg,decimal=True,show_decimal_unit=True)
                dec.set_ticklabel(exclude_overlapping=True,size=fontsize-2)
                dec.set_major_formatter('d.dddd')
                dec.set_ticks_position('l')  # Set ticks on only left, not right


            if filename:  # Save figure
                fig.savefig(filename, bbox_inches="tight")


        return shifted_data


    def save_fits(self,filename,include_errors=True):
        """ Saves the data product to a FITS file.
        
        Parameters
        ----------
        filename: str
            Filepath to save the FITS file to. Must end in .fits
        include_errors : bool (default: True)
            To include the error data in the FITS file
        
         """

        if filename[-5:] != ".fits":  # Check that a valid filename was given
            raise IOError("Must give the filename of a FITS file (ending in .fits).")
        else:
            pass
        
        if self.hdu_list is None:
            raise AttributeError("Missing HDU list attribute.")
        else:
            self.hdu_list.writeto(filename) # Save HDU list to FITS file


class SpectralDataCube(JWST_DataProduct):
    """ A class for integral field unit spectral-imaging data.

    Attributes
    ----------
    filepath : str
        Filepath to FITS file with data cube data.
    data_hdu : class 'astropy.io.fits.hdu.image.ImageHDU'
        Cube data in the form of an astropy ImageHDU object
    cube_hdu : class 'astropy.io.fits.hdu.image.ImageHDU'
        Alias for data_hdu
    data : numpy.ndarray
        Cube data in the form of a numpy array
    cube : numpy.ndarray
        Alias for data
    err_hdu : class 'astropy.io.fits.hdu.image.ImageHDU'
        Cube/image data errors in the form of an astropy ImageHDU object
    err : numpy.ndarray
        Cube data errors in the form of a numpy array
    fits_header : class 'astropy.io.fits.Header'
        FITS header for the data
    wcs : class 'astropy.wcs.WCS'
        World Coordinate System (WCS) of the data
    celestial_wcs : class 'astropy.wcs.WCS'
        Celestial WCS (WCS without the spectral axis, if the data has one)
    fov_angle : float
        Position angle of the V3 reference axis eastward relative to north when projected onto the sky.
        For more information, see:
        https://jwst-docs.stsci.edu/jwst-observatory-characteristics/jwst-position-angles-ranges-and-offsets
    data_product_dimensions : int
        Dimensions of the data product (should be 3 for a data cube)
    ra_num : int
        Number of values in the R.A. dimension of the array, from NAXIS1 in the FITS header
    dec_num : int
        Number of values in the Dec. dimension of the array, from NAXIS2 in the FITS header
    wave_num : int or None
        Number of values in the spectral axis, from NAXIS3 in the FITS header
    instrument : str
        JWST instrument that collected the data
    filter_chan : str or None
        Instrument channel/filter
    channel : str or None
        Alias to filter_chan
    pixel_size : str
        Spatial resolution (size in arcseconds of one pixel)
    label : str, optional
        Label given to data product (default is Instrument + Filter/Channel)
    spectral_axis : array-like
        Spectral axis data

    Methods
    -------
    __init__(self,filepath,channel=None,wavelengths=None,label=None):
        Constructor for all spectral data cube objects.
    get_slice(slice_wavelength=None,slice_index=None,plot=False,cmap="viridis",apertures=None,aperture_colours="r",aperture_labels=None,plot_title=None,filename=None)
        Returns the image data for a slice of the data cube.
    extract_spectrum(self,aperture,get_errors=True):
        Extracts the spectrum within an aperture from the data cube.

    """

    def __init__(self,filepath,channel=None,wavelengths=None,label=None):
        """ Constructor for all spectral data cube objects.

        Parameters
        ----------
        filepath : str
            Filepath to FITS file with data cube/image data.
        channel : str or None
            MIRI MRS channel (or None for NIRSpec) (not case sensitive)
        wavelengths : str or array-like, optional
            Filepath to table with spectral axis data or array of spectral axis
            data (if None is passed, will attempt to extract spectral axis from the data)
        label : str
            Label given to the data product (if None is passed, defaults to the
            Instrument + Channel)

        """

        super().__init__(filepath=filepath,filter_chan=channel,label=label) # Pass arguments to the broad JWST data product parent class

        # Define aliases to the data
        self.cube_hdu = self.data_hdu
        self.cube = self.data
        self.channel = self.filter_chan


        # Define spectral axis
        if wavelengths is not None: # Use file of wavelengths as spectral axis, if given

            if isinstance(wavelengths,str): # Filepath to a list of wavelengths
                wavelength_table = pd.read_table(wavelengths) # Open file of wavelengths
                wavelength_array = wavelength_table[wavelength_table.columns.values[0]].values # This just takes the column at index 0 and converts it to a numpy array
            elif isinstance(wavelengths,(np.ndarray,list)): # Array or list of wavelengths
                wavelength_array =  wavelengths

        else:  # If wavelengths not given, extract spectral axis from data cube
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Supress warnings that spectral axis is not last
                spec_data = Spectrum1D(flux=self.cube*u.Jy/u.sr,wcs=self.wcs)
                wavelength_array = spec_data.spectral_axis
                wavelength_array = wavelength_array.to(u.micron).value # Convert array to microns

        self.spectral_axis = wavelength_array

        # Raise error if the spectral axis is not the same length as the
        # wavelength axis of the data cube
        if len(self.spectral_axis) != self.wave_num:
            raise ValueError("Spectral axis length ({}) does not match the wavelength axis of the data cube ({})".format(len(self.spectral_axis),self.wave_num))


    def get_slice(self,slice_wavelength=None,slice_index=None,plot=False, cmap="viridis",apertures=None,aperture_colours="r",aperture_labels=None,plot_title=None,filename=None):
        """ Return the data for a slice of a data cube.

        Can either retrieve the slice at a given index or the slice closest to given
        wavelength (ex. the slice at 6.2 microns). Can optionally plot the data
        cube slice as a 2D image.

        Parameters
        ----------
        slice_wavelength : float, optional
            Wavelength of slice to plot
        slice_index : int, optional
            Index of slice to plot
        plot : bool
            Plot the data cube slice as a 2D image (default is False)
        cmap : str
            Colour map to use for plot (default is "viridis")
        apertures : array-like [tuple (float,float,float)], optional
            List of circular apertures to overplot. Each aperture must be given
            as an array-like or tuple of (x, y, radius) in pixel units.
        aperture_colours : str or array-like [str]
            Colour(s) to plot each aperture with (default is red)
        aperture_labels : array-like [str] or None
            Labels to overplot on each aperture
        plot_title : str, optional
            Plot title
        filename : str, optional
            Filepath to save plot to (if not passed, plot is not saved)

        Returns
        -------
        slice_data : class 'Image'
            Slice data

        """

        # Check that either a slice index or wavelength was given
        if slice_index is None and slice_wavelength is None: # Neither
            raise AttributeError("Must pass either a slice index or wavelength.")
        elif slice_index and slice_wavelength: # Both
            raise AttributeError("Can only accept either a slice index or wavelength, but not both.")
        else:
            pass


        # If a wavelength was given, find index of slice corresponding to that wavelength
        if slice_wavelength:

            # Check that given wavelength is within the spectral axis
            if min(self.spectral_axis) < slice_wavelength and slice_wavelength < max(self.spectral_axis):
                pass
            else:
                raise IndexError("Given wavelength is out of range of the data cube's spectral axis.")

            slice_index = find_nearest(self.spectral_axis,slice_wavelength)[1] # Use [1] to keep only the index returned by the function, not the closest value
        else:
            pass

        cube_slice = self.cube[slice_index,:,:]  # Numpy array of data for cube slice to plot
        err_slice = self.err[slice_index,:,:]

        # Define cube slice as an Image() object
        slice_data = Image(data=cube_slice,err=err_slice,filt=self.filter_chan,header=self.fits_header,label=self.label)

        if plot:  # Plot slice
            slice_data.plot_image(cmap=cmap,apertures=apertures,aperture_colours=aperture_colours,aperture_labels=aperture_labels,plot_title=plot_title,filename=filename)

        return slice_data


    def extract_spectrum(self,aperture,get_errors=True,remove_nans=True):
        """ Extract spectrum with errors from within an aperture of the cube.

        Function also removes any NaN flux values in the spectrum and their corresponding spectral axis value.

        Parameters
        ----------
        aperture : array-like [float,float,float], 'photutils.aperture.circle.CircularAperture'
            Aperture (specified in pixel units) to extract spectrum from. If an
            array-like is given, must be in the form of (x, y, radius).
        get_errors : boolean (default: True)
            To extract the spectrum flux errors from the error cube.
        remove_nans : boolean (dfault: True)
            To remove NaN flux values from spectrum

        Returns
        -------
        spectrum_object : class 'Spectrum'
            Extracted spectrum

        NOTE: Currently only allows the user to extract spectra for one aperture at a time.

        """

        # Define aperture as a CircularAperture object, if it is not already
        if isinstance(aperture, CircularAperture):
            aperture_obj = aperture
        elif isinstance(aperture,(list,np.ndarray,tuple)):
            x,y,radius = aperture # Unpack aperture values
            aperture_obj = CircularAperture((x,y),r=radius)
        else:
            raise TypeError("Invalid object type for aperture: {}. Aperture must either be a photutils aperture or array, list, or tuple (x, y, radius) of aperture.")


        def _get_spectrum(cube_array,aperture):
            """ Helper function to extract flux from within an aperture from a
            data cube """
            
            # Swap the R.A. and Dec. axes of the cube
            cube_array = cube_array * u.Jy/u.sr
            cube_array = np.swapaxes(cube_array, 1, 2) # This swaps the axes at indices 1 and 2
           
           # Load spectral data from data cube
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Supress warnings that spectral axis is not last
                spectral_data = Spectrum1D(flux=cube_array, wcs=self.wcs)
                cubeviz = Cubeviz()
                cubeviz.load_data(spectral_data, data_label='Full Spectrum')  # Load spectral data into Cubeviz

            # Extract spectrum from region within the aperture
            cubeviz.load_regions([aperture]) 
            spectrum = cubeviz.specviz.get_spectra(apply_slider_redshift=False)  # Extracted spectrum
            spectrum_flux = spectrum["Subset 1"].flux.value  # Array of flux values

            return spectrum_flux

        # Extract spectrum data
        spectral_axis = self.spectral_axis
        spectrum_flux = _get_spectrum(self.cube,aperture_obj)

        # Extract spectrum flux errors
        if get_errors:
            spectrum_flux_err = _get_spectrum(self.err,aperture_obj)
        else:
            spectrum_flux_err = None


        # Remove data points of NaN flux values and their corresponding spectral axis values
        if remove_nans:
            not_nan_ind = np.where(~np.isnan(spectrum_flux))[0] # Indices of flux values that are not NaN
                
            spectrum_flux = [spectrum_flux[i] for i in not_nan_ind]
            spectral_axis = [spectral_axis[i] for i in not_nan_ind]

            if spectrum_flux_err is not None:
                spectrum_flux_err = [spectrum_flux_err[i] for i in not_nan_ind]
            else:
                pass

        else:
            pass
        
        
        # Define aperture information to store in the Spectrum() object
        x,y = aperture_obj.positions   # Center coordinates, in pixels
        r_pix = aperture_obj.r        # Radius, in pixels

        # Convert to sky coordinates
        ra,dec = self.coord_of_pixel(x,y)         # Right ascension and declination
        r_arcsec = self.convert_pix_to_arcsec(r_pix)   # Radius, in arcseconds


        # Create Spectrum object
        spectrum_object = Spectrum(spectral_axis=spectral_axis,flux=spectrum_flux,flux_err=spectrum_flux_err,continuum_subtracted=False,aperture_ra=ra,aperture_dec=dec,aperture_radius=r_arcsec)

        return spectrum_object


class Image(JWST_DataProduct):
    """ A class for 2D image data

    Attributes
    ----------
    filepath : str
        Filepath to FITS file with data cube data.
    data_hdu : class 'astropy.io.fits.hdu.image.ImageHDU'
        Image data in the form of an astropy ImageHDU object
    image_hdu : class 'astropy.io.fits.hdu.image.ImageHDU'
        Alias for data_hdu
    data : numpy.ndarray
        Image data in the form of a numpy array
    image : numpy.ndarray
        Alias for data
    err_hdu : class 'astropy.io.fits.hdu.image.ImageHDU'
        Image data errors in the form of an astropy ImageHDU object
    err : numpy.ndarray
        Image data errors in the form of a numpy array
    fits_header : class 'astropy.io.fits.Header'
        FITS header for the data
    wcs : class 'astropy.wcs.WCS'
        World Coordinate System (WCS) of the data
    celestial_wcs : class 'astropy.wcs.WCS'
        Celestial WCS (WCS without the spectral axis, if the data has one)
    fov_angle : float
        Position angle of the V3 reference axis eastward relative to north when projected onto the sky.
        For more information, see:
        https://jwst-docs.stsci.edu/jwst-observatory-characteristics/jwst-position-angles-ranges-and-offsets
    data_product_dimensions : int
        Dimensions of the data product (should be 2 for an image)
    ra_num : int
        Number of values in the R.A. dimension of the array, from NAXIS1 in the FITS header
    dec_num : int
        Number of values in the Dec. dimension of the array, from NAXIS2 in the FITS header
    instrument : str
        JWST instrument that collected the data
    filter_chan : str or None
        Instrument channel/filter
    filt : str or None
        Alias to filter_chan
    pixel_size : str
        Spatial resolution (size in arcseconds of one pixel)
    label : str, optional
        Label given to data product (default is Instrument + Filter/Channel)


    Methods
    -------
    __init__(self,filepath,filt=None,label=None):
        Constructor for all 2D image objects.
    plot_image(cmap="viridis",apertures=None,aperture_colours="r",aperture_labels=None,plot_title=None,vmin=-1.,vmax=250.
    filename=None)
        Plots 2D image data.

    """

    def __init__(self,filepath,filt=None,label=None):
        """ Constructor for all 2D image objects.

        Parameters
        ----------
        filepath : str
            Filepath to FITS file with data cube/image data.
        filt : str or None
            NIRCam filter (not case sensitive)
        wavelengths : str or array-like, optional
            Filepath to table with spectral axis data or array of spectral axis
            data (if None is passed, will attempt to extract spectral axis from the data)
        label : str
            Label given to the data product (if None is passed, defaults to the
            Instrument + Filter)

        """

        super().__init__(filepath=filepath,filter_chan=filt,label=label) # Pass arguments to the broad JWST data product parent class

        # Define aliases to the data
        self.image_hdu = self.data_hdu
        self.image = self.data


    def plot_image(self,cmap="viridis",apertures=None,aperture_colours="r",aperture_labels=None,plot_title=None,vmin=-1.,vmax=250.,filename=None):
        """ Plots the 2D image data.

        Parameters
        ----------
        cmap : str
            Colour map to use for plot (default is "viridis")
        apertures : array-like [tuple (float,float,float)], optional
            List of circular apertures to overplot. Each aperture must be given
            as an array-like or tuple of (x, y, radius) in pixel units.
            NOTE: Currently does not accept photutils CircularAperture objects
            as valid input
        aperture_colours : str or array-like [str], optional
            Colour(s) to plot each aperture with (default is red)
        aperture_labels : array-like [str] or None, optional
            Labels to overplot on each aperture
        plot_title : str, optional
            Plot title
        vmin : float
            Data value to map the minimumn colourmap value to (default is -1.0)
        vmax : float
            Data value to map the maximum colourmap value (default is 250.0)
        filename : str, optional
            Filepath to save plot to (if not passed, plot is not saved)

        """

        # Set up plot
        fontsize = 25
        fig = plt.figure(figsize=(8,10))
        ax = fig.add_subplot(111, projection=self.wcs, slices=("x","y",self.wave_num))
        ax.set_title(plot_title,fontsize=fontsize)

        # Plot cube slice
        ax.imshow(self.image, origin='lower', cmap=cmap,vmin=vmin, vmax=vmax)

        # Format axis labels
        ax.set_xlabel("R.A.", fontsize=fontsize)
        ax.set_ylabel("Dec.",fontsize=fontsize)

        ra = ax.coords[0]
        dec = ax.coords[1]

        ra.set_format_unit(u.deg,decimal=True,show_decimal_unit=True)
        ra.set_ticklabel(exclude_overlapping=True,size=fontsize-2)
        ra.set_major_formatter('d.ddd')
        ra.set_ticks_position('b')  # Set ticks on only bottom, not top

        dec.set_format_unit(u.deg,decimal=True,show_decimal_unit=True)
        dec.set_ticklabel(exclude_overlapping=True,size=fontsize-2)
        dec.set_major_formatter('d.dddd')
        dec.set_ticks_position('l') # Set ticks on only left, not right


        # Plot apertures, if given.
        if not apertures: # If no apertures passed, continue
            pass

        else:

            apertures = np.array(apertures)  # Convert apertures to an array

            # Check array is of the correct shape (want a list/array containing
            # lists/arrays of 3 elements)
            if len(apertures.shape) == 2:   # Check a list/array containing lists/arrays was given
                if apertures.shape[1] == 3:  # Check each aperture has 3 elements
                    pass
                else:
                    raise ValueError("Invalid aperture format. Aperture lists/arrays must contain 3 elements. Number of elements in each aperture given: {}.".format(apertures.shape[1]))
            else: # Will occur if either list/array does not contain lists/arrays or not all lists/arrays have the same number of elements
                raise ValueError("Invalid aperture format. Apertures must be passed as a list/array containing lists/arrays of 3 elements. Check number of brackets and number of elements in each list/array.")

            # Format aperture colour(s) as a list of the same length as the
            # list/array of apertures
            if isinstance(aperture_colours,(list,np.ndarray)): # If multiple colours were given, check if list/array is the same length (or longer) as the list of apertures
                if len(aperture_colours) >= len(apertures):
                    aperture_colour_list = aperture_colours
                else:
                    raise ValueError("Mismatched length between list of aperture colours and list of apertures.")
            elif isinstance(aperture_colours, str):
                # Create a list of the same length as the list/array of
                # apertures, where each element is the aperture colour
                aperture_colour_list = [aperture_colours] * len(apertures)
            else:
                raise TypeError("Invalid object type for aperture_colour. Must be a string or a list/np.ndarray of strings.")

            # If list of aperture labels given, check it is the same length (or longer) as the list/array of apertures
            if aperture_labels:
                if len(aperture_labels) >= len(apertures):
                    pass
                else:
                    raise ValueError("Mismatched length between list of aperture labels and list of apertures.")
            else:
                pass

            # Plot each aperture
            for i in range(len(apertures)):

                x_pix,y_pix,r_pix = apertures[i]
                aperture_colour = aperture_colour_list[i]
                ax.add_patch(Circle((x_pix,y_pix),r_pix,alpha=0.35,color=aperture_colour))

                if aperture_labels:  # Plot aperture labels, if given
                    ax.annotate(aperture_labels[i], xy=(x_pix + 3.5, y_pix + 2.2), color=aperture_colour,ha='right',va="center",fontsize=fontsize-3)
                else:
                    pass

        if filename: # Save figure
            fig.savefig(filename, bbox_inches="tight")