import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18
from . import source_model

def IGM_tomography_parameters(box_len, filter_name='NB816', cosmo=None):
    '''
    Set IGM tomography parameters.

    This function sets the parameters required for Intergalactic Medium (IGM) tomography
    using a specified narrow-band filter. It computes various observational parameters
    based on the chosen filter and the given cosmological model.

    Parameters:
    -----------
    box_len : float
        The side length of the simulation box in Mpc/h.
    filter_name : str, optional
        The name of the filter to be used for IGM tomography. Default is 'NB816'.
        Currently, only 'NB816' is supported.
    cosmo : astropy.cosmology.Cosmology, optional
        The cosmological model to be used. Default is Planck18.

    Returns:
    --------
    dict
        A dictionary containing the following IGM tomography parameters:
        - 'zmin' : float
            The minimum redshift for the IGM tomography.
        - 'zmax' : float
            The maximum redshift for the IGM tomography.
        - 'mNB_lim' : astropy.units.quantity.Quantity
            The limiting magnitude in the narrow-band filter.
        - 'muv_lim' : astropy.units.quantity.Quantity
            The limiting UV magnitude.
        - 'smoothing_length' : float
            The smoothing length in cMpc/h.
        - 'Ngal_in_box' : int
            The number of galaxies in the simulation box.

    Notes:
    ------
    - The function assumes the use of the 'NB816' filter, which is tied to specific
      observational parameters related to the Hyper Suprime-Cam (HSC) field of view.
    - The cosmological model defaults to Planck18 if not provided.
    - The source model framework used for galaxy surface density calculations is
      instantiated using the given cosmological model.

    Example:
    --------
    >>> params = IGM_tomography_parameters(100, 'NB816')
    >>> print(params['zmin'], params['zmax'])
    5.81 6.86
    '''
    assert filter_name in ['NB816']

    if cosmo is None: cosmo = Planck18
    BKG_MODEL = source_model.SOURCE_MODEL_FRAMEWORK(cosmo=cosmo)

    if filter_name=='NB816':
        print('### Set IGM tomography parameters with NB816 ###')
        redshift = 5.7

        FoV = 1.76*u.deg**2  # HSC FoV
        R_FoV = (np.sqrt(FoV.to('rad2').value/np.pi)*cosmo.comoving_distance(redshift)*cosmo.h).to('Mpc').value # cMpc/h
        print('HSC FoV [deg2] = ', FoV.to('deg^2'))
        # bkg. redshift for NB816 IGM tomography
        zmin = 5.81
        zmax = 6.86
        zbg  = (zmax+zmin)/2

        muv_lim = 26.0*u.mag
        mNB_3sigma = 27.5*u.mag # 27.5 # 27.37
        mNB_lim = mNB_3sigma+2.5*np.log10(3/1)*u.mag

        realistic_factor = 0.20 # completeness factor
        ngal_2D = realistic_factor*BKG_MODEL.LBG_surface_density(muv_lim, zmin, zmax) # [1/cMpc2]
        Ngal = BKG_MODEL.FoV_cMpc2(FoV,zbg) * ngal_2D
        ngal_2D_deg2 = Ngal/FoV
        smoothing_length = (1/np.sqrt(ngal_2D) * cosmo.h).to('Mpc').value # [cMpc/h]
        print('FoV =', FoV)
        print('Galaxy surface density = ', ngal_2D,      ' [1/cMpc2]')
        print('Galaxy surface density = ', ngal_2D_deg2, ' [1/deg2]')
        print('Galaxy count = ', Ngal)
        print('transverse resolution/smoothing length = %.2f [cMpc/h]' % smoothing_length)

        # Set IGM tomography parameters with NB816
        Ngal_in_box = int(ngal_2D*(box_len*u.Mpc/cosmo.h)**2)

        return {
            'zmin': zmin,
            'zmax': zmax,
            'mNB_lim': mNB_lim,
            'muv_lim': muv_lim,
            'smoothing_length': smoothing_length,
            'Ngal_in_box': Ngal_in_box,
            'R_FoV': R_FoV,
            }