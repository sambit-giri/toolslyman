import numpy as np
from astropy import units as u
from scipy import special
from tqdm import tqdm
from .scipy_func import *

from .constants import *
from . import cosmology
from .cosmology_calc import *

def column_density_along_skewer(z_source, xHI, dn, dr, X_H=0.76, cosmo=None):
    """
    Compute the cumulative neutral hydrogen column density along a skewer.

    Parameters
    ----------
    z_source : float
        Redshift of the background source.
    xHI : ndarray
        Neutral hydrogen fraction along the skewer.
    dn : ndarray
        Overdensity field (δ = ρ/ρ̄ - 1).
    dr : Quantity or float
        Comoving cell length (can be with or without units).
    X_H : float, optional
        Hydrogen mass fraction. Default is 0.76.
    cosmo : astropy.cosmology, optional
        Cosmology instance. If None, uses default from toolslyman.

    Returns
    -------
    N_HI : ndarray
        Comoving neutral hydrogen column density along the skewer (in cm^-2).
    """
    if dn.min()>1:
        dn = dn/dn.mean()-1

    if cosmo is None:
        cosmo = cosmology.cosmo
    
    try:
        dr = dr.to('cm')
    except:
        dr *= u.Mpc
        print('The comoving cell distance (dr) is assumed to be in Mpc unit.')

    nH = (1+dn)*(X_H*cosmo.Ob0*cosmo.critical_density0/(const.m_p+const.m_e)).to('1/cm^3')
    nHI_comving = xHI*nH
    if nHI_comving.ndim==1:
        nHI_comving = nHI_comving[None,:]
    N_HI = (1+z_source)**(-4)*np.cumsum(nHI_comving*dr, axis=1)
    return N_HI

def optical_depth_lyA_along_skewer(z_source, xHI, dn, dr, temp=1e4*u.K, X_H=0.76, cosmo=None, f_alpha=0.4164, lambda_bins=1000, damped=True, verbose=False):
    """
    Compute the Lyman-alpha optical depth (τ) along a cosmological skewer.

    Parameters
    ----------
    z_source : float
        Redshift of the background source.
    xHI : ndarray
        Neutral hydrogen fraction along the skewer.
    dn : ndarray
        Overdensity field (δ = ρ/ρ̄ - 1).
    dr : Quantity or float
        Comoving cell length (can be with or without units).
    temp : Quantity, optional
        Gas temperature (default: 1e4 K).
    X_H : float, optional
        Hydrogen mass fraction (default: 0.76).
    cosmo : astropy.cosmology, optional
        Cosmology instance. If None, uses default from toolslyman.
    f_alpha : float, optional
        Oscillator strength for Lyman-alpha transition (default: 0.4164).
    lambda_bins : int or ndarray, optional
        If int or float: number of wavelength bins between 1100–1300 Å (rest-frame).
        If array: specifies the wavelength bin edges (assumed in Ångström).
    damped : bool, optional
        Whether to include the damping wing (Voigt profile). If False, uses Gaussian core only.
    verbose : bool, optional
        If True, prints progress information every 10 cells.

    Returns
    -------
    tau_lambda : ndarray
        Total Lyman-alpha optical depth at each wavelength bin.
    lambda_obs : Quantity
        Observed wavelength array corresponding to the computed τ values (in Å).
    """
    if cosmo is None:
        cosmo = cosmology.cosmo
    
    try:
        dr = dr.to('cm')
    except:
        dr *= u.Mpc
        print('The comoving cell distance (dr) is assumed to be in Mpc unit.')
        
    try:
        temp = temp.to('K')
    except:
        temp *= u.K
        print('The temperature is assumed to be in Kelvin unit.')
    
    r_src = cosmo.comoving_distance(z_source)
    r_arr = r_src-dr*(np.arange(xHI.shape[0])+1)
    z_arr = cdist_to_z(r_arr, cosmo=cosmo)
    
    lambda_0 = 1215.67*u.AA
    if isinstance(lambda_bins,(int,float)):
        lambda_obs = np.linspace(1100, 1300, lambda_bins)*u.AA*(1+z_source)
    else:
        try:
            lambda_obs = lambda_bins.to('AA')
        except:
            lambda_obs = lambda_bins*u.AA
            print('The wavelength bins (lambda_bins) provided are assumed to be in Angstrom unit.')


    # Setup physical constants
    m_H = const.m_p.to('g')
    kboltz = const.k_B.to('erg/K')
    # sigma_T = const.sigma_T.to('cm^2')
    # sigma_0 = (np.sqrt(3 * np.pi * sigma_T / 8) * 1e-8 * lambda_0 * f_alpha).to('cm^2')

    # Doppler parameter
    bpar = np.sqrt(2 * kboltz * temp / m_H).to('cm/s')
    # Cpar = (sigma_0 * const.c).to('cm^3/s')

    # Optical depth normalization
    prefactor = (np.sqrt(np.pi) * const.e.esu**2 * f_alpha * lambda_0) / (const.m_e * const.c * bpar)
    prefactor = prefactor.to('cm^2')  # absorption cross-section
    Cpar = prefactor*bpar

    # Convert wavelength grid to redshift grid
    # z_grid = lambda_obs / lambda_0 - 1
    tau_lambda = np.zeros_like(lambda_obs.value)

    n_arr = len(z_arr)
    for i in tqdm(range(n_arr)):
        z = z_arr[i]
        lam_rest = lambda_obs / (1 + z)
        u_i = ((lam_rest / lambda_0 - 1) * const.c / bpar).to('').value
        apar = (6.25e8 / u.s * lambda_0 / (4 * np.pi * bpar)).to('').value
        if (i+1)%10==0 and verbose:
            print(f"{i+1}/{n_arr} | Lyman-alpha wavelength={(lambda_0*(1+z)).to('AA').value:.1f}AA at z={z:.3f}")
        if damped:
            H_a = special.voigt_profile(u_i, np.sqrt(0.5), apar)
        else:
            H_a = np.exp(-u_i ** 2) / np.sqrt(np.pi)

        nH = (1 + dn[i]) * (X_H * cosmo.Ob0 * cosmo.critical_density0 / (const.m_p + const.m_e)).to('1/cm^3')
        nHI = xHI[i] * nH
        dN_HI = nHI * dr
        tau_0 = (Cpar * dN_HI / bpar).to('').value
        tau_lambda += tau_0 * H_a

    return tau_lambda, lambda_obs