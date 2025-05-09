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
    if cosmo is None:
        cosmo = cosmology.cosmo

    if dn.min()>1:
        dn = dn/dn.mean()-1
    
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

def optical_depth_lyA_along_skewer(z_source, xHI, dn, dr=None, z_arr=None, temp=1e4*u.K, X_H=0.76, cosmo=None, f_alpha=0.4164, damped=True, verbose=False):
    """
    Compute the Lyman-alpha optical depth (τ) along one or more cosmological skewers.

    This function supports both single and multiple skewers (2D arrays). The source is 
    assumed to be at the origin or index 0 of the skewer array if dr is provided. 
    Use `np.roll` to adjust the line-of-sight skewer if necessary. 

    Parameters
    ----------
    z_source : float
        Redshift of the background source (e.g., a quasar).
    xHI : ndarray
        Neutral hydrogen fraction along the skewer. Shape can be (N,) or (N_skewers, N).
    dn : ndarray
        Baryon overdensity field (δ = ρ/ρ̄ - 1). Shape must match `xHI`.
    dr : Quantity or float
        Comoving length of each cell along the skewer (e.g., in Mpc or cm).
    z_arr : ndarray
        Redshift array corresponding to the skewer.
    temp : Quantity or ndarray, optional
        Gas temperature in Kelvin. Can be scalar, 1D, or 2D array matching shape of `xHI`.
        Default is 1e4 K.
    X_H : float, optional
        Hydrogen mass fraction. Default is 0.76.
    cosmo : astropy.cosmology.Cosmology, optional
        Cosmology instance to use. If None, uses the default from `toolslyman`.
    f_alpha : float, optional
        Oscillator strength for the Lyman-alpha transition. Default is 0.4164.
    damped : bool, optional
        If True, include the damping wing using a Voigt profile. Otherwise, use only 
        the Doppler core (Gaussian).
    verbose : bool, optional
        If True, print progress every 10 steps. Also prints skewer number if multiple
        skewers are processed.

    Returns
    -------
    tau_lambda : ndarray
        Optical depth as a function of observed wavelength. Shape is (N_lambda,) for single skewer
        or (N_skewers, N_lambda) for multiple skewers.
    lambda_obs : Quantity
        Observed wavelength grid corresponding to `tau_lambda`, in Ångström.
    """
    assert dr is not None or z_arr is not None 

    if cosmo is None:
        cosmo = cosmology.cosmo

    if dn.min()>1:
        dn = dn/dn.mean()-1
    
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
    if np.array(temp.value).ndim==0:
        temp = temp*np.ones_like(dn)

    if xHI.ndim==2:
        tau_lambda_list = []
        n_skewer = xHI.shape[0]
        for j in range(n_skewer):
            if verbose:
                print(f"Skewer Number {j+1}/{n_skewer}")
            tau_lambdaj, lambda_obsj = optical_depth_lyA_along_skewer(
                                            z_source, xHI[j,:], 
                                            dn[j,:] if dn.ndim==2 else dn, 
                                            dr, z_arr, 
                                            temp=temp[j,:] if temp.ndim==2 else temp, 
                                            X_H=X_H, cosmo=cosmo, f_alpha=f_alpha,
                                            damped=damped, verbose=False)
            tau_lambda_list.append(tau_lambdaj)
        return np.array(tau_lambda_list), lambda_obsj    
    
    lambda_0 = 1215.67*u.AA
    r_src = cosmo.comoving_distance(z_source)
    if z_arr is None:
        r_arr = r_src-dr*(np.arange(-xHI.shape[0],xHI.shape[0]))
        z_arr = cdist_to_z(r_arr, cosmo=cosmo)
    else:
        r_arr = cosmo.comoving_distance(z_arr)
    lambda_obs = lambda_0*(1+z_arr)

    # Setup physical constants
    m_H = const.m_p.to('g')
    kboltz = const.k_B.to('erg/K')

    # Doppler parameter
    bpar = np.sqrt(2 * kboltz * temp / m_H).to('cm/s')
    # bpar = np.sqrt(2 * kboltz * temp_lam / m_H).to('cm/s')

    # Optical depth normalization
    prefactor = (np.sqrt(np.pi) * const.e.esu**2 * f_alpha * lambda_0) / (const.m_e * const.c * bpar)
    prefactor = prefactor.to('cm^2')  # absorption cross-section
    Cpar = prefactor*bpar

    lam_rest = lambda_obs / (1 + z_arr[-xHI.shape[0]:,None])
    u_i = ((lam_rest / lambda_0 - 1) * const.c / bpar[:,None]).to('').value
    apar = (6.25e8 / u.s * lambda_0 / (4 * np.pi * bpar)).to('').value
    if damped:
        H_a = special.voigt_profile(u_i, np.sqrt(0.5), apar[:,None])
    else:
        H_a = np.exp(-u_i ** 2) / np.sqrt(np.pi)

    nH = (1 + dn) * (X_H * cosmo.Ob0 * cosmo.critical_density0 / (const.m_p + const.m_e)).to('1/cm^3')
    nHI = xHI * nH
    dN_HI = nHI * dr
    tau_0 = (Cpar * dN_HI / bpar).to('').value
    tau_lambda_arr = tau_0[:,None] * H_a
    tau_lambda = np.sum(tau_lambda_arr, axis=0)

    return tau_lambda, lambda_obs