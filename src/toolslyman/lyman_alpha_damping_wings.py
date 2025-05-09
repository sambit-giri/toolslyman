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

def optical_depth_lyA_along_skewer(z_source, xHI, dn, dr, temp=1e4*u.K, X_H=0.76, cosmo=None, f_alpha=0.4164, lambda_bins=1000, xHI_limit=1e-8, damped=True, verbose=False):
    """
    Compute the Lyman-alpha optical depth (τ) along one or more cosmological skewers.

    This function supports both single and multiple skewers (2D arrays). The source is 
    assumed to be at the origin or index 0 of the skewer array. Use `np.roll` to adjust
    the line-of-sight skewer if necessary.

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
    temp : Quantity or ndarray, optional
        Gas temperature in Kelvin. Can be scalar, 1D, or 2D array matching shape of `xHI`.
        Default is 1e4 K.
    X_H : float, optional
        Hydrogen mass fraction. Default is 0.76.
    cosmo : astropy.cosmology.Cosmology, optional
        Cosmology instance to use. If None, uses the default from `toolslyman`.
    f_alpha : float, optional
        Oscillator strength for the Lyman-alpha transition. Default is 0.4164.
    lambda_bins : int or ndarray, optional
        If int or float: number of observed wavelength bins between 1100–1300 Å (rest-frame).
        If array-like: specifies the wavelength bins directly in Ångström.
    xHI_limit : float, optional
        A limit on neutral fraction values to remove residual neutral hydrogen 
        present due to floating point error. Default is 1e-8.
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

    xHI[xHI<=xHI_limit] = 0.

    if xHI.ndim==2:
        tau_lambda_list = []
        n_skewer = xHI.shape[0]
        for j in range(n_skewer):
            if verbose:
                print(f"Skewer Number {j+1}/{n_skewer}")
            tau_lambdaj, lambda_obsj = optical_depth_lyA_along_skewer(
                                            z_source, xHI[j,:], 
                                            dn[j,:] if dn.ndim==2 else dn, 
                                            dr, temp=temp[j,:] if temp.ndim==2 else temp, 
                                            X_H=X_H, cosmo=cosmo, f_alpha=f_alpha,
                                            lambda_bins=lambda_bins, damped=damped, verbose=False)
            tau_lambda_list.append(tau_lambdaj)
        return np.array(tau_lambda_list), lambda_obsj    
    
    lambda_0 = 1215.67*u.AA
    # if isinstance(lambda_bins,(int,float)):
    #     lambda_obs = np.linspace(1100, 1300, lambda_bins)*u.AA*(1+z_source)
    # else:
    #     try:
    #         lambda_obs = lambda_bins.to('AA')
    #     except:
    #         lambda_obs = lambda_bins*u.AA
    #         print('The wavelength bins (lambda_bins) provided are assumed to be in Angstrom unit.')

    r_src = cosmo.comoving_distance(z_source)
    r_arr = r_src-dr*(np.arange(-xHI.shape[0],xHI.shape[0]))
    z_arr = cdist_to_z(r_arr, cosmo=cosmo)
    lambda_obs = lambda_0*(1+z_arr)

    # dz_lam = (np.gradient(lambda_obs)[0]/lambda_0).value
    # z_lam  = np.arange(z_source,z_arr.min(),-dz_lam)
    # r_lam  = cosmo.comoving_distance(z_lam)

    # dn_fit = interp1d(z_arr, dn, kind='linear', fill_value='extrapolate')
    # xHI_fit = interp1d(z_arr, xHI, kind='nearest-up', fill_value='extrapolate')
    # temp_fit = interp1d(z_arr, temp.to('K').value, kind='linear', fill_value='extrapolate')
    # dn_lam = dn_fit(z_lam)
    # xHI_lam = xHI_fit(z_lam)
    # temp_lam = temp_fit(z_lam)*u.K

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

    # lam_rest = lambda_obs / (1 + z_arr[:,None])
    # lam_rest = lambda_obs / (1 + z_lam[:,None])
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
    # nH = (1 + dn_lam) * (X_H * cosmo.Ob0 * cosmo.critical_density0 / (const.m_p + const.m_e)).to('1/cm^3')
    # nHI = xHI_lam * nH
    # dN_HI = nHI * np.abs(np.gradient(r_lam))
    tau_0 = (Cpar * dN_HI / bpar).to('').value
    tau_lambda_arr = tau_0[:,None] * H_a
    tau_lambda = np.sum(tau_lambda_arr, axis=0)

    return tau_lambda, lambda_obs