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

def optical_depth_lyA_along_skewer(z_source, xHI, dn, dr=None, z_arr=None, temp=1e4*u.K, X_H=0.76, cosmo=None, f_alpha=0.4164, damped=True, verbose=False, use_compute_spectrum=True):
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

    if use_compute_spectrum:
        # Set up required velocity grid in km/s
        z_grid = z_arr[-xHI.shape[0]:]
        v_grid = ((lambda_obs / lambda_0 - 1) * const.c).to('km/s').value

        v_in = ((lambda_0 * (1 + z_grid) / lambda_0 - 1) * const.c).to('km/s').value
        v_out = v_grid

        m_H = const.m_p.to('g').value
        lambda0_val = lambda_0.to('angstrom').value
        temp_val = temp.to('K').value
        cdens = (xHI * (1 + dn) * (X_H * cosmo.Ob0 * cosmo.critical_density0 / const.m_p) * dr).to('1/cm2').value

        tau_lambda = compute_spectrum(v_in, v_out, cdens, temp_val, lambda0_val,
                                    f_alpha, m_H, damped, periodic=False) 
        
        return tau_lambda, lambda_obs

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
        H_a = special.voigt_profile(u_i, np.sqrt(0.5), apar[:,None]/np.sqrt(np.log(2))) * np.sqrt(np.pi)
    else:
        H_a = np.exp(-u_i ** 2) / np.sqrt(np.pi)

    nH = (1 + dn) * (X_H * cosmo.Ob0 * cosmo.critical_density0 / (const.m_p + const.m_e)).to('1/cm^3')
    nHI = xHI * nH
    dN_HI = nHI * dr
    tau_0 = (Cpar * dN_HI / bpar).to('').value
    tau_lambda_arr = tau_0[:,None] * H_a
    tau_lambda = np.sum(tau_lambda_arr, axis=0)

    return tau_lambda, lambda_obs

def compute_spectrum(xvel_in,xvel_out,cdens,temp,lambda0,fvalue,mass,damped,periodic):
    '''
    Returns optical depth array

    xvel_in  : velocity (km/s) of array element (x-coordinate of spectrum)
    xvel_out : velocity (km/s) of array element (x-coordinate of spectrum)
    cdens    : column density (particles/cm^2)
    temp     : temperature (K)
    lambda0  : rest wavelength (Å)
    fvalue   : oscillator strength
    mass     : mass of atom (g)
    
    '''
  
    if damped:
        #    IF round(lambda0) NE 1216. THEN $
        #      message,'Damping wings only possible for HI Ly-alpha!'
        gf_Lya = 0.8323
        g2_Lya = 8.
        # Natural line-width (km/s)
        v_Lya = 0.6679e-5 * gf_Lya / (g2_Lya*(lambda0*1.e-8)*4.*np.pi) 
        v_lya = 0.00606076 # in km/s. Note that the value above is incorrect
        print(' including a damping wing')
        
        
    minbother = 1.e-2               # Min. max. optical depth for inclusion
        
    c = 2.9979e10                   # cm/s
    kboltz = 1.3807e-16             # erg/K
    sigma_T = 6.6525e-25            # cm^2
    
    # Cross section in cm^2: 
    sigma_0 = np.sqrt(3.*np.pi*sigma_T/8.) * 1.e-8 * lambda0 * fvalue
    
    Tpar = 2.0 * kboltz / mass      # erg/K/g
    Cpar = sigma_0 * c              # cm^3/s
    
    nveloc_in = xvel_in.size
    nveloc_out = xvel_out.size
    nveloc1_out = nveloc_out - 1
    
    tau = np.zeros(nveloc_out)
    tauv = np.zeros(nveloc_out)
    
    taumin = minbother / nveloc_in
    
    bpar_inv = 1. / np.sqrt(Tpar * temp) # s/cm
    tauc = Cpar * cdens * bpar_inv  # Central optical depth
    bpar_inv = bpar_inv * 1.e5      # s/km
    
    if damped: apar = v_lya * bpar_inv
    if periodic:
        boxkms  = max(xvel_in)
        boxkms2 = 0.5*boxkms
        
    for i in range(nveloc_in):
        if tauc[i] >= taumin:
            vpar = abs(xvel_out - xvel_in[i])
            if periodic:
                nn = np.where(vpar > boxkms2)
                while (nn.size > 0):
                    print,' count = ',nn.size
                    vpar[nn]=abs(vpar[nn]-boxkms)
                    nn = np.where(vpar > boxkms2)
                    
            vpar = vpar * bpar_inv[i]
            if damped:
                # The voigt_profile function from scipy and the voigt function
                # from IDL map according to
                #    voigt(gamma,x)/sqrt(pi)=voigt_profile(x,sqrt(0.5),gamma).
                # See
                # https://www.nv5geospatialsoftware.com/docs/VOIGT.html and
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.voigt_profile.html
                #dtau = tauc[i] * voigt(apar[i],vpar) / np.sqrt(np.pi)
                dtau = tauc[i] * special.voigt_profile(vpar,np.sqrt(0.5),apar[i])
            else:
                dtau = tauc[i] * np.exp(-vpar*vpar) / np.sqrt(np.pi)
                                
            tau = tau + dtau

    return tau