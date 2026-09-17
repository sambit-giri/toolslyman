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

def optical_depth_lyA_along_skewer(z_source, xHI, dn, dr=None, z_arr=None, temp=1e4*u.K, vpec=None, X_H=0.76, cosmo=None, f_alpha=0.4164, damped=True, verbose=False, use_compute_spectrum=True):
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
    vpec : Quantity or ndarray, optional
        Line-of-sight peculiar velocity of the gas in each cell (e.g., in km/s), positive
        when receding from the observer. Shape must match `xHI`. Only used when
        `use_compute_spectrum=True`. Default is None (no peculiar velocity, i.e. pure
        Hubble flow).
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

    if vpec is not None:
        try:
            vpec = vpec.to('km/s')
        except:
            vpec *= u.km/u.s
            print('The peculiar velocity (vpec) is assumed to be in km/s unit.')

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
                                            vpec=vpec[j,:] if (vpec is not None and vpec.ndim==2) else vpec,
                                            X_H=X_H, cosmo=cosmo, f_alpha=f_alpha,
                                            damped=damped, verbose=False,
                                            use_compute_spectrum=use_compute_spectrum)
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
        z_grid = z_arr[-xHI.shape[0]:]

        m_H = const.m_p.to('g').value
        lambda0_val = lambda_0.to('angstrom').value
        temp_val = temp.to('K').value
        vpec_val = vpec.to('km/s').value if vpec is not None else None
        # Proper-frame column density: dr is comoving, but the proper density scales
        # as (1+z)^3 and the proper path length as dr/(1+z), giving a net (1+z)^2
        # conversion factor that must be applied at each cell's own redshift.
        cdens = (xHI * (1 + dn) * (1 + z_grid)**2 *
                 (X_H * cosmo.Ob0 * cosmo.critical_density0 / const.m_p) * dr).to('1/cm2').value

        tau_lambda = _damping_wing_spectrum(z_grid, z_arr, cdens, temp_val, lambda0_val,
                                             f_alpha, m_H, damped, vpec=vpec_val)

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

def _damping_wing_spectrum(z_grid, z_obs, cdens, temp, lambda0, fvalue, mass, damped, vpec=None):
    """
    Cosmologically-correct analogue of compute_spectrum() for a sightline that spans
    a redshift range too large for the small-Delta_z, velocity-difference picture
    (v = v_in - v_out) to hold.

    The proper-frame frequency detuning of a photon observed at z_obs, evaluated at
    a cell of redshift z', is Delta_v/c = (1+z')/(1+z_obs) - 1 (Miralda-Escude 1998),
    which is a *ratio* of (1+z) factors rather than a simple velocity difference; it
    reduces to the naive v_in - v_out picture only for |z' - z_obs| << 1 + z_obs. A
    cell's own line-of-sight peculiar velocity adds directly to this detuning (it is
    always non-relativistic, so a simple velocity addition is appropriate regardless
    of the cosmological redshift), the same way it would if the gas had no bulk motion
    but sat at a slightly different redshift.

    The standard Voigt-Hjerting profile assumes the narrow-line (Delta_lambda << lambda)
    limit of the Lyman-alpha cross section, which drops the (omega/omega_alpha)^4
    numerator and replaces (omega/omega_alpha)^6 with 1 in the resonance denominator
    (Peebles 1993, Sec. 23; Miralda-Escude 1998, Appendix, Eq. A1). That is an
    excellent approximation for narrow absorption features, but a broad damping wing
    with Delta_lambda/lambda ~ 0.01-0.1 (exactly the EoR regime this module targets)
    is well outside it; Miralda-Escude (1998) explicitly keeps both terms for this
    reason. Both terms are applied here as a multiplicative correction to the
    standard Voigt-Hjerting result, which leaves the near-resonance (thermally
    dominated) part of the profile essentially unchanged while correcting the far
    wing shape.

    Parameters
    ----------
    z_grid : ndarray, shape (N,)
        Redshift of each cell along the skewer.
    z_obs : ndarray, shape (M,)
        Redshift corresponding to each observed-wavelength output bin.
    cdens : ndarray, shape (N,)
        Proper-frame HI column density of each cell (cm^-2).
    temp : ndarray, shape (N,)
        Gas temperature of each cell (K).
    lambda0 : float
        Rest wavelength (Angstrom).
    fvalue : float
        Oscillator strength.
    mass : float
        Absorbing particle mass (g).
    damped : bool
        If True, include the Lorentzian damping wing; otherwise Doppler core only.
    vpec : ndarray, shape (N,), optional
        Line-of-sight peculiar velocity of each cell (km/s), positive when receding
        from the observer. Default is None (no peculiar velocity).

    Returns
    -------
    tau : ndarray, shape (M,)
        Optical depth at each z_obs.
    """
    minbother = 1.e-2
    c_cgs = 2.9979e10
    c_kms = 2.9979e5
    kboltz = 1.3807e-16
    sigma_T = 6.6525e-25

    sigma_0 = np.sqrt(3.*np.pi*sigma_T/8.) * 1.e-8 * lambda0 * fvalue
    Tpar = 2.0 * kboltz / mass
    Cpar = sigma_0 * c_cgs

    nveloc_in = cdens.size
    taumin = minbother / nveloc_in

    bpar_inv_cgs = 1. / np.sqrt(Tpar * temp)   # s/cm
    tauc = Cpar * cdens * bpar_inv_cgs         # central optical depth
    bpar_inv_kms = bpar_inv_cgs * 1.e5         # s/km

    if damped:
        v_lya = 0.00606076   # natural line-width of Lyman-alpha, in km/s

    if vpec is None:
        vpec = np.zeros(nveloc_in)

    tau = np.zeros_like(z_obs, dtype=float)
    for i in range(nveloc_in):
        if tauc[i] >= taumin:
            vpar_signed = c_kms * ((1 + z_grid[i]) / (1 + z_obs) - 1) + vpec[i]
            vpar = np.abs(vpar_signed)
            x = vpar * bpar_inv_kms[i]
            if damped:
                apar = v_lya * bpar_inv_kms[i]
                # See compute_spectrum() docstring: voigt_profile(x,sqrt(0.5),gamma)*sqrt(pi)
                # is the standard Voigt-Hjerting function H(a,x).
                dtau = tauc[i] * special.voigt_profile(x, np.sqrt(0.5), apar) * np.sqrt(np.pi)
                # Exact-cross-section correction (Miralda-Escude 1998, Eq. A1): restores the
                # (omega/omega_alpha)^4 numerator and (omega/omega_alpha)^6 denominator terms
                # dropped by the narrow-line Voigt-Hjerting approximation.
                nu_ratio = 1. + vpar_signed / c_kms
                dtau = dtau * nu_ratio**4 * (x**2 + apar**2) / (x**2 + apar**2 * nu_ratio**6)
            else:
                dtau = tauc[i] * np.exp(-x**2) / np.sqrt(np.pi)
            tau = tau + dtau

    return tau

def tau_GP(z, xHI, X_H=1.0, cosmo=None):
    """
    On-resonance Gunn-Peterson optical depth (Gunn & Peterson 1965; Eq. 2 of
    Miralda-Escude 1998, as given in Huberty et al. 2025).

    Miralda-Escude (1998)'s original formula (Sec. 2) is
    tau_0 = 2.1e5 * [Omega_b h (1-Y)/0.03] * [H_0(1+z)^1.5/H(z)] * [(1+z)/6]^1.5,
    with (1-Y) the hydrogen mass fraction (Y = helium abundance). Huberty et al.
    (2025)'s reproduction of this formula (their Eq. 2, the one implemented here)
    omits the (1-Y) factor entirely, i.e. implicitly assumes X_H = 1-Y = 1 (pure
    hydrogen, no helium correction) -- the two prefactors otherwise agree to ~2%.
    X_H is exposed here so the formula can be evaluated with the same hydrogen
    mass fraction used elsewhere in this module (e.g. `optical_depth_lyA_along_skewer`,
    which defaults to X_H=0.76) for a direct, apples-to-apples comparison.

    Parameters
    ----------
    z : float or ndarray
        Redshift(s) at which to evaluate the optical depth.
    xHI : float
        Volume-averaged neutral hydrogen fraction.
    X_H : float, optional
        Hydrogen mass fraction (1-Y). Default is 1.0, matching Huberty et al.
        (2025)'s Eq. 2 exactly; pass X_H=0.76 to match the physically-motivated
        value used by the rest of toolslyman (and by Miralda-Escude 1998).
    cosmo : astropy.cosmology, optional
        Cosmology instance. If None, uses the default from toolslyman.

    Returns
    -------
    tau_GP : float or ndarray
        Gunn-Peterson optical depth.
    """
    if cosmo is None:
        cosmo = cosmology.cosmo
    h = cosmo.h
    OmegaM = cosmo.Om0
    OmegaB0h2 = cosmo.Ob0 * h**2
    return 1.8e5 * h**-1 * OmegaM**-0.5 * (OmegaB0h2/0.02) * X_H * ((1+z)/7)**1.5 * xHI

def _I_func(x):
    """
    Helper function I(x) entering the analytic damping-wing optical depth
    (Eq. 1 of Miralda-Escude 1998 / Huberty et al. 2025). Only valid for
    0 < x < 1; the algebraic and logarithmic terms both change sign across
    the removable pole at x=1.
    """
    sqx = np.sqrt(x)
    return x**4.5/(1-x) + 9./7*x**3.5 + 9./5*x**2.5 + 3*x**1.5 + 9*x**0.5 \
        - 4.5*np.log((1+sqx)/(1-sqx))

def tau_igm_analytic(z_obs, z_source, xHI, z_end=6.0, z_bubble=None, X_H=1.0, cosmo=None):
    """
    Analytic IGM Lyman-alpha damping-wing optical depth (Miralda-Escude 1998;
    Eq. 1-2 of Huberty et al. 2025, arXiv:2501.13899), for a uniform-xHI
    neutral patch extending from z_end up to z_bubble along the line of
    sight to a source at z_source.

    The line of sight is split into three pieces: fully transparent below
    z_end, on-resonance Gunn-Peterson saturation inside the neutral patch
    (z_end <= z_obs <= z_bubble), and the analytic red damping wing redward
    of the bubble edge (z_obs > z_bubble). The closed-form wing expression
    is only real-valued there (both arguments of I(x) are < 1); this is the
    physical "red damping wing" from gas near the source bleeding into
    wavelengths redder than its own line.

    Parameters
    ----------
    z_obs : float or ndarray
        Observed-frame redshift(s) at which to evaluate the optical depth.
    z_source : float
        Redshift of the background source.
    xHI : float
        Volume-averaged neutral hydrogen fraction of the patch.
    z_end : float, optional
        Redshift marking the end of reionization (low-z edge of the neutral
        patch). Default is 6.0.
    z_bubble : float, optional
        Redshift corresponding to the edge of the ionized bubble around the
        source (high-z edge of the neutral patch). Defaults to z_source
        (no bubble).
    X_H : float, optional
        Hydrogen mass fraction (1-Y), passed through to `tau_GP`. Default is
        1.0, matching Huberty et al. (2025)'s Eq. 1-2 exactly; pass X_H=0.76
        to match `optical_depth_lyA_along_skewer`'s default and Miralda-Escude
        (1998)'s original formula. See `tau_GP` for details.
    cosmo : astropy.cosmology, optional
        Cosmology instance. If None, uses the default from toolslyman.

    Returns
    -------
    tau_IGM : float or ndarray
        Analytic IGM optical depth, same shape as z_obs.
    """
    if z_bubble is None:
        z_bubble = z_source

    scalar_input = np.ndim(z_obs) == 0
    z_obs = np.atleast_1d(np.asarray(z_obs, dtype=float))
    tau = np.zeros_like(z_obs)

    m_trough = (z_obs >= z_end) & (z_obs <= z_bubble)
    tau[m_trough] = tau_GP(z_obs[m_trough], xHI, X_H=X_H, cosmo=cosmo)

    m_wing = z_obs > z_bubble
    R_alpha = 2.02e-8
    zw = z_obs[m_wing]
    pref = tau_GP(z_source, xHI, X_H=X_H, cosmo=cosmo) * R_alpha/np.pi * ((1+zw)/(1+z_source))**1.5
    x1 = (1+z_bubble)/(1+zw)
    x2 = (1+z_end)/(1+zw)
    tau[m_wing] = pref * (_I_func(x1) - _I_func(x2))

    return tau[0] if scalar_input else tau