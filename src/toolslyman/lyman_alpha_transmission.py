import numpy as np
from glob import glob
from tqdm import tqdm
from scipy import integrate
from scipy.special import wofz
import astropy
from astropy import constants, units
from astropy.cosmology import Planck18


def approximation_voigt_faddeeva(x, a):
    """
    Compute the Voigt profile using the Faddeeva function.

    The Voigt profile combines Doppler and pressure broadening effects and can be computed using the Faddeeva function (complex error function). This approximation is particularly useful for calculating the Voigt profile efficiently.

    Parameters
    ----------
    x : float or ndarray
        The frequency or wavelength offset from the line center, scaled by the Doppler width. For example, \( x = \frac{\nu - \nu_0}{\Delta \nu_d} \), where \( \nu \) is the frequency, \( \nu_0 \) is the center frequency, and \( \Delta \nu_d \) is the Doppler width.
        
    a : float
        The line a-factor, which characterizes the ratio of the Lorentzian width to the Doppler width. It is defined as \( a = \frac{\gamma}{\Delta \nu_d} \), where \( \gamma \) is the Lorentzian width and \( \Delta \nu_d \) is the Doppler width.

    Returns
    -------
    H : float or ndarray
        The value of the Voigt profile at the given `x` values. The output shape matches the input shape.

    Notes
    -----
    The Voigt profile is computed using the Faddeeva function as follows:

    \[
    V(x, a) = \text{Re}\left[ \text{wofz}(x + i a) \right]
    \]

    Here, `wofz` is the Faddeeva function, which is a standard way to compute the Voigt profile.

    Example
    -------
    >>> x = 0.5
    >>> a = 0.1
    >>> profile = approximation_voigt_faddeeva(x, a)
    >>> print(profile)
    """
    # Convert a to the H(a, x) form of the Voigt profile
    u = x + 1j * a
    # Compute the Voigt profile using the wofz function
    H = wofz(u).real
    return H

def direct_voigt_profile(x, a):
    """
    Compute the Voigt profile using direct numerical integration of the line profile function.

    The Voigt profile is a combination of a Lorentzian and a Gaussian profile, used to model spectral lines broadened by both Doppler and pressure broadening effects. This function calculates the Voigt profile by directly integrating the line profile function.

    Parameters
    ----------
    x : array_like
        The array of positions (e.g., frequency or wavelength offsets) at which to evaluate the Voigt profile.

    a : float or array_like
        The line a-factor, which characterizes the ratio of the Lorentzian to Gaussian broadening. It is defined as \( a = \frac{\gamma}{\sigma \sqrt{2}} \), where \( \gamma \) is the Lorentzian width and \( \sigma \) is the Gaussian width.

    Returns
    -------
    V : ndarray
        The Voigt profile evaluated at the input positions `x`. The shape of the output matches the shape of the input `x`.

    Notes
    -----
    The Voigt profile is computed using the integral of the line profile function:
    
    \[
    V(x, a) = \frac{a}{\pi} \int_{-\infty}^\infty \frac{e^{-y^2}}{(y-x)^2 + a^2} \, dy
    \]

    where \( a \) is the line a-factor and \( x \) represents the spectral line's frequency or wavelength offset.

    Example
    -------
    >>> x = np.linspace(-5, 5, 1000)
    >>> a = 0.5
    >>> profile = direct_voigt_profile(x, a)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, profile)
    >>> plt.xlabel('x')
    >>> plt.ylabel('Voigt Profile')
    >>> plt.title('Voigt Profile using Direct Numerical Integration')
    >>> plt.show()
    """
    Iy = lambda y,x,a: np.exp(-y**2) / ((y - x)**2 + a**2)
    def out_func(x,a):
        return integrate.quad(Iy, -np.inf, np.inf, args=(x,a))[0]
    out = (a / np.pi) * np.vectorize(out_func)(x,a)
    return out

def lyA_scattering_cross_section(nu, Tk):
    """
    Calculate the Lyman-alpha scattering cross-section for a given frequency and temperature.
    
    Parameters
    ----------
    nu : float or astropy.units.quantity.Quantity
        The frequency at which to calculate the cross-section. Can be given as a float in Hz or as an astropy Quantity.
    
    Tk : float or astropy.units.quantity.Quantity
        The temperature of the gas. Can be given as a float in K or as an astropy Quantity.

    Returns
    -------
    sigmaA : astropy.units.quantity.Quantity
        The Lyman-alpha scattering cross-section at the given frequency and temperature, in units of cm^2.
    
    Notes
    -----
    This function uses parameters from e.g. Dijkstra (2014, arXiv:1406.7292).
    The Voigt profile is approximated using the Faddeeva function.
    """
    if isinstance(Tk, units.quantity.Quantity):
        Tk4 = Tk.to('K').value/1e4
    else:
        Tk4 = Tk/1e4
        Tk *= units.K
    sigmaA0 = 5.9e-14/np.sqrt(Tk4)*units.cm**2 #2004.13065
    a_nu = 4.7e-4/np.sqrt(Tk4) #2004.13065

    nuA = 2.46e15 #units.Hz
    dnu_D = nuA*np.sqrt((constants.k_B*Tk/constants.m_p/constants.c**2).to(''))
    if isinstance(nu, units.quantity.Quantity):
        nu = nu.to('Hz').value
    x_nu = (nu-nuA)/dnu_D

    phi_xa = approximation_voigt_faddeeva(x_nu, a_nu)
    sigmaA = sigmaA0*phi_xa
    return sigmaA 

def tau_nu_lyA(z, xHI, Tk, dr, nu=2.46e15*units.Hz, X_H=0.76, cosmo=None):
    """
    Estimate the Lyman-alpha optical depth on a simulation grid.

    Parameters
    ----------
    z : float
        The redshift at which the optical depth is being calculated.
    xHI : ndarray
        The neutral hydrogen fraction at each grid point.
    Tk : float or astropy.units.quantity.Quantity
        The temperature of the gas at each grid point. Can be given as a float in K or as an astropy Quantity.
    dr : astropy.units.quantity.Quantity
        The comoving path length through the grid cells. Should be an astropy Quantity.
    nu : float or astropy.units.quantity.Quantity, optional
        The frequency at which to calculate the optical depth, default is the Lyman-alpha frequency (2.46e15 Hz).
    X_H : float, optional
        The hydrogen mass fraction, default is 0.76.
    cosmo : astropy.cosmology instance, optional
        The cosmology to use for the calculation. If None, Planck18 cosmology is assumed.

    Returns
    -------
    tau_nu : ndarray
        The Lyman-alpha optical depth at each grid point.

    Notes
    -----
    This function uses Eq. 7 from Smith, A. et al. (2015, arXiv:1409.4480).
    The cross-section and Voigt profile parameters are taken from e.g. Dijkstra (2014, arXiv:1406.7292).

    Example
    -------
    >>> from astropy import units as u
    >>> from astropy.cosmology import Planck18
    >>> z = 6.0
    >>> xHI = np.random.rand(100, 100, 100)
    >>> Tk = 1e4 * u.K
    >>> dr = 1.0 * u.Mpc
    >>> tau = tau_nu_lyA(z, xHI, Tk, dr)
    """
    if cosmo is None:
        print('Assuming Planck18 cosmology')
        cosmo = Planck18

    if isinstance(Tk, units.quantity.Quantity):
        Tk4 = Tk.to('K').value/1e4
    else:
        Tk4 = Tk/1e4
        Tk *= units.K
    sigmaA0 = 5.9e-14/np.sqrt(Tk4)*units.cm**2 #2004.13065
    a_nu = 4.7e-4/np.sqrt(Tk4) #2004.13065

    nuA = 2.46e15 #units.Hz
    dnu_D = nuA*np.sqrt((constants.k_B*Tk/constants.m_p/constants.c**2).to(''))
    if isinstance(nu, units.quantity.Quantity):
        nu = nu.to('Hz').value
    x_nu = (nu-nuA)/dnu_D
    nH = (X_H*cosmo.Ob0*(1+z)**3*cosmo.critical_density0/constants.m_p).to('1/cm^3')
    nHI = xHI*nH
    H_xa = approximation_voigt_faddeeva(x_nu, a_nu)
    tau_nu = 1.820e5*H_xa/np.sqrt(Tk4)*nHI.to('1/cm^3').value*dr.to('pc').value
    return tau_nu
