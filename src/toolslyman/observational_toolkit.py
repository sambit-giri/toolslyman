import numpy as np
import astropy.units as u
import astropy.constants as consts
from astropy.cosmology import Planck18

def FoV_cMpc2(FoV, z, cosmo=None):
    """
    Convert field-of-view to comoving megaparsecs squared (cMpc^2).

    Parameters
    ----------
    FoV : astropy.units.quantity.Quantity
        The field-of-view angle, should be provided in angular units (e.g., steradians).
    z : float
        The redshift.
    cosmo : astropy.cosmology instance, optional
        The cosmology to use for the calculation. If None, Planck18 cosmology is assumed.

    Returns
    -------
    FoV_cMpc2 : astropy.units.quantity.Quantity
        The field-of-view in comoving megaparsecs squared (cMpc^2).
    """
    if cosmo is None:
        print('Assuming Planck18 cosmology')
        cosmo = Planck18
    D = cosmo.comoving_distance(z)
    FoV_cMpc2 = FoV.to('rad2').value * D**2
    return FoV_cMpc2

def mag2flux(mag):
    """
    Convert apparent magnitude to flux.

    Parameters
    ----------
    mag : astropy.units.quantity.Quantity
        The apparent magnitude, should be provided in magnitude units.

    Returns
    -------
    flux : astropy.units.quantity.Quantity
        The flux in units of erg/s/cm^2/Hz.
    """
    flux = 10**(-(mag.to('mag').value + 48.60) / 2.5) * u.erg / u.s / u.cm**2 / u.Hz
    return flux

def Muv2muv(Muv, z, cosmo=None):
    """
    Convert absolute magnitude to apparent magnitude.

    Parameters
    ----------
    Muv : float
        The absolute magnitude.
    z : float
        The redshift.
    cosmo : astropy.cosmology instance, optional
        The cosmology to use for the calculation. If None, Planck18 cosmology is assumed.

    Returns
    -------
    m_uv : float
        The apparent magnitude.
    """
    if cosmo is None:
        print('Assuming Planck18 cosmology')
        cosmo = Planck18
    Kcorrection = 2.5 * np.log10(1 + z)
    m_uv = Muv + cosmo.distmod(z).value - Kcorrection
    return m_uv

def muv2Muv(muv, z, cosmo=None):
    """
    Convert apparent magnitude to absolute magnitude.

    Parameters
    ----------
    muv : float
        The apparent magnitude.
    z : float
        The redshift.
    cosmo : astropy.cosmology instance, optional
        The cosmology to use for the calculation. If None, Planck18 cosmology is assumed.

    Returns
    -------
    Muv : float
        The absolute magnitude.
    """
    if cosmo is None:
        print('Assuming Planck18 cosmology')
        cosmo = Planck18
    Kcorrection = 2.5 * np.log10(1 + z) * u.mag
    Muv = muv - cosmo.distmod(z) + Kcorrection
    return Muv

def Muv2L1500(M_AB):
    """
    Convert absolute magnitude to luminosity at 1500 Å.

    Parameters
    ----------
    M_AB : float
        The absolute magnitude in the AB system.

    Returns
    -------
    L1500 : astropy.units.quantity.Quantity
        The luminosity at 1500 Å in units of erg/s/Hz.
    """
    SI2cgs = 1e7  # Conversion factor from W/Hz to erg/s/Hz
    L1500 = SI2cgs * 10**((34.1 - M_AB.to('mag').value) / 2.5) * u.erg / u.s / u.Hz
    return L1500
