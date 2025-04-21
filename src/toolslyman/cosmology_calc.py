import numpy as np
from astropy import units as u
from scipy.interpolate import splev, splrep
from . import cosmology

def cdist_to_z(cdist, cosmo=None, zmin=1e-4, zmax=1100, n_bins=1000):
    """
    Convert comoving distance(s) to redshift(s) using an interpolated inverse of the 
    comoving distance-redshift relation.

    Parameters
    ----------
    cdist : Quantity or array-like
        Comoving distance(s) to convert (can be with or without units).
    cosmo : astropy.cosmology instance, optional
        Cosmological model to use. Defaults to toolslyman.cosmology.cosmo if None.
    zmin : float, optional
        Minimum redshift to consider for interpolation (default: 1e-4).
    zmax : float, optional
        Maximum redshift to consider for interpolation (default: 1100).
    n_bins : int, optional
        Number of redshift samples to build the interpolation grid (default: 1000).

    Returns
    -------
    z_arr : ndarray
        Redshift(s) corresponding to the input comoving distance(s).

    Notes
    -----
    - The function uses a spline interpolation in log-log space for better accuracy 
      across a wide range of redshifts and distances.
    - If any values fall outside the initial interpolation range, the grid is automatically
      extended and the interpolation recalculated.
    """
    if cosmo is None:
        cosmo = cosmology.cosmo
    
    try:
        cdist = cdist.to('Mpc')
    except:
        cdist *= u.Mpc
        print('The comoving distances provided are assumed to be in Mpc units.')

    # Build interpolation from comoving distance to redshift
    z_grid = np.logspace(np.log10(zmin), np.log10(zmax), n_bins, base=10)
    r_grid = cosmo.comoving_distance(z_grid).to('Mpc').value
    r_to_z = splrep(np.log10(r_grid), np.log10(z_grid))

    z_arr = 10**splev(np.log10(cdist.value), r_to_z)
    
    recalculate = False
    if z_arr.min()<zmin:
        zmin = z_arr.min()/2
        recalculate = True
    if z_arr.max()>zmax:
        zmax = z_arr.max()*2
        recalculate = True

    if recalculate:
        z_arr = cdist_to_z(cdist, cosmo=cosmo, zmin=zmin, zmax=zmax, n_bins=n_bins)

    return z_arr