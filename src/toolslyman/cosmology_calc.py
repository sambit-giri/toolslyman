import numpy as np
from astropy import units as u
from scipy.interpolate import splev, splrep
from . import cosmology

def cdist_to_z(cdist, cosmo=None, zmin=1e-4, zmax=1100, n_bins=1000):
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