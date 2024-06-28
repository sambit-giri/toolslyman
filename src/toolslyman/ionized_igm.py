import numpy as np
from glob import glob
from tqdm import tqdm
import astropy
from astropy import constants, units
from astropy.cosmology import Planck18

def make_zion_field(filenames_or_dict, xion_thres=0.99, file_redshifts=None, reading_function=None):
    """
    Create a z_ion field where pixels define the redshift when it was ionised.

    Parameters:
    -----------
    filenames_or_dict : str or dict
        The coeval cubes of ionization fields. Can be one of the following:
        - An array of file names
        - A dictionary containing ionization fields with redshifts as keys
        
    xion_thres : float, optional
        The threshold value of x_HII to use to define completion of ionization. Default is 0.99.
        
    file_redshifts : array-like, optional
        The redshifts of the coeval cubes if provided as file names.
        
    reading_function : function, optional
        A function to read the files if file names are provided.

    Returns:
    --------
    zion : ndarray
        The redshift of ionization field.
    """
    if isinstance(filenames_or_dict, dict):
        xfrac_dict = filenames_or_dict
        xfrac_reds = np.array(list(xfrac_dict.keys()))
    else:
        assert len(file_redshifts) == len(filenames_or_dict), "Number of file names and redshifts must match."
        xfrac_reds = file_redshifts
        xfrac_dict = {}
        print('Reading ionization data...')
        for ii, zi in tqdm(enumerate(xfrac_reds)):
            xfrac_dict[zi] = reading_function(filenames_or_dict[ii])
        print('...done')
        
    print('Making the z_ion field...')
    zion = np.ones_like(xfrac_dict[xfrac_reds[0]])
    xfrac_reds = np.sort(xfrac_reds)
    for ii, zi in tqdm(enumerate(xfrac_reds)):
        xi = xfrac_dict[zi]
        zion[xi >= xion_thres] = zi
    print('...done')
    
    return zion

def value_at_zion(value_dict, z_ion):
    """
    Determine the value of each pixel at its redshift of ionization.

    Parameters:
    -----------
    value_dict : dict
        A dictionary containing fields with redshifts as keys.
        
    z_ion : array
        The redshift of ionization field.

    Returns:
    --------
    value_ion : ndarray
        The values at the redshift of ionization.
    """
    reds = np.sort(np.array(list(value_dict.keys())))
    value_ion = np.zeros_like(value_dict[reds[0]])
    print('Making the field with values at the redshift of ionization...')
    for ii, zi in tqdm(enumerate(reds)):
        value_ion[zi == z_ion] = value_dict[zi][zi == z_ion]
    print('...done')
    return value_ion

def gas_temp_inside_ionized_igm_single(z, dens, z_ion, dens_ion, cosmo=None):
    """
    Approximate temperature of the ionized regions using the fit from McQuinn & Upton Sanderbeck (2015).

    Parameters:
    -----------
    z : float
        Redshift.
        
    dens : ndarray
        Density field.
        
    z_ion : ndarray
        Redshift of ionization field.
        
    dens_ion : ndarray
        Density at the redshift of ionization.
        
    cosmo : astropy.cosmology.Cosmology, optional
        Cosmology object. If None, assumes Planck18 cosmology.

    Returns:
    --------
    Tkgamma : ndarray
        Temperature of the ionized regions.
    """
    if cosmo is None:
        print('Assuming Planck18 cosmology.')
        cosmo = Planck18

    LL = lambda z: (1+z)/7.1
    L = LL(z)
    L_ion = LL(z_ion)
    Tion_I = 2e4*units.K
    Tlim = 1.775*L*1e4*units.K
    gamma = 1.7

    Tkgamma = Tion_I**gamma * ((L/L_ion)**3 * dens/dens_ion)**(2*gamma/3) * np.exp(L**2.5) / np.exp(L_ion**2.5) + Tlim**gamma * dens/dens.mean(dtype=np.float64)
    return Tkgamma**(1/gamma)

def gas_temp_inside_ionized_igm(dens_dict, z_ion=None, dens_ion=None, xfrac_dict=None, xion_thres=0.99, cosmo=None):
    """
    Approximate temperature of the ionized regions using the fit from McQuinn & Upton Sanderbeck (2015).

    Parameters:
    -----------
    dens_dict : dict
        A dictionary containing density fields with redshifts as keys.
        
    z_ion : ndarray, optional
        The redshift of ionization field. If None, it will be created from xfrac_dict.
        
    dens_ion : ndarray, optional
        Density at the redshift of ionization. If None, it will be computed from dens_dict and z_ion.
        
    xfrac_dict : dict, optional
        A dictionary containing ionization fraction fields with redshifts as keys.
        
    xion_thres : float, optional
        The threshold value of x_HII to use to define completion of ionization. Default is 0.99.
        
    cosmo : astropy.cosmology.Cosmology, optional
        Cosmology object. If None, assumes Planck18 cosmology.

    Returns:
    --------
    temp_dict : dict
        A dictionary containing the temperature of the ionized regions with redshifts as keys.
    """
    if cosmo is None:
        print('Assuming Planck18 cosmology.')
        cosmo = Planck18

    assert z_ion is not None or xfrac_dict is not None, "Either provide z_ion or xfrac_dict to construct z_ion."
    
    if z_ion is None:
        z_ion = make_zion_field(xfrac_dict, xion_thres=xion_thres)
    if dens_ion is None:
        dens_ion = value_at_zion(dens_dict, z_ion)
    
    reds = np.sort(np.array(list(dens_dict.keys())))
    print('Estimating the temperature inside ionized IGM at different redshifts...')
    temp_dict = {}
    for ii, zi in tqdm(enumerate(reds)):
        Tki = gas_temp_inside_ionized_igm_single(zi, dens_dict[zi], z_ion, dens_ion, cosmo=cosmo)
        temp_dict[zi] = Tki 
    print('...done')
    
    return temp_dict


