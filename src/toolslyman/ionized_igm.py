import numpy as np
from glob import glob
from tqdm import tqdm
# import astropy
# from astropy import constants, units
# from astropy.cosmology import Planck18

def make_zion_field(filenames_or_dict, xion_thres=0.99, file_redshifts=None, reading_function=None):
    '''
    Make a lightcone from xfrac, density or dT data. Replaces freq_box.
    
    Parameters:
        filenames_or_dict (string or dict): The coeval cubes of ionization fields.
            Can be either any of the following:
            
                - An array with the file names
                
                - A dictionary containing ionization fields with redshifts as keys
        
        xion_thres (float): The threshold value of x_HII to use to define completion of ionisation.
        file_redshifts (array): The redshifts of the coeval cubes if provided as file names.
        reading_function (function): a function to read the files if file names are provided.

    Returns:
        zion array
    '''
    if isinstance(filenames_or_dict, dict):
        xfrac_dict = filenames_or_dict
        xfrac_reds = np.array([ke for ke in xfrac_dict.keys()])
    else:
        assert len(file_redshifts) == len(filenames_or_dict)
        xfrac_reds = file_redshifts
        xfrac_dict = {}
        print('Reading ionisation data...')
        for ii,zi in tqdm(enumerate(xfrac_reds)):
            xfrac_dict[zi] = reading_function(filenames_or_dict[ii])
        print('...done')
    print('MAking the z_ion field...')
    zion = np.ones_like(xfrac_dict[xfrac_reds[0]])
    xfrac_reds = np.sort(xfrac_reds)
    for ii,zi in tqdm(enumerate(xfrac_reds)):
        xi = xfrac_dict[zi]
        zion[xi>=xion_thres] = zi
    print('...done') 
    return zion

def gas_temp_inside_ionized_igm_single(z, dens, z_ion, dens_ion, cosmo=None):
    '''
    Approximate temperature of the ionized regions using the fit from McQuinn & Upton Sanderbeck (1505.07875).
    '''
    if cosmo is None:
        print('Assuming Planck18 cosmology.')
        cosmo = Planck18

    LL = lambda z: (1+z)/7.1
    L = LL(z)
    L_ion = LL(z_ion)
    Tion_I = 2e4*units.K
    Tlim = 1.775*L*1e4*units.K
    gamma = 1.7

    Tkgamma = Tion_I**gamma*((L/L_ion)**3*dens/dens_ion)**(2*gamma/3)*np.exp(L**2.5)/np.exp(L_ion**2.5)+Tlim**gamma*dens/dens.mean(dtype=np.float64)
    return Tkgamma**(1/gamma)

