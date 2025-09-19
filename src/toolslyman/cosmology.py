import astropy
from astropy import cosmology as astropy_cosmo

# Define the global cosmology object at the module level
cosmo = None

def set_cosmology(name='Planck', **kwargs):
    global cosmo

    astropy_defined_cosmo = {
        'wmap1': astropy.cosmology.WMAP1,
        'wmap3': astropy.cosmology.WMAP3,
        'wmap5': astropy.cosmology.WMAP5,
        'wmap7': astropy.cosmology.WMAP7,
        'wmap9': astropy.cosmology.WMAP9,
        'planck13': astropy.cosmology.Planck13,
        'planck15': astropy.cosmology.Planck15,
        'planck18': astropy.cosmology.Planck18,
        'planck'  : astropy.cosmology.Planck18,
    }
    
    if isinstance(name, astropy.cosmology.Cosmology):
        cosmo = name
    elif name.lower() in astropy_defined_cosmo:
        cosmo = astropy_defined_cosmo[name.lower()]
    else:
        H0 = kwargs.get('H0', 100*kwargs.get('h0', kwargs.get('h', 0.67)))
        Om0 = kwargs.get('Om0', 0.31)
        Ob0 = kwargs.get('Ob0', 0.049)
        Tcmb0 = kwargs.get('Tcmb0', 2.725)
        cosmo = astropy.cosmology.FlatLambdaCDM(H0=H0, Om0=Om0, Ob0=Ob0, Tcmb0=Tcmb0, name=name)
    
    return cosmo

# Initialize the default cosmology at the module level
cosmo = set_cosmology(name='Planck')
