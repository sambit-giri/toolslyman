import numpy as np
from tqdm import tqdm 

def angular_bins(min_bin,max_bin,bins,type='linear-bin'):  
    """
    Define angular bins for 2D profile analysis

    min_bin : minimum angular scale
    max_bin : maximum angular scale
    bins    : number of angular bins
    type    : 'linear-bin' or 'log-bin'

    return theta_bins, theta_edges
    """

    if type=='linear-bin':
        theta_edges=np.linspace(min_bin,max_bin,bins+1)
    if type=='log-bin':
        theta_edges=np.logspace(np.log10(min_bin),
                                np.log10(max_bin),
                                bins+1)
        theta_edges=theta_edges
    theta_bins=np.zeros(bins)
    for i in range(bins):
        theta_bins[i]=(theta_edges[i]+theta_edges[i+1])/2
    return theta_bins, theta_edges

def mean_2D_profile_around_objects(x_objects=None, y_objects=None, n_objects=None,
                                    x_sightlines=None, y_sightlines=None, weights=None,
                                    min_bin=0.1, max_bin=10.0, bins=100, type='log-bin'):
    """
    Compute mean 2D profile around foreground objects
    
    x_objects (numpy.array with N objects): x coordinates of foreground objects
    y_objects (numpy.array with N objects): y coordinates of foreground objects
    n_objects (numpy.array with N objects): number of foreground objects at each position
    x_sightlines (numpy.array with N sample): x coordinates of background sightlines
    y_sightlines (numpy.array with N sample): y coordinates of background sightlines
    weights (numpy.array with N sample): weight of background sightlines (e.g., TIGM)
    min_bin (float): minimum angular scale
    max_bin (float): maximum angular scale
    bins (int): number of angular bins
    type (str): 'linear-bin' or 'log-bin'

    return mean_flux, npairs, theta_bins
    """
    theta_bins, theta_edges = angular_bins(min_bin, max_bin, bins, type=type)
    mean_flux=np.zeros(bins)
    npairs=np.zeros(bins)
    print('computing an angular mean profile ...')
    N_objects=len(x_objects)
    for i in tqdm(range(N_objects)):
        theta = np.sqrt((x_sightlines-x_objects[i])**2 + (y_sightlines-y_objects[i])**2)
        for n in range(bins):
            in_bin=( (theta>=theta_edges[n]) & (theta< theta_edges[n+1]) )
            mean_flux[n]+=np.sum(weights[in_bin]*n_objects[i])
            npairs[n]+=weights[in_bin].size*n_objects[i]
    mean_flux=mean_flux/npairs
    return(mean_flux, npairs, theta_bins)

def mean_2D_profile_around_objects_in_2D_map(LAEs, TIGM, Lbox=200, Ngrid=None):
    if Ngrid is None: Ngrid = TIGM.shape[0]

    try:
        # Unpack and format data
        x_LAE, y_LAE = np.nonzero(LAEs)
        n_LAE = LAEs[x_LAE, y_LAE]
        x_LAE = x_LAE * Lbox / Ngrid
        y_LAE = y_LAE * Lbox / Ngrid
    except:
        x_LAE, y_LAE, n_LAE = LAEs

    x_sightline, y_sightline = np.nonzero(TIGM)
    TIGM_sightline = TIGM[x_sightline, y_sightline]
    x_sightline = x_sightline * Lbox / Ngrid
    y_sightline = y_sightline * Lbox / Ngrid

    # Compute angular-averaged IGM transmission around LAEs 
    mean_flux, npairs, r_bins = mean_2D_profile_around_objects(
        x_objects=x_LAE, y_objects=y_LAE, n_objects=n_LAE,
        x_sightlines=x_sightline, y_sightlines=y_sightline, weights=TIGM_sightline,
        min_bin=1.0, max_bin=300.0, bins=15, type='log-bin'
        )
    return mean_flux, npairs, r_bins

def mean_2D_profile_around_objects_in_2D_map_err(LAEs, varTIGM, Lbox=200, Ngrid=None):
    mean_flux, npairs, r_bins = mean_2D_profile_around_objects_in_2D_map(LAEs, varTIGM, Lbox=Lbox, Ngrid=Ngrid)
    return np.sqrt(mean_flux/npairs), npairs, r_bins