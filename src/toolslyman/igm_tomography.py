import numpy as np
from tqdm import tqdm 
try: from scipy.integrate import trapz
except: from scipy.integrate import trapezoid as trapz
import astropy.units as u
import astropy.constants as consts
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM,Planck18
from . import observational_toolkit
from . import source_model, telescope_filters

class PhotometricIGM():
    def __init__(self):
        print('--- Photometric IGM Tomography Toolkits ---')  

    def load_simulation(self, tau_eff='Davies18_simulation/taumap_30Mpc_fluct.fits', box_len=546.0):
        if isinstance(tau_eff,str):
            print('Loaded simulation from ... ', tau_eff)
            if '.fits' in tau_eff[-8:]:
                # If the file is in fits format
                hdul=fits.open(tau_eff)
                print('Keys:')
                print(' .tau_eff : simulated effective optical depth over 30cMpc length ')
                self.tau_eff=hdul[0].data # simulated effective optical depth over 30cMpc length
            elif '.npy' in tau_eff[-8:]:
                self.tau_eff = np.load(tau_eff)
            elif '.pkl' in tau_eff[-8:]:
                import pickle
                self.tau_eff = pickle.load(open(tau_eff,'rb'))
            else:
                print(f'Unknown Format')
                print('Either provide a numpy array or the following formats: fits, npy, pkl')
                pass
        else:
            self.tau_eff = tau_eff
        print('Lyman-alpha Transmission T_IGM : exp(-tau_eff) ')
        self.TIGM = np.exp(-self.tau_eff) 

        self.Lbox = box_len
        self.Ngrid = self.tau_eff.shape[0]
        self.dx_grid = self.Lbox/self.Ngrid
        self.x = np.linspace(0,self.Lbox,self.Ngrid)
        self.y = np.linspace(0,self.Lbox,self.Ngrid)
        self.x_mesh, self.y_mesh = np.meshgrid(self.x,self.y)

        print('Lbox [cMpc/h]= ', self.Lbox, '  Ngrid = ', self.Ngrid)

    def load_catalog(self, filename='Davies18_simulation/galaxy_slice_30Mpc_M18.dat',
                    format='ascii', names=['X [Mpc/h]','Y [Mpc/h]','Muv']):
        print('Loaded galaxy catalog from ... ', filename)
        self.catalog = Table.read(filename,format=format,names=names)

    def compute_lya_luminosity(self, redshift):
        import sys; sys.path.insert(1, '../model')
        import LAE_SOURCE_MODEL_FRAMWORK
        LAE_MODEL = LAE_SOURCE_MODEL_FRAMWORK.LAE_SOURCE_MODEL()

        print('Computing Lya luminosity based on Muv of simulated galaxy catalog ... ')
        N = len(self.catalog)
        La = np.logspace(40,45,500)
        La_sample = np.zeros(N)
        for i in tqdm(range(N)):
            Muv = self.catalog['Muv'][i]
            PDF_La = LAE_MODEL.PDF_La(La*u.erg/u.s ,Muv*u.mag, redshift)
            La_sample[i] = self.random_sampling(PDF_La,x=La, size=1)
        self.catalog.add_column(La_sample,name='La [erg/s]')


    def tomographic_map_reconstruction( self, x, y, smoothing_length,
                                        x_sample, y_sample, F_sample,
                                        apply_density_mask=False, density_level=10.0,
                                        verbose=False ):
        """
        2D tomographic map reconstruction using Gaussian kernel

        x (N_pixels, N_pixels) : x coordinates of desired IGM map 
        y (N_pixels, N_pixels) : y coordinates of desired IGM map
        smoothing_length : Gaussian smoothing length of map reconstruction
        x_sample (numpy.array with N sample): x coordinates of background galaxies with TIGM measurement
        y_sample (numpy.array with N sample): y coordinates of background galaxies with TIGM measurement
        F_sample (numpy.array with N sample): Measured TIGM values of background galaxies
        apply_density_mask (optional)
        density_level (optional)
        
        """
        import numpy.ma as ma
        def gaussian2d(x,y,x0,y0,s):
            gaussian2d = 1/(2*np.pi*s**2)*np.exp(-(x-x0)**2/(2*s**2)-(y-y0)**2/(2*s**2) )
            return gaussian2d
        print('2D tomographic map reconstruction ...')
        N_sample = F_sample.size
        reconst_map = np.zeros(x.shape)
        sightline_map = np.zeros(x.shape)
        for s in tqdm(range(N_sample)):
            if verbose: print('sample = ', s)
            reconst_map[:,:] += F_sample[s]*gaussian2d(x,y,x_sample[s],y_sample[s],smoothing_length)
            sightline_map[:,:] += gaussian2d(x,y,x_sample[s],y_sample[s],smoothing_length)
        reconst_field=reconst_map/sightline_map
        if apply_density_mask:
            mask = (sightline_map<density_level)
            masked_reconst_map = ma.masked_array(reconst_field, mask=mask)
            return masked_reconst_map, sightline_map
        else:
            reconst_map = reconst_field
            return reconst_map, sightline_map

    def variance_map_reconstruction( self, x, y, smoothing_length,
                                     x_sample, y_sample, var_sample,
                                     apply_density_mask=False, density_level=10.0):
        import numpy.ma as ma
        def gaussian2d(x,y,x0,y0,s):
            gaussian2d = 1/(2*np.pi*s**2)*np.exp(-(x-x0)**2/(2*s**2)-(y-y0)**2/(2*s**2) )
            return gaussian2d
        print('2D tomographic map reconstruction (variance image) ...')
        N_sample = var_sample.size
        variance_map = np.zeros(x.shape)
        sightline_map = np.zeros(x.shape)
        for s in tqdm(range(N_sample)):
            variance_map[:,:] += var_sample[s]*gaussian2d(x,y,x_sample[s],y_sample[s],smoothing_length)**2
            sightline_map[:,:] += gaussian2d(x,y,x_sample[s],y_sample[s],smoothing_length)
        return variance_map/(sightline_map**2)

    # Random sampling of UV magnitudes from a given probablity distribution 
    def random_sampling(self,pdf,x=None, size=100):
        """
        Randomly sample value, x_sample, from a given PDF, pdf, defined in range x
        """
        from scipy.interpolate import interp1d
        # get cumulative luminosity function and its inverse
        cdf = np.cumsum(pdf)
        cdf = cdf/cdf.max()
        inv_cdf = interp1d(cdf,x)
        # inverse transform sampling method to draw a sample
        uniform_sample = np.random.random(size=size)
        x_sample = inv_cdf(uniform_sample)
        return x_sample
    
    # Random sampling of photometric noise from Gaussian distribution
    def noise_sampling(self, noise_sigma, size=100):
        noise_flux = np.random.normal(0.0, noise_sigma.cgs.value, size=size)
        return noise_flux * u.erg/u.s/u.cm**2/u.Hz

    def get_background_galaxy_sample(self, sample_size=None, 
                                           dpdmuv=(None, None),
                                           mNB_lim=None,
                                           random_seed=123456):
        np.random.seed(random_seed)
        # get random skewers in the simulation box
        i_idx = np.random.randint(0, self.Ngrid, size=sample_size)
        j_idx = np.random.randint(0, self.Ngrid, size=sample_size)
        x_sample = self.x[i_idx]
        y_sample = self.y[j_idx]
        TIGM_sample = self.TIGM[i_idx,j_idx]

        # sample m_uv of background galaxies
        muv, pdf = dpdmuv
        muv_sample = self.random_sampling(pdf, x=muv, size=sample_size) * u.mag
        BB_flux = observational_toolkit.mag2flux(muv_sample) # ASSUMING FLAT UV CONTINUM !

        # sample NB flux noise
        NB_flux_sigma = observational_toolkit.mag2flux(mNB_lim)
        NB_flux_noise = self.noise_sampling(NB_flux_sigma,size=sample_size)

        # TIGM noise
        TIGM_noise = NB_flux_noise.cgs.value/BB_flux.cgs.value 

        # observed TIGM
        TIGM_obs = TIGM_sample+TIGM_noise

        # estimated Var[TIGM]
        TIGM_var = (NB_flux_sigma.cgs.value/BB_flux.cgs.value)**2

        return x_sample, y_sample, TIGM_sample, TIGM_obs, TIGM_noise, muv_sample, TIGM_var

def IGM_tomographic_map_reconstruction(box_len, n_grid, tau_eff,
                                       Nbin=100, cosmo=None,
                                       IGM_tomography_params=None, filter_name='NB816',
                                       muv_low=20*u.mag,
                                       ):
    ### IGM tomographic map reconstruction ###
    # Nbin: Define m_uv PDF from which m_uv of b/g galaxies are sampled

    if IGM_tomography_params is None:
        IGM_tomography_params = telescope_filters.IGM_tomography_parameters(box_len, filter_name=filter_name, cosmo=cosmo)

    BKG_MODEL = source_model.SOURCE_MODEL_FRAMEWORK(cosmo=cosmo)
    sim = PhotometricIGM()
    sim.load_simulation(tau_eff=tau_eff, box_len=box_len)

    zmin = IGM_tomography_params['zmin']
    zmax = IGM_tomography_params['zmax']
    mNB_lim = IGM_tomography_params['mNB_lim']
    muv_lim = IGM_tomography_params['muv_lim']
    smoothing_length = IGM_tomography_params['smoothing_length']
    Ngal_in_box = IGM_tomography_params['Ngal_in_box']

    # Define grid for IGM tomographic map reconstruction
    x_map, y_map = np.mgrid[0:box_len:n_grid*1j, 0:box_len:n_grid*1j]

    PDF=np.zeros(Nbin) *1/u.mag
    muv=np.linspace(muv_low, muv_lim, Nbin)
    for i in range(Nbin):
        PDF[i]=BKG_MODEL.dPdmuv(muv[i],muv_lim,zmin,zmax,norm=True)

    # Sample background galaxies and observed TIGM along the skewers
    x_sample, y_sample, TIGM_sample, TIGM_obs, TIGM_noise, muv_sample, TIGM_var = \
        sim.get_background_galaxy_sample(sample_size=Ngal_in_box, dpdmuv=(muv,PDF), mNB_lim=mNB_lim)

    # Truth: simulated IGM map with smoothing but no noise
    reconst_map, sightline_map = \
        sim.tomographic_map_reconstruction( x_map, y_map, smoothing_length,
                                            x_sample, y_sample, TIGM_sample )
    # Observed: mock IGM map with smoothing and photometric noise
    noisy_reconst_map, sightline_map = \
        sim.tomographic_map_reconstruction( x_map, y_map, smoothing_length,
                                            x_sample, y_sample, TIGM_obs )

    variance_map = \
        sim.variance_map_reconstruction( x_map, y_map, smoothing_length,
                                        x_sample, y_sample, TIGM_var )
    
    return {
        'photo_sim': sim,
        'IGM_tomography_params': IGM_tomography_params,
        'x_sample': x_sample,
        'y_sample': y_sample,
        'x_map': x_map,
        'y_map': y_map,
        'TIGM_sample': TIGM_sample,
        'TIGM_obs': TIGM_obs,
        'TIGM_noise': TIGM_noise,
        'muv_sample': muv_sample,
        'TIGM_var': TIGM_var,
        'reconst_map': reconst_map, 
        'sightline_map': sightline_map,
        'noisy_reconst_map': noisy_reconst_map,
        'variance_map': variance_map,
    }