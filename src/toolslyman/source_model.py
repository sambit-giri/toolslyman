import numpy as np
import astropy.units as u
import astropy.constants as consts
from astropy.cosmology import FlatLambdaCDM,Planck18
from scipy.integrate import trapz

class SOURCE_MODEL_FRAMEWORK():
    def __init__(self, cosmo=None):
        self.cosmo = Planck18 if cosmo is None else cosmo

    def FoV_cMpc2(self, FoV, redshift):
        # FoV [deg2] to FoV [cMpc^2]
        D = self.cosmo.comoving_distance(redshift)
        FoV_cMpc2 = FoV.to('rad2').value*D**2
        return FoV_cMpc2
    
    def mag2flux(self, mag):
        # apparent magnitude to flux
        flux = 10**(-(mag.to('mag').value+48.60)/2.5) * u.erg/u.s/u.cm**2/u.Hz
        return flux
    
    def Muv2muv(self, Muv, z):
        # convert absolute to apparent magnitude
        Kcorrection = 2.5*np.log10(1+z) 
        m_uv = Muv + self.cosmo.distmod(z).value-Kcorrection
        return m_uv
    
    def muv2Muv(self, muv, z):
        # convert apparent to absolute magnitude
        Kcorrection = 2.5*np.log10(1+z) * u.mag
        Muv = muv-self.cosmo.distmod(z) + Kcorrection
        return Muv
    
    def Muv2L1500(self, M_AB):
        # absolute magnitude to luminosity [erg/s/Hz]
        SI2cgs = 1e7 # 1 W/Hz=1e7
        L1500 = SI2cgs*10.**((34.1-M_AB.to('mag').value)/2.5) * u.erg/u.s/u.Hz # erg/s/Hz
        return L1500
    
    def angular_bins(self, r_min, r_max, bins, type='log-bin'):
        # Angular bins
        if type=='linear-bin':
            r_edges=np.linspace(r_min, r_max, bins+1)
        if type=='log-bin':
            r_edges = np.logspace(np.log10(r_min),
                                np.log10(r_max),
                                bins+1)
        r_bins  = np.zeros(bins)
        dr_bins = np.zeros(bins)
        for i in range(bins):
            r_bins[i]  = (r_edges[i]+r_edges[i+1])/2
            dr_bins[i] = r_edges[i+1]-r_edges[i]
        return r_bins, r_edges, dr_bins
    
    def dndMuv(self,Muv,z): 
        # Bouwens+21 z~2-10 UV-LF
        # UV luminosity function [1/cMpc3]
        zt = 2.46
        if z<zt:
            Muv_c = ( -20.89-1.09*(z-zt) ) * u.mag # [mag]
        else:
            Muv_c = ( -21.03-0.04*(z-6.) ) * u.mag # [mag]
        phi_c = 0.40e-3 * 10.0**( -0.33*(z-6.0)-0.024*(z-6.0)**2 ) * 1/u.Mpc**3 # 1/cMpc3
        faintend_slope = -1.94-0.11*(z-6.0)
        dndMuv = phi_c*(np.log(10.)/2.5)*10.0**(-0.4*(Muv-Muv_c)/u.mag*(faintend_slope+1))*np.exp(-10.**(-0.4*(Muv-Muv_c)/u.mag))
        return dndMuv / u.mag
    
    def LBG_surface_density(self, muv_lim, z1, z2):
        # LBG surface number density (<muv) [1/cMpc2]
        z = np.linspace(z1,z2,100)
        drdz = consts.c/self.cosmo.H(z) # cMpc
        dndz = np.zeros(z.size)* 1/u.Mpc**2
        for i in range(z.size):
            Muv_lim = self.muv2Muv(muv_lim,z[i])
            Muv = np.linspace(-30*u.mag, Muv_lim, 100)
            dndz[i] = drdz[i]*trapz( self.dndMuv(Muv,z[i]), Muv )
        LBG_surface_density = trapz( dndz, z )
        return LBG_surface_density
    
    def LBG_number_count(self,muv_lim,z1,z2,redshift=None,FoV=None):
        # LBG number count in FoV 
        return self.FoV_cMpc2(FoV,redshift)*self.LBG_surface_density(muv_lim,z1,z2)

    def dPdmuv(self,muv,muv_lim,z1,z2,norm=True):
        # Apparent UV mag PDF for background galaxies (z1<z<z2), dn_2D/dmuv
        z = np.linspace(z1,z2,100)
        drdz = consts.c/self.cosmo.H(z) # cMpc
        dndmdz = np.zeros(z.size)*1/u.Mpc**2/u.mag
        for i in range(z.size):
            Muv = self.muv2Muv(muv,z[i])
            dndmdz[i] = drdz[i]*self.dndMuv(Muv,z[i])
        # dPdm = trapz(dndmdz, z) 
        dndm = trapz(dndmdz, z)  # [1/cMpc2/mag]
        if norm:
            dPdm = dndm / self.LBG_surface_density(muv_lim,z1,z2)
            return dPdm # [1/mag], Normalised PDF (default)
        else: 
            return dndm # [1/cMpc2/mag] Non-normalised PDF

    def Var_TIGM_individual(self, muv_lim, mNB_lim, z1, z2 ):
        # Expectation value of Var[TIGM] for a typical background galaxy
        # m_uv grid
        muv=np.linspace(22*u.mag, muv_lim, 100)
        # TIGM error per background source
        NB_flux_error=self.mag2flux(mNB_lim)
        UV_flux = self.mag2flux(muv)
        dTIGM = NB_flux_error/UV_flux
        # Computing the expectation value of TIGM error
        dTIGM2dmuv = np.zeros(muv.size)  * 1/(u.Mpc**2*u.mag)
        for i in range(muv.size):
            dTIGM2dmuv[i] = dTIGM[i]**2 * self.dPdmuv(muv[i],muv_lim,z1,z2,norm=False)
        Var_TIGM = trapz(dTIGM2dmuv, muv ) / self.LBG_surface_density(muv_lim,z1,z2)
        return Var_TIGM
    
    def Var_TIGM_around_galaxies(self, r_edges, N_fg, muv_lim, mNB_lim, z1, z2):
        # Var[TIGM]
        # angular bins
        r_bins=(r_edges[1:]+r_edges[:-1])/2
        A_bins=np.pi*(r_edges[1:]**2-r_edges[:-1]**2)
        # typical TIGM variance of a background source (expectation value)
        Var_TIGM_typical = self.Var_TIGM_individual(muv_lim, mNB_lim, z1, z2)
        print('typical TIGM error of a bkg. source = ', np.sqrt(Var_TIGM_typical) )
        # background source surface density
        n2d=self.LBG_surface_density(muv_lim,z1,z2)
        # Annulus area and numberof pairs per bin
        Npairs=N_fg*A_bins.to('Mpc2')*n2d.to('1/Mpc2')
        # variance of mean TIGM around fg. galaxies
        Var_TIGM_around_galaxies = Var_TIGM_typical/Npairs
        return Var_TIGM_around_galaxies, r_bins, Npairs   
    
if __name__ == "__main__":
    # main
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 18

    # Import HSC toolkits
    import sys; sys.path.insert(1, '../analysis')
    import HSC_toolkits; hsc=HSC_toolkits.HSC_toolkits()
    import LAE_SOURCE_MODEL_FRAMWORK

    BKG = SOURCE_MODEL_FRAMEWORK(cosmo=Planck18)

    # define LAE source model
    LAE_MODEL = LAE_SOURCE_MODEL_FRAMWORK.LAE_SOURCE_MODEL()
    # Define foreground NB center
    NB816 = hsc.filter(filter='NB816')
    zmin_fg = NB816['z_min']
    zmax_fg = NB816['z_max']

    # Define redshift of interest and bkg. redshift for NB816 IGM tomography
    redshift = NB816['z_peak']
    zmin = 5.81
    zmax = 6.86

    # Define FoV of survey
    FoV_HSC = np.pi*(90*u.arcmin/2)**2        # HSC FoV
    print('HSC FoV [deg2] = ', FoV_HSC.to('deg^2'))

    # Define foreground LAE number and limiting magnitudes
    La_lim  = 10**42.8 *u.erg/u.s
    n2d_LAE = LAE_MODEL.foreground_LAE_surface_density(La_lim, zmin_fg, zmax_fg)
    n2d_LAE*BKG.FoV_cMpc2(FoV_HSC,redshift)
    N_fg = n2d_LAE*BKG.FoV_cMpc2(FoV_HSC,redshift)   #263
    muv_lim = 25.5*u.mag
    mNB_3sigma = 27.5*u.mag
    mNB_lim = mNB_3sigma+2.5*np.log10(3/1)*u.mag
    print('fg. LAE count = %i' % N_fg)

    # Set angular bins
    # bins = 10
    # min_bin = 0.1 # Mpc 
    # max_bin = 200 # Mpc
    # bin_type = 'log-bin'
    # r_bins,r_edges, dr_bins = BKG.angular_bins(min_bin,max_bin,bins,type=bin_type)
    # r_edges = r_edges*u.Mpc

    bins = 10
    min_bin = 0.1*u.arcmin #1*u.arcmin
    max_bin = 30*u.arcmin
    bin_type = 'log-bin'
    theta_bins,theta_edges = hsc.angular_bins(min_bin,max_bin,bins,type=bin_type)
    D = BKG.cosmo.comoving_distance(redshift)
    r_edges = theta_edges.to('rad').value*D

    # parent galaxy density
    muv = np.linspace(23.5,26.5) * u.mag
    n2d_LBG = np.zeros(muv.size)*1/u.Mpc**2
    N_LBG = np.zeros(muv.size)
    for i in range(muv.size):
        n2d_LBG[i] = BKG.LBG_surface_density(muv[i], zmin, zmax)
        N_LBG[i] = BKG.LBG_number_count(muv[i], zmin, zmax, redshift=redshift, FoV=FoV_HSC)

    # Compute TIGM variance around galaxies
    TIGM_mean=np.exp(-hsc.tau_eff(redshift,type='Bosman21'))
    Var_TIGM, R_bins, Npairs = BKG.Var_TIGM_around_galaxies(r_edges, N_fg, muv_lim, mNB_lim, zmin, zmax)
    Var_CCF=Var_TIGM/TIGM_mean**2

    # Plot
    fig,ax=plt.subplots(1,3,figsize=(15,5))
    
    # background LBG surface density
    ax[0].semilogy(muv, n2d_LBG/BKG.cosmo.h**2, 'r-', lw=2)
    ax[0].set_xlim(26.5,23.5)
    ax[0].set_ylim(5e-5,1)
    ax[0].set_xlabel('$m_{\\rm UV}$ $\\rm [mag]$')
    ax[0].set_ylabel('$\\Sigma_{\\rm LBG}(<m_{\\rm UV})$ $[(h^{-1}\\rm cMpc)^2]$')
    
    # background LBG number cound
    ax[1].semilogy(muv, N_LBG, 'r-', lw=2)
    ax[1].set_xlim(26.5,23.5)
    ax[1].set_xlabel('$m_{\\rm UV}$ $\\rm [mag]$')
    ax[1].set_ylabel('Number count')
    
    # Error on galaxy-Lya forest cross-correlation
    ax[2].loglog(R_bins, np.sqrt(Var_CCF),'ko-')
    ax[2].set_ylabel('$\\sqrt{{\\rm Var}[\,\\omega_{\\rm g\\alpha}(\\theta)]}$')
    ax[2].set_xlabel('$R_\\perp$ [cMpc]')

    label='$z=%.2f$ $(%.2f<z_{bg}<%.2f)$\n$m_{\\rm UV}(5\\sigma)=%.2f$\n$m_{\\rm NB}(3\\sigma)=%.2f$' % (redshift,zmin,zmax,muv_lim.value,mNB_3sigma.value)
    ax[2].text(2,0.5,label,fontsize=14)

    plt.tight_layout()
    plt.show(block=False)

    plt.savefig('error_forecast_z5.7.png',format='png') 