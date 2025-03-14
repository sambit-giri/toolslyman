import numpy as np
from scipy import special

def compute_spectrum_after_IGM_transmission(xvel_in,xvel_out,cdens,temp,lambda0,fvalue,mass,damped,periodic):
    '''
    Returns optical depth array

    xvel_in  : velocity (km/s) of array element (x-coordinate of spectrum)
    xvel_out : velocity (km/s) of array element (x-coordinate of spectrum)
    cdens    : column density (particles/cm^2)
    temp     : temperature (K)
    lambda0  : rest wavelength (Å)
    fvalue   : oscillator strength
    mass     : mass of atom (g)
    
    '''
  
    if damped:
        #    IF round(lambda0) NE 1216. THEN $
        #      message,'Damping wings only possible for HI Ly-alpha!'
        gf_Lya = 0.8323
        g2_Lya = 8.
        # Natural line-width (km/s)
        v_Lya = 0.6679e-5 * gf_Lya / (g2_Lya*(lambda0*1.e-8)*4.*np.pi) 
        v_lya = 0.00606076 # in km/s. Note that the value above is incorrect
        print(' including a damping wing')
        
        
    minbother = 1.e-2               # Min. max. optical depth for inclusion
        
    c = 2.9979e10                   # cm/s
    kboltz = 1.3807e-16             # erg/K
    sigma_T = 6.6525e-25            # cm^2
    
    # Cross section in cm^2: 
    sigma_0 = np.sqrt(3.*np.pi*sigma_T/8.) * 1.e-8 * lambda0 * fvalue
    
    Tpar = 2.0 * kboltz / mass      # erg/K/g
    Cpar = sigma_0 * c              # cm^3/s
    
    nveloc_in = xvel_in.size
    nveloc_out = xvel_out.size
    nveloc1_out = nveloc_out - 1
    
    tau = np.zeros(nveloc_out)
    tauv = np.zeros(nveloc_out)
    
    taumin = minbother / nveloc_in
    
    bpar_inv = 1. / np.sqrt(Tpar * temp) # s/cm
    tauc = Cpar * cdens * bpar_inv  # Central optical depth
    bpar_inv = bpar_inv * 1.e5      # s/km
    
    if damped: apar = v_lya * bpar_inv
    if periodic:
        boxkms  = max(xvel_in)
        boxkms2 = 0.5*boxkms
        
    for i in range(nveloc_in):
        if tauc[i] >= taumin:
            vpar = abs(xvel_out - xvel_in[i])
            if periodic:
                nn = np.where(vpar > boxkms2)
                while (nn.size > 0):
                    print,' count = ',nn.size
                    vpar[nn]=abs(vpar[nn]-boxkms)
                    nn = np.where(vpar > boxkms2)
                    
            vpar = vpar * bpar_inv[i]
            if damped:
                # The voigt_profile function from scipy and the voigt function
                # from IDL map according to
                #    voigt(gamma,x)/sqrt(pi)=voigt_profile(x,sqrt(0.5),gamma).
                # See
                # https://www.nv5geospatialsoftware.com/docs/VOIGT.html and
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.voigt_profile.html
                #dtau = tauc[i] * voigt(apar[i],vpar) / np.sqrt(np.pi)
                dtau = tauc[i] * special.voigt_profile(vpar,np.sqrt(0.5),apar[i])
            else:
                dtau = tauc[i] * np.exp(-vpar*vpar) / np.sqrt(np.pi)
                                
            tau = tau + dtau                            

    return tau

