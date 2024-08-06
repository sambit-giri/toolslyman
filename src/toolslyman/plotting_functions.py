import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 
from scipy.ndimage import gaussian_filter

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 18

class Plot_PhotometricIGM():
    def __init__(self, tomo_map_reconst=None):
        self.tomo_map_reconst = tomo_map_reconst

    def sightline_map(self, cmap='viridis', **kwargs):
        print('#### Plot: sightline map #####')
        x_map = kwargs.get('x_map', self.tomo_map_reconst['x_map'])
        y_map = kwargs.get('y_map', self.tomo_map_reconst['y_map'])
        x_sample = kwargs.get('x_sample', self.tomo_map_reconst['x_sample'])
        y_sample = kwargs.get('y_sample', self.tomo_map_reconst['y_sample'])
        sightline_map = kwargs.get('sightline_map', kwargs.get('los_map', self.tomo_map_reconst['sightline_map']))

        fig,ax = plt.subplots(figsize=(7, 5))
        CS = ax.pcolormesh(x_map,y_map,sightline_map,cmap=cmap)
        cbar=fig.colorbar(CS)
        ax.set_aspect('equal')
        ax.plot(x_sample,y_sample,'ko',mfc='none',alpha=0.5)
        plt.tight_layout()
        plt.show(block=False)

    def true_Tigm_map(self, cmap='RdYlBu', **kwargs):
        print('#### Plot: Simulated IGM map (Truth) #####')
        sim = kwargs.get('photo_sim', self.tomo_map_reconst['photo_sim'])
        smoothing_length = self.tomo_map_reconst['IGM_tomography_params']['smoothing_length']

        fig,ax = plt.subplots(1,2, figsize=(10, 4))
        ax[0].set_title('Truth')
        pm=ax[0].pcolormesh(sim.x_mesh, sim.y_mesh, sim.TIGM.T, 
                        cmap=cmap, shading='auto')
        cbar=fig.colorbar(pm, label='${\\rm exp}(-\\tau_{\\rm eff})$')
        ax[0].set_aspect('equal')
        ax[1].set_title('Truth+smoothed')
        pm=ax[1].pcolormesh(sim.x_mesh, sim.y_mesh, gaussian_filter(sim.TIGM,smoothing_length).T, 
                        cmap=cmap, shading='auto')
        cbar=fig.colorbar(pm, label='${\\rm exp}(-\\tau_{\\rm eff})$')
        ax[1].set_aspect('equal')
        plt.tight_layout()
        plt.show(block=False)

    def reconstructed_Tigm_map(self, cmap='RdYlBu', **kwargs):
        x_map = kwargs.get('x_map', self.tomo_map_reconst['x_map'])
        y_map = kwargs.get('y_map', self.tomo_map_reconst['y_map'])
        sim = kwargs.get('photo_sim', self.tomo_map_reconst['photo_sim'])
        smoothing_length = self.tomo_map_reconst['IGM_tomography_params']['smoothing_length']
        R_FoV = self.tomo_map_reconst['IGM_tomography_params']['R_FoV']

        reconst_map = self.tomo_map_reconst['reconst_map']
        noisy_reconst_map = self.tomo_map_reconst['noisy_reconst_map']

        fig,axs = plt.subplots(1,2, figsize=(10, 4))

        #### Plot: Truth+smoothed
        ax = axs[0]
        ax.set_title('Truth+smoothed')
        dTIGM_reconst=reconst_map-np.mean(reconst_map)
        self.dTIGM_reconst = dTIGM_reconst
        CS = ax.pcolormesh(x_map,y_map,dTIGM_reconst,cmap=cmap,vmin=-0.1,vmax=0.1)#,vmin=0,vmax=0.13)
        # cax1 = fig.add_axes([0.78, 0.1, 0.03, 0.8])
        cbar=fig.colorbar(CS, #cax=cax1,
        #        ticks=arange(-0.05,0.05+0.01,0.01),
                label='$\\langle \\Delta T_{\\rm IGM}^{\\rm 2D}(x)\\rangle$'
                )
        # ax.plot(x_sample,y_sample,'ko',mfc='none',alpha=0.5)

        # survey footprint
        circ1=plt.Circle((sim.Lbox/2,sim.Lbox/2),R_FoV, fill=False, lw=1)
        ax.add_artist( circ1 )
        circ2=plt.Circle((sim.Lbox/2-R_FoV,sim.Lbox/2+R_FoV),R_FoV, fill=False, lw=1)
        ax.add_artist( circ2 )
        circ3=plt.Circle((sim.Lbox/2+R_FoV,sim.Lbox/2+R_FoV),R_FoV, fill=False, lw=1)
        ax.add_artist( circ3 )
        circ4=plt.Circle((sim.Lbox/2-R_FoV,sim.Lbox/2-R_FoV),R_FoV, fill=False, lw=1)
        ax.add_artist( circ4 )
        circ5=plt.Circle((sim.Lbox/2+R_FoV,sim.Lbox/2-R_FoV),R_FoV, fill=False, lw=1)
        ax.add_artist( circ5 )
        ax.set_aspect('equal')

        #### plot: Noisy+smoothed
        ax = axs[1]
        ax.set_title('Noisy+smoothed')
        dTIGM_noisy_reconst=noisy_reconst_map-np.mean(noisy_reconst_map)
        self.dTIGM_noisy_reconst = dTIGM_noisy_reconst
        CS = plt.pcolormesh(x_map,y_map,dTIGM_noisy_reconst,cmap=cmap,vmin=-0.1,vmax=0.1)#,vmin=0,vmax=0.13)
        # cax1 = fig.add_axes([0.78, 0.1, 0.03, 0.8])
        cbar=fig.colorbar(CS, #cax=cax1,
        #        ticks=arange(-0.05,0.05+0.01,0.01),
                label='$\\langle \\Delta T_{\\rm IGM}^{\\rm 2D}(x)\\rangle$'
                )
        # survey footprint
        circ1=plt.Circle((sim.Lbox/2,sim.Lbox/2),R_FoV, fill=False, lw=1)
        ax.add_artist( circ1 )
        circ2=plt.Circle((sim.Lbox/2-R_FoV,sim.Lbox/2+R_FoV),R_FoV, fill=False, lw=1)
        ax.add_artist( circ2 )
        circ3=plt.Circle((sim.Lbox/2+R_FoV,sim.Lbox/2+R_FoV),R_FoV, fill=False, lw=1)
        ax.add_artist( circ3 )
        circ4=plt.Circle((sim.Lbox/2-R_FoV,sim.Lbox/2-R_FoV),R_FoV, fill=False, lw=1)
        ax.add_artist( circ4 )
        circ5=plt.Circle((sim.Lbox/2+R_FoV,sim.Lbox/2-R_FoV),R_FoV, fill=False, lw=1)
        ax.add_artist( circ5 )
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.show(block=False)
