import numpy as np
from tqdm import tqdm 
from astropy.convolution import convolve_fft
from scipy.signal import fftconvolve

def profile_to_3Dkernel(profile, n_grid, box_len):
    """
    Generate a 3D kernel from a radial profile function.

    Parameters
    ----------
    profile : function
        A function representing the profile that depends on radius.
    n_grid : int
        Number of grid points along each dimension.
    box_len : float
        Size of the box in cMpc.

    Returns
    -------
    np.ndarray
        A 3D array of size (n_grid, n_grid, n_grid) with the profile centered.
    """
    x = np.linspace(-box_len / 2, box_len / 2, n_grid)
    rx, ry, rz = np.meshgrid(x, x, x, sparse=True)
    rgrid = np.sqrt(rx**2 + ry**2 + rz**2)
    kern = profile(rgrid)
    return kern

def put_profiles_group(source_pos, nbr_of_halos, profile_kern, n_grid=None):
    """
    Place halo profiles on a 3D grid.

    Parameters
    ----------
    source_pos : array-like
        The positions of halo centers in units of grid cells. Shape is (3, N), where N is the number of halos (X, Y, Z).
    nbr_of_halos : array-like
        The number of halos at each source position, array of size len(source_pos).
    profile_kern : np.ndarray
        The profile kernel to place on the grid around each source position, multiplied by nbr_of_halos (output of profile_to_3Dkernel).
    n_grid : int, optional
        The number of grid points along each dimension. If None, it defaults to the size of profile_kern.

    Returns
    -------
    np.ndarray
        A 3D grid with the halo profiles applied.
    """
    if n_grid is None:
        n_grid = profile_kern.shape[0]

    source_grid = np.zeros((n_grid, n_grid, n_grid))
    source_grid[source_pos[0], source_pos[1], source_pos[2]] = nbr_of_halos

    out = convolve_fft(source_grid, profile_kern, boundary='wrap', normalize_kernel=False, allow_huge=True)
    return out

def put_sphere_group(source_pos, Rmax_list, n_grid):
    """
    Place spherical profiles on a 3D grid.

    Parameters
    ----------
    source_pos : array-like
        The positions of halo centers in units of grid cells. Shape is (3, N), where N is the number of halos (X, Y, Z).
    Rmax_list : array-like
        Radii of the spheres in grid units.
    n_grid : int
        The number of grid points along each dimension.

    Returns
    -------
    np.ndarray
        A 3D grid with the halo profiles applied.
    """
    out = np.zeros((n_grid, n_grid, n_grid))
    for rmax in np.unique(Rmax_list):
        profile_ball = lambda rr: 1-sigmoid_func(rr, rmax, 2)
        profile_kern = profile_to_3Dkernel(profile_ball, n_grid, n_grid)
        source_grid = np.zeros((n_grid, n_grid, n_grid))
        source_pos_rmax = source_pos[Rmax_list==rmax,:]
        source_grid[source_pos_rmax[:,0], source_pos_rmax[:,1], source_pos_rmax[:,2]] = 1
        out_rmax = convolve_fft(source_grid, profile_kern, boundary='wrap', normalize_kernel=False, allow_huge=True)
        out_rmax = np.clip(out_rmax, 0, 1)
        out += out_rmax
    out = np.clip(out, 0, 1)
    return out

def sigmoid_func(x, x0, b):
    """
    Sigmoid function used to model transition.

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s).
    x0 : float
        Midpoint of the sigmoid.
    b : float
        Steepness of the sigmoid.

    Returns
    -------
    float or np.ndarray
        Output value(s) of the sigmoid function.
    """
    S = 1/(1+np.exp(-b*(x-x0)))
    return S


class SphereModel:
    '''
    Modelling reionization bubbles with spheres.
    '''
    def __init__(self, n_grid=256, box_len=300):
        """
        Initialize the model with grid size and box length.

        Parameters
        ----------
        n_grid : int, optional
            Number of grids along each dimension, default is 256.
        box_len : float, optional
            Length of the box in Mpc, default is 300.
        """
        self.n_grid = n_grid #Number of grids
        self.box_len = box_len #box-length in Mpc
        self.draw_R = None
        self.draw_source_pos = None 

    def set_ionization_history(self, zs=None, xHI=None):
        """
        Set the ionization history.

        Parameters
        ----------
        zs : array-like, optional
            Redshifts.
        xHI : array-like, optional
            Mean neutral fraction.
        """
        if zs is None or xHI is None:
            zs = np.arange(5.5,11.5,0.1)
            xHI = sigmoid_func(zs, 8, 1.5)
            print('A sigmoid shaped reionization history is assumed with a mid-point at z=8.')
            print('To any other history, provide redshifts and mean neutral fraction through zs and xHI parameters, respectively.')
        self.zs = zs 
        self.xHI = xHI 
    
    def set_lognormal_bsd(self, logR_mean=2.303, logR_std=1):
        """
        Set bubble size distribution to a log-normal distribution.

        Parameters
        ----------
        logR_mean : float, optional
            Mean of the log-normal distribution, default is 2.303.
        logR_std : float, optional
            Standard deviation of the log-normal distribution, default is 1.
        """
        self.draw_logR = lambda N=1: np.random.normal(logR_mean, logR_std, N)
        self.draw_R = lambda N=1: np.exp(self.draw_logR(N))
        print(f'Bubble sizes will be drawn randomly from a log-normal distribution with peak position at log(R/Mpc)={logR_mean:.3f} (R={np.exp(logR_mean):.3f} and distribution width of {logR_std:.3f}.')

    def set_uniform_bsd(self, logR_min=0, logR_max=3):
        """
        Set bubble size distribution to a uniform distribution.

        Parameters
        ----------
        logR_min : float, optional
            Minimum size of the bubbles in log scale, default is 0.
        logR_max : float, optional
            Maximum size of the bubbles in log scale, default is 3.
        """
        self.draw_logR = lambda N=1: np.random.uniform(logR_min, logR_max, N)
        self.draw_R = lambda N=1: np.exp(self.draw_logR(N))
        print(f'Bubble sizes will be drawn randomly from a uniform distribution with minimum and maximum sizes of log(R/Mpc)={logR_min:.3f} (R={np.exp(logR_min):.3f} Mpc) and {logR_max:.3f} (R={np.exp(logR_max):.3f} Mpc), respectively.')

    def set_constant_bsd(self, logR=2.303):
        """
        Set bubble size to a constant value.

        Parameters
        ----------
        logR : float, optional
            Constant size of the bubbles in log scale, default is 2.303.
        """
        self.draw_logR = lambda N=1: np.ones(N)*logR
        self.draw_R = lambda N=1: np.exp(self.draw_logR(N))
        print(f'Bubble sizes set to constant value of log(R/Mpc)={logR:.3f} (R={np.exp(logR):.3f} Mpc).')

    def set_bsd_manually(self, logR):
        """
        Set bubble size distribution manually.

        Parameters
        ----------
        logR : float or np.ndarray
            Log of bubble sizes to be used.
        """
        self.draw_logR = lambda N=1: logR
        self.draw_R = lambda N=1: np.exp(self.draw_logR(N))
        print(f'Bubble size are set manually.')

    def set_source_positions_manually(self, source_pos):
        """
        Set source positions manually.

        Parameters
        ----------
        source_pos : np.ndarray
            Array of source positions.
        """
        n_grid = self.n_grid
        self.draw_source_pos = lambda N=1: source_pos
        print(f'Source positions on ({n_grid},{n_grid},{n_grid}) grid are set manually.')

    def paint_sphere(self, box=None, n_batch=10):
        """
        Paint spheres on the grid.

        Parameters
        ----------
        box : np.ndarray, optional
            Initial grid. If None, a new grid will be created.
        n_batch : int, optional
            Number of spheres to paint in each batch, default is 10.

        Returns
        -------
        np.ndarray
            Updated grid with spheres painted.
        """
        n_grid = self.n_grid
        if box is None: box = np.zeros((n_grid, n_grid, n_grid))
        Rmax_list = self.draw_R(n_batch)
        source_pos = self.draw_source_pos(n_batch)
        box_i = put_sphere_group(source_pos, Rmax_list, n_grid)
        box += box_i
        box = np.clip(box, 0, 1)
        return box

    def each_step(self, box, xhi, n_batch=10):
        n_grid = self.n_grid
        ## Brute force algorithm of putting sphere at random positions ##
        total_volume = n_grid**3
        progress_bar = tqdm(total=int((1-xhi) * total_volume), desc="Filling box", unit=f" voxels")

        while box.mean() < (1 - xhi):
            box = self.paint_sphere(box=box, n_batch=n_batch)
            progress_bar.update(np.count_nonzero(box))

        progress_bar.close()
        return box
    
class RandomSphereModel(SphereModel):
    '''
    Modelling reionization by randomly filling a box with spherical bubbles.
    '''
    def __init__(self, n_grid=256, box_len=300):
        super().__init__(n_grid, box_len)
        self.set_source_positions()

    def set_source_positions(self):
        n_grid = self.n_grid
        self.draw_source_pos = lambda N: np.random.randint(0, n_grid, (N, 3))
        print(f'Source positions on ({n_grid},{n_grid},{n_grid}) grid will be drawn randomly.')

    def run(self, n_batch=10):
        n_grid = self.n_grid
        zs, xHI = self.zs, self.xHI
        zs, xHI = zs[np.argsort(1-xHI)], xHI[np.argsort(1-xHI)]
        xfrac_cubes = {}
        box = np.zeros((n_grid, n_grid, n_grid))

        for ii, (zi, xhi) in enumerate(zip(zs, xHI)):
            print(f'{ii+1}/{len(zs)} | z={zi:.3f}, xhi={xhi:.3f}')
            box = self.each_step(box, xhi, n_batch=n_batch)
            print(f'Final box mean: {box.mean():.3f}')
            xfrac_cubes[zi] = box
        
        self.xfrac_cubes = xfrac_cubes
        return xfrac_cubes