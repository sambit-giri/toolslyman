import numpy as np
import astropy.units as u
import astropy.constants as const
import toolslyman as tl
from toolslyman.cosmology_calc import cdist_to_z

LAMBDA_0 = 1215.67  # AA
Z_SOURCE = 8.76
Z_END = 6.0
X_H = 0.76


def _uniform_skewer(xhi_val, N, dr=0.5 * u.Mpc):
    return np.ones(N) * xhi_val, np.zeros(N), dr


def _n_cells(z_source=Z_SOURCE, z_end=Z_END, dr=0.5 * u.Mpc, cosmo=None):
    if cosmo is None:
        cosmo = tl.cosmology.cosmo
    r_source = cosmo.comoving_distance(z_source)
    r_end = cosmo.comoving_distance(z_end)
    return int((r_source - r_end).to('Mpc').value / dr.to('Mpc').value)


def test_tau_GP_scales_linearly_with_xHI_and_X_H():
    z = 7.0
    base = tl.tau_GP(z, 1.0, X_H=1.0)
    assert base > 0
    assert np.isclose(tl.tau_GP(z, 0.5, X_H=1.0), 0.5 * base)
    assert np.isclose(tl.tau_GP(z, 1.0, X_H=0.76), 0.76 * base)


def test_tau_igm_analytic_wing_decreases_away_from_resonance():
    lam_grid = LAMBDA_0 * (1 + Z_SOURCE) + np.array([50.0, 150.0, 300.0, 600.0])
    z_obs = lam_grid / LAMBDA_0 - 1
    tau = tl.tau_igm_analytic(z_obs, Z_SOURCE, 0.5, z_end=Z_END, X_H=X_H)
    assert np.all(tau > 0)
    assert np.all(np.diff(tau) < 0), "optical depth should fall monotonically with distance from resonance"


def test_column_density_matches_optical_depth_internal_column_density():
    N = 100
    xHI, dn, dr = _uniform_skewer(0.5, N)

    N_HI = tl.column_density_along_skewer(Z_SOURCE, xHI, dn, dr, X_H=X_H)

    cosmo = tl.cosmology.cosmo
    r_src = cosmo.comoving_distance(Z_SOURCE)
    r_arr = r_src - dr.to('cm') * np.arange(N)
    z_arr = cdist_to_z(r_arr, cosmo=cosmo)
    expected_total = (
        xHI * (1 + dn) * (1 + z_arr) ** 2
        * (X_H * cosmo.Ob0 * cosmo.critical_density0 / const.m_p)
        * dr.to('cm')
    ).to('1/cm2').sum()

    assert np.isclose(N_HI[0, -1].to('1/cm2').value, expected_total.value, rtol=1e-10)


def test_numerical_skewer_matches_analytic_damping_wing():
    """Regression test for the sqrt(pi) double-counting bug (issue #3): before the
    fix the numerical skewer overestimated tau by a constant factor of sqrt(pi) (~1.77x)
    relative to the closed-form Miralda-Escude (1998) / Huberty et al. (2025) formula."""
    N = _n_cells()
    xHI, dn, dr = _uniform_skewer(0.5, N)

    tau_num, lam_num = tl.optical_depth_lyA_along_skewer(
        Z_SOURCE, xHI, dn, dr, temp=1e4 * u.K, damped=True, verbose=False)
    lam_num = lam_num.to('AA').value
    idx = np.argsort(lam_num)
    lam_s, tau_s = lam_num[idx], tau_num[idx]

    for offset in [100.0, 300.0, 600.0]:
        lam_test = LAMBDA_0 * (1 + Z_SOURCE) + offset
        tau_numerical = np.interp(lam_test, lam_s, tau_s)
        tau_analytic = tl.tau_igm_analytic(
            lam_test / LAMBDA_0 - 1, Z_SOURCE, 0.5, z_end=Z_END, X_H=X_H)
        ratio = tau_numerical / float(tau_analytic)
        assert 0.85 < ratio < 1.10, (
            f"numerical/analytic ratio {ratio:.3f} at +{offset}AA is outside the "
            "expected few-percent agreement (thermal broadening in the numerical "
            "calculation is the only expected residual)"
        )


def test_positive_peculiar_velocity_redshifts_the_wing():
    """A positive (receding) line-of-sight peculiar velocity should shift absorption
    features redward (to longer wavelength), matching the v_obs = H(z) r + v_pec
    convention."""
    N = 50
    xHI, dn, dr = _uniform_skewer(1.0, N)

    tau0, lam0 = tl.optical_depth_lyA_along_skewer(
        Z_SOURCE, xHI, dn, dr, temp=1e4 * u.K, damped=True, verbose=False)
    vpec = np.ones(N) * 100 * u.km / u.s
    tau1, _ = tl.optical_depth_lyA_along_skewer(
        Z_SOURCE, xHI, dn, dr, temp=1e4 * u.K, vpec=vpec, damped=True, verbose=False)

    lam = lam0.to('AA').value
    idx = np.argsort(lam)
    lam_s, tau0_s, tau1_s = lam[idx], tau0[idx], tau1[idx]

    def half_transmission_wavelength(tau_arr):
        trans = np.exp(-tau_arr)
        mid = len(lam_s) // 2
        return np.interp(0.5, trans[mid:], lam_s[mid:])

    lam_no_vpec = half_transmission_wavelength(tau0_s)
    lam_with_vpec = half_transmission_wavelength(tau1_s)
    shift = lam_with_vpec - lam_no_vpec
    expected_shift = LAMBDA_0 * (1 + Z_SOURCE) * 100 / 2.9979e5

    assert shift > 0, "positive (receding) peculiar velocity should redshift the wing"
    assert np.isclose(shift, expected_shift, rtol=0.05)


def test_multi_skewer_matches_single_skewer_loop():
    """Regression test for a bug where the 2D recursive branch didn't forward all
    keyword arguments to the per-skewer 1D calls."""
    N = 40
    dr = 0.5 * u.Mpc
    xHI_2d = np.vstack([np.ones(N) * 0.3, np.ones(N) * 0.8])
    dn_2d = np.zeros((2, N))

    tau_2d, lam_2d = tl.optical_depth_lyA_along_skewer(
        Z_SOURCE, xHI_2d, dn_2d, dr, temp=1e4 * u.K, damped=True, verbose=False)
    assert tau_2d.shape == (2, len(lam_2d))

    for j in range(2):
        tau_1d, _ = tl.optical_depth_lyA_along_skewer(
            Z_SOURCE, xHI_2d[j], dn_2d[j], dr, temp=1e4 * u.K, damped=True, verbose=False)
        assert np.allclose(tau_1d, tau_2d[j])
