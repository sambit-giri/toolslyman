import numpy as np 
import toolslyman as tl

def test_age_estimator():
	tl.cosmology.set_cosmology(name='planck18')
	xx = tl.cosmology.cosmo.age(0).to('Gyr')
	assert np.abs(xx.value-13.78)<0.01

def test_comoving_distance():
	tl.cosmology.set_cosmology(name='planck18')
	xx = tl.cosmology.cosmo.comoving_distance(1).to('Mpc')
	assert np.abs(xx.value-3395.63)<0.01

def test_lookback_distance():
	tl.cosmology.set_cosmology(name='planck18')
	xx = tl.cosmology.cosmo.lookback_distance(1).to('Mpc')
	assert np.abs(xx.value-2433.048)<0.01