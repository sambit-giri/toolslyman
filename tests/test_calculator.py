import numpy as np 
import toolslyman

def test_age_estimator():
	param = toolslyman.param()
	t0 = toolslyman.age_estimator(param, 0)
	assert np.abs(t0-13.74)<0.01