import numpy as np

tau_bins = np.geomspace(1e-4, 1e4, num=50)
taus = 0.5 * (tau_bins[1:] + tau_bins[:-1])
