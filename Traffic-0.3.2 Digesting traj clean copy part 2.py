import numpy as np
import os
import traffic_data_generator
from digest_util import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'
rand_seed=116
np.random.seed(rand_seed)


# In[]
# =========================== #
# Data generation coefficient #
# =========================== #
delta_gen = 0.05
n_samples = 100000
n_bins=10
bin_size=13


# =========================== #
#   parameter domain set up   #
# =========================== #
combinations = np.array([[64, .25], [40, .25], [24, .25], [16, .25], [8, .25],
                        [32, .5], [20, .5], [12, .5], [8, .5], [4, .5],
                        [16, 1], [10, 1], [6, 1], [4, 1], [2, 1],
                        [8, 2], [5, 2], [3, 2], [2, 2]])
n_new_combo=len(combinations)


# In[]
# =========================== #
#     Output directories      #
# =========================== #
ith_run = 12
example_name = 'traffic'
data_dir = traffic_data_generator.set_up(example_name)


def data_save_path(i_combo, Delta_t, len_hist_MZ, ith_run, var, prefix):
    path = data_dir + f'/digest/{prefix}{var}_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}.npy'
    return path


# In[]
# =========================== #
#     Change the delta        #
# =========================== #
for prefix in ("abs_weight_smoothed_", ""):
    for i_combo in range(len(combinations)):
        len_hist_MZ=int(combinations[i_combo, 0])
        Delta_t=combinations[i_combo, 1]
        delta_ratio=int(round(Delta_t/delta_gen))
        x=np.load(data_save_path(i_combo, Delta_t, len_hist_MZ, ith_run, 'x', prefix))
        y=np.load(data_save_path(i_combo, Delta_t, len_hist_MZ, ith_run, 'y', prefix))
        new_len_hist_MZ=int(len_hist_MZ/2)
        new_Delta_t=Delta_t*2
        new_delta_ratio=int(round(new_Delta_t/delta_gen))
        if new_len_hist_MZ in combinations[combinations[:, 1]==new_Delta_t, 0]:
            new_i_combo=np.intersect1d(np.where(combinations[:, 1]==new_Delta_t), np.where(combinations[:, 0]==new_len_hist_MZ))[0]
            if not os.path.exists(data_save_path(new_i_combo, new_Delta_t, new_len_hist_MZ, ith_run, 'x', prefix)):
                x_new=x[:, 1::2, :]
                y_new=y
                np.save(data_save_path(new_i_combo, new_Delta_t, new_len_hist_MZ, ith_run, 'x', prefix), x_new)
                np.save(data_save_path(new_i_combo, new_Delta_t, new_len_hist_MZ, ith_run, 'y', prefix), y_new)


# In[]
# =========================== #
# fitting the 5th polynomial  #
# =========================== #
for i_combo in range(len(combinations)):
    len_hist_MZ=int(round(combinations[i_combo, 0]))
    Delta_t=combinations[i_combo, 1]
    delta_ratio=int(round(Delta_t/delta_gen))
    x = np.load(f'{data_dir}/x_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}.npy')
    y = np.load(f'{data_dir}/y_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}.npy')
    n_samples = x.shape[0]
    n_bins = x.shape[2]  # 10

    # Data transformation by fitting the 5th polynomial
    deg = 5
    bin_size = 13
    bin_edge = -np.arange(n_bins)*bin_size
    x_new = np.zeros_like(x)
    y_new = np.zeros_like(y)
    for i_sample in range(n_samples):
        if i_sample % 5000==0:
            print(i_sample)
        for i_x in range(len_hist_MZ):
            x_new[i_sample, i_x, :] = poly_fit_eval(bin_edge, x[i_sample, i_x, :], deg)
        y_new[i_sample, :] = poly_fit_eval(bin_edge, y[i_sample, :], deg)

    # Store the data
    np.save(f'{data_dir}/poly_deg={deg}_x_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}.npy', x_new)
    np.save(f'{data_dir}/poly_deg={deg}_y_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}.npy', y_new)
