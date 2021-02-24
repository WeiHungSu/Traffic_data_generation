import numpy as np
import time
import os
import traffic_data_generator
import random as rn
from digest_util import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rand_seed = 116


# In[]
# =========================== #
# Data generation coefficient #
# =========================== #

delta_gen = 0.05
n_traj = 25000
n_samples = 100000
n_bins = 20
bin_size = 25
n_traj_per_run = 50
sample_per_traj=int(round(n_samples/n_traj))

# =========================== #
#   parameter domain set up   #
# =========================== #
combinations = np.array(
    [[64, .25], [40, .25], [24, .25], [16, .25], [8, .25],
     [32, .5], [20, .5], [12, .5], [8, .5], [4, .5],
     [16, 1], [10, 1], [6, 1], [4, 1], [2, 1],
     [8, 2], [5, 2], [3, 2], [2, 2]])


# In[]
# =========================== #
#     Output directories      #
# =========================== #
ith_run = 12
example_name = "traffic"
data_dir = traffic_data_generator.set_up(example_name)
experiment_name = f'_no_share_{n_samples}_Delta_{delta_gen}'
traj_dir = data_dir + f'/parameters AW{ith_run}d_gen={delta_gen}n_samples={n_samples}'
domain = np.load(data_dir + f'/parameters AW{ith_run}d_gen={delta_gen}n_samples={n_samples}.npy')
traj_len = int(np.random.uniform(domain[9, 0], domain[9, 1]))  # total time
n_runs = int(np.ceil(n_traj / n_traj_per_run))


# In[]
# =========================== #
#     Generate the data       #
# =========================== #
# standard trajectory build and abs_weighted_indicator_func build
run_index = 5  # 0, 5
run_index_list = np.array([run_index*50, (run_index+1)*50])
y_index_mat = np.load(data_dir + '/y_index for sampling.npy')
for ind_f in (abs_weighted_indicator_func, indicator_func):
    for i_combo in range(5):
        len_hist_MZ = int(combinations[i_combo, 0])
        Delta_t = combinations[i_combo, 1]
        delta_ratio = int(round(Delta_t / delta_gen))
        for i_run in range(run_index_list[0], run_index_list[1]):  # range(n_runs):
            print(f'{i_run}/{run_index_list[1]-run_index_list[0]}')
            np.random.seed(rand_seed + i_run)
            location = np.load(traj_dir + f'/location{i_run * n_traj_per_run}.npy')

            v = np.zeros_like(location[0, :, 0])
            for i_traj in range(n_traj_per_run):
                for i_sample in range(sample_per_traj):
                    y_index = int(y_index_mat[i_run, i_traj, int(round(Delta_t * len_hist_MZ)), i_sample])
                    y_build = ind_f(location[i_traj, :, y_index], v, bin_size, n_bins)[1, :-1]
                    y_build = y_build.reshape([1, -1])
                    x_build = ind_f(location[i_traj, :, y_index - delta_ratio], v, bin_size, n_bins)[1, :-1]
                    x_build = x_build.reshape([1, 1, -1])
                    for i_x in range(1, len_hist_MZ):
                        x_temp = ind_f(location[i_traj, :, y_index - delta_ratio * i_x], v, bin_size, n_bins)[1, :-1]
                        x_temp = x_temp.reshape([1, 1, -1])
                        x_build = np.concatenate((x_build, x_temp), axis=1)
                    if i_traj == 0 and i_run == run_index_list[0] and i_sample == 0:
                        x = x_build
                        y = y_build
                    else:
                        x = np.concatenate((x, x_build), axis=0)
                        y = np.concatenate((y, y_build), axis=0)
        print(i_combo)
        if ind_f == abs_weighted_indicator_func:
            np.save(data_dir + f'/digest/abs_weight_smoothed_x_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}_run_index_{run_index}.npy', x)
            np.save(data_dir + f'/digest/abs_weight_smoothed_y_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}_run_index_{run_index}.npy', y)

        else:
            np.save(data_dir + f'/digest/x_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}_run_index_{run_index}.npy', x)
            np.save(data_dir + f'/digest/y_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}_run_index_{run_index}.npy', y)

