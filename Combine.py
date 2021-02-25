import numpy as np
import traffic_data_generator

# In[]
# =========================== #
# Data generation coefficient #
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


# In[] Combine the data
for ind_f_name in ['abs_weight_smoothed_', '']:
    for i_combo in range(5):
        len_hist_MZ = int(combinations[i_combo, 0])
        Delta_t = combinations[i_combo, 1]
        for run_index in range(10):
            x_temp = np.load(data_dir + f'/digest/{ind_f_name}x_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}_run_index_{run_index}.npy')
            y_temp = np.load(data_dir + f'/digest/{ind_f_name}y_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}_run_index_{run_index}.npy')
            if run_index == 0:
                x = x_temp
                y = y_temp
            else:
                x = np.concatenate((x, x_temp), axis=0)
                y = np.concatenate((y, y_temp), axis=0)
        np.save(data_dir + f'/digest/{ind_f_name}x_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}.npy', x)
        np.save(data_dir + f'/digest/{ind_f_name}y_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}.npy', y)
