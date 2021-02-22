import os
import numpy as np
import traffic_data_generator
from traffic2_util import *
from Aw_model import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'
rand_seed=116

# In[]
# =========================== #
# Data generation coefficient #
# =========================== #
delta_gen = 0.05
n_traj = 25000
n_samples = 100000  # for the training data
n_traj_per_run = 50
n_runs = int(np.ceil(n_traj / n_traj_per_run))


# In[]
# =========================== #
#   parameter domain set up   #
# =========================== #
domain=np.array([[400, 400],  # n_cars_intended
                 [20, 20],  # T_r
                 [2, 2],  # C
                 [1, 1],  # rho_max
                 [1, 1],  # v_max
                 [.35, .35],  # vel_init_mean
                 [1., 1.],  # vel_variance as fraction of mean if normal, upper lower bound if uniform
                 [.4, .7],  # separation_mean
                 [1/30, 1/30],  # separation_variance  if normal, upper lower bound if
                 # uniform_separation,min_separation if uniform
                 [20., 20],  # traj_length
                 [1, 1]  # zero here indicates we add cars until we get to n_cars while throwing out too close cars,
                 # 1 is for first car at inf and add cars until reach n_cars
                 ])

n_cars = int(np.random.uniform(domain[0, 0], domain[0, 1]))  # int(domain[0,this_combo[0]])
traj_len = int(np.random.uniform(domain[9, 0], domain[9, 1]))  # total time


# In[]
# =========================== #
#     Output directories      #
# =========================== #
ith_run = 12
example_name="traffic"
data_dir = traffic_data_generator.set_up(example_name)
traj_dir = data_dir + '/parameters AW' + str(ith_run) + 'd_gen=' + str(delta_gen) + 'n_samples=' + str(n_samples)
if not os.path.exists(traj_dir):
    os.mkdir(traj_dir)


# In[]
# =========================== #
#     Generate the data       #
# =========================== #
for i_run in range(n_runs):
    print(f'{i_run+1}/{n_runs}')
    location = np.full((n_traj_per_run, n_cars, int(round(traj_len / delta_gen))), -np.inf)
    velocity = np.full((n_traj_per_run, n_cars, int(round(traj_len / delta_gen))), -np.inf)

    for i_traj in range(n_traj_per_run):
        np.random.seed(rand_seed + i_traj + n_traj_per_run * i_run)
        acc_matrix = "stop"

        while acc_matrix == "stop":
            C = np.random.uniform(domain[2, 0], domain[2, 1])
            gamma = 0
            A = 1
            T_r = np.random.uniform(domain[1, 0], domain[1, 1])
            car_length = 1 / 40
            v_max = np.random.uniform(domain[4, 0], domain[4, 1])
            rho_max = np.random.uniform(domain[3, 0], domain[3, 1])

            def U(rho):
                return v_max * (np.pi / 2. + np.arctan(11 * (rho - .22) / (rho - 1))) / (
                            np.pi / 2 + np.arctan(11 * .22)) * (rho < 1)


            def V(rho):
                return U(rho / rho_max)


            T1 = T2 = 0
            acc_function = Aw(C, gamma, A, T_r, car_length, V, T1, T2, delta_gen).acc_func
            lead_acc_function = Aw(C, gamma, A, T_r, car_length, V, T1, T2, delta_gen).first_car_acc
            separation_mean = np.random.uniform(domain[7, 0], domain[7, 1])
            separation_var = np.random.uniform(domain[8, 0], domain[8, 1])

            # initial set up
            loc_init = 'uniform'
            init_loc_matrix, n_cars = init_loc(separation_mean, separation_var, loc_init, n_cars)
            vel_init = 'uniform'
            vel_init_mean = np.random.uniform(domain[5, 0], domain[5, 1])
            vel_init_var = np.random.uniform(domain[6, 0], domain[6, 1]) * vel_init_mean
            lead_car_speed = np.random.uniform(vel_init_mean - vel_init_var, vel_init_mean + vel_init_var)
            init_vel_matrix = init_vel(lead_car_speed, vel_init, vel_init_mean, vel_init_var, n_cars)

            # Generate the data
            location[i_traj, :n_cars], velocity[i_traj, :n_cars], acc_matrix, time_log = simulation_1(  # ??? why n_car inside of the parenthesis
                init_loc_matrix, init_vel_matrix, delta_gen, traj_len, acc_function, car_length, "last",
                lead_acc_function)
            if acc_matrix == "stop":print(stop)  # ????
            n_cars = int(np.random.uniform(domain[0, 0], domain[0, 1]))  # ???

    # np.save(traj_dir + '/velocity' + str(i_run * n_traj_per_run) + ".npy", velocity)
    np.save(traj_dir + '/location' + str(i_run * n_traj_per_run) + ".npy", location)
    print(f'Done_{i_run}')

np.save(data_dir+f'/parameters AW{ith_run}d_gen={delta_gen}n_samples={n_samples}.npy', domain)
