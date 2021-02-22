import numpy as np


def init_loc(separation_mean, separation_var, loc_init, n_cars):
    if loc_init == 'equal_space':
        init_loc_matrix = -np.arange(n_cars) * separation_mean
    elif loc_init == 'uniform_separation':
        separation_vector = np.random.uniform(separation_mean - separation_var, separation_mean + separation_var,
                                              n_cars - 1)
        init_loc_matrix[0] = 0
        for i_car in range(n_cars - 1):
            init_loc_matrix[i_car + 1] = init_loc_matrix[i_car] - separation_vector[i_car]
    elif loc_init == 'uniform':
        min_separation = separation_var
        init_loc_matrix = np.zeros(1)
        while len(init_loc_matrix) < n_cars:
            a = np.random.uniform(-n_cars * separation_mean, 0, n_cars - len(init_loc_matrix))
            init_loc_matrix = np.sort(np.concatenate((init_loc_matrix, a)))[::-1]
            i_car = 0
            while i_car < init_loc_matrix.shape[0] - 1:
                if init_loc_matrix[i_car] - init_loc_matrix[i_car + 1] < min_separation:
                    init_loc_matrix = np.delete(init_loc_matrix, i_car + 1)
                #                 init_loc_matrix[i_car+1]=init_loc_matrix[i_car]-min_separation
                else:
                    i_car += 1
    return init_loc_matrix, n_cars


def init_vel(lead_car_speed, vel_init, vel_init_mean, vel_init_var, n_cars):
    init_vel_matrix = np.zeros(n_cars)
    if vel_init == 'uniform':
        init_vel_matrix = np.random.uniform(vel_init_mean - vel_init_var, vel_init_mean + vel_init_var, n_cars)
    elif vel_init == 'constant':
        init_vel_matrix = np.ones(n_cars) * vel_init_mean
    elif vel_init == 'normal':
        init_vel_matrix = np.max([np.zeros(n_cars), np.random.normal(vel_init_mean, vel_init_var, n_cars)], axis=0)
    elif vel_init == 'worst_case_uniform':
        for i in range(1, n_cars):
            init_vel_matrix[i] = vel_init_mean - vel_init_var * (-1) ** i
    init_vel_matrix[0] = lead_car_speed
    return init_vel_matrix


def move_cars_delta_t(loc_vec, vel_vec, acc_vec, acc_function, delta_t, lead_acc_function):
    new_vel_vec = np.zeros(vel_vec.shape[0])
    new_loc_vec = np.zeros(loc_vec.shape[0])
    if np.all(acc_vec[1:] == 0):
        new_acc_vec = np.zeros(acc_vec.shape[0])
        new_acc_vec[0] = lead_acc_function(vel_vec[0])
        for i_car in range(1, loc_vec.shape[0]):
            new_acc_vec[i_car] = acc_function(vel_vec[i_car - 1:i_car + 1],
                                              loc_vec[i_car - 1:i_car + 1])
    else:
        new_acc_vec = acc_vec
    new_vel_vec = vel_vec + new_acc_vec * delta_t
    new_loc_vec = loc_vec + vel_vec * delta_t + new_acc_vec * delta_t ** 2 / 2

    return new_loc_vec, new_vel_vec, new_acc_vec


def calc_collision_time(loc_vec, vel_vec, acc_vec, acc_function, delta_ideal, car_length, lead_acc_function):
    collision_time_fraction = 3
    delta_loc = loc_vec[1:] - loc_vec[:-1] + car_length
    if np.any(delta_loc >= 0):
        print("cars closer than car length")
        return "error", "too short car length"
    delta_vel = vel_vec[1:] - vel_vec[:-1]
    acc_vec[0] = lead_acc_function(vel_vec[0])
    for i_car in range(1, loc_vec.shape[0]):
        acc_vec[i_car] = acc_function(vel_vec[i_car - 1:i_car + 1],
                                      loc_vec[i_car - 1:i_car + 1])
    delta_acc = acc_vec[1:] - acc_vec[:-1]
    #     print(delta_acc)
    if np.all(delta_acc != 0):
        linear_min = delta_ideal * collision_time_fraction + 1
    elif np.all(delta_vel[delta_acc == 0] <= 0):
        linear_min = delta_gen * collision_time_fraction + 1
    else:
        linear_min = np.min(-delta_loc[delta_acc == 0 and delta_vel > 0] / delta_vel[delta_acc == 0 and delta_vel > 0])
    real_roots = (delta_vel ** 2 - 2 * delta_acc * delta_loc) >= 0
    acc_non_zero = (delta_acc != 0)
    dc = np.all([real_roots, acc_non_zero], axis=0)  # dc means don't calculate if this is false, we don't calculate it
    first_real_quad_root = (-delta_vel[dc] + np.sqrt(delta_vel[dc] ** 2 - 2 * delta_acc[dc] * delta_loc[dc])) / \
                           delta_acc[dc]
    second_real_quad_root = (-delta_vel[dc] - np.sqrt(delta_vel[dc] ** 2 - 2 * delta_acc[dc] * delta_loc[dc])) / \
                            delta_acc[dc]
    if np.all(first_real_quad_root <= 0):
        first_min = delta_ideal * collision_time_fraction + 1
    else:
        first_min = np.min(first_real_quad_root[first_real_quad_root > 0])
    if np.all(second_real_quad_root <= 0):
        second_min = delta_ideal * collision_time_fraction + 1
    else:
        second_min = np.min(second_real_quad_root[second_real_quad_root > 0])
    collision_time = np.min([first_min, second_min, linear_min]) / collision_time_fraction
    #     print("ct" + str(collision_time))

    return acc_vec, collision_time


def move_cars_delta_max(loc_vec, vel_vec, acc_vec, acc_function, delta_ideal, car_length, lead_acc_function):
    delta_max = np.inf
    first_time = True
    while delta_ideal > 0:
        acc_vec, collision_time = calc_collision_time(loc_vec, vel_vec, acc_vec, acc_function, delta_ideal, car_length,
                                                      lead_acc_function)
        time_step = np.min([delta_ideal, collision_time, delta_max])
        if collision_time < 10 ** -10:
            return loc_vec_log, vel_vec_log, "stop", time_log
        else:
            loc_vec, vel_vec, acc_vec = move_cars_delta_t(loc_vec, vel_vec, acc_vec, acc_function, time_step,
                                                          lead_acc_function)
        #             print("dm" + str(delta_max))
        if first_time:
            loc_vec_log = loc_vec.reshape([-1, 1])
            vel_vec_log = vel_vec.reshape([-1, 1])
            acc_vec_log = acc_vec.reshape([-1, 1])
            time_log = np.ones(1) * np.min([delta_ideal, collision_time, delta_max])
            first_time = False
        #             print("a loc_vec" +str(loc_vec.reshape([-1,1]).shape))
        else:
            #             print("b loc_vec" +str(loc_vec.reshape([-1,1]).shape))
            loc_vec_log = np.concatenate((loc_vec_log, loc_vec.reshape([-1, 1])), axis=1)
            vel_vec_log = np.concatenate((vel_vec_log, vel_vec.reshape([-1, 1])), axis=1)
            acc_vec_log = np.concatenate((acc_vec_log, acc_vec.reshape([-1, 1])), axis=1)
            time_log = np.concatenate((time_log, np.ones(1) * (time_step + time_log[-1])), axis=0)
        delta_max = 2 * time_step
        delta_ideal -= time_step
    #         print(loc_vec_log.shape)
    return loc_vec_log, vel_vec_log, acc_vec_log, time_log


def simulation_1(init_loc_matrix, init_vel_matrix, delta_gen, traj_len, acc_function, car_length, rtn,
                 lead_acc_function):
    n_cars = init_loc_matrix.shape[0]
    n_t_steps_traj = int(round(traj_len / delta_gen))

    loc_matrix = np.zeros([n_cars, n_t_steps_traj])
    loc_matrix[:, 0] = init_loc_matrix
    vel_matrix = np.zeros([n_cars, n_t_steps_traj])
    vel_matrix[:, 0] = init_vel_matrix
    acc_matrix = np.zeros([n_cars, n_t_steps_traj])

    time = np.arange(n_t_steps_traj) * delta_gen

    loc_matrix[0, :] = init_vel_matrix[0] * time
    vel_matrix[0, :] = init_vel_matrix[0] + 0 * time
    acc_matrix[0, :] = 0 * time

    if rtn == "last":
        for i in range(n_t_steps_traj - 1):
            loc_vec_log, vel_vec_log, acc_vec_log, time_log = move_cars_delta_max(loc_matrix[:, i], vel_matrix[:, i],
                                                                                  acc_matrix[:, i], acc_function,
                                                                                  delta_gen, car_length,
                                                                                  lead_acc_function)
            if acc_vec_log == "stop":
                return loc_matrix, vel_matrix, "stop", time
            loc_matrix[:, i + 1], vel_matrix[:, i + 1], acc_matrix[:, i] = loc_vec_log[:, -1], vel_vec_log[:,
                                                                                               -1], acc_vec_log[:, -1]

    elif rtn == "all":
        loc_matrix = loc_matrix[:, 0].reshape([-1, 1])
        vel_matrix = vel_matrix[:, 0].reshape([-1, 1])
        acc_matrix = acc_matrix[:, 0].reshape([-1, 1])
        time = np.zeros(1)
        for i in range(n_t_steps_traj - 1):
            loc_vec_log, vel_vec_log, acc_vec_log, time_log = move_cars_delta_max(loc_matrix[:, -1], vel_matrix[:, -1],
                                                                                  acc_matrix[:, -1], acc_function,
                                                                                  delta_gen, car_length,
                                                                                  lead_acc_function)
            if acc_vec_log == "stop":
                return loc_matrix, vel_matrix, "stop", time
            loc_matrix = np.concatenate((loc_matrix, loc_vec_log), axis=1)
            vel_matrix = np.concatenate((vel_matrix, vel_vec_log), axis=1)
            acc_matrix = np.concatenate((acc_matrix, acc_vec_log), axis=1)
            time = np.concatenate((time, time_log + time[-1]), axis=0)
    #             print("loc_matrix"+str(loc_matrix.shape))
    #             print(time.shape)

    for i in range(1, n_cars):
        loc_matrix[i, :] -= loc_matrix[0, :]
    loc_matrix[0, :] -= loc_matrix[0, :]

    return loc_matrix, vel_matrix, acc_matrix, time