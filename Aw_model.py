import numpy as np


class Aw():
    def __init__(self, C, gamma, A, T_r, car_length, V, T1, T2, delta_t):
        self.C = C
        self.gamma = gamma
        self.T_r = T_r
        self.car_length = car_length
        self.V = V
        self.T1 = T1
        self.T2 = T2
        self.delta_t = delta_t
        self.A = A

    def acc_func(self, v, x):
        T1_n_steps = int(round(self.T1 / self.delta_t))
        T2_n_steps = int(round(self.T2 / self.delta_t))
        i_T1, i_T2 = None, None
        if len(v.shape) == 2: i_T1 = np.max([v.shape[1] - T1_n_steps - 1, 0])
        if len(x.shape) == 2: i_T2 = np.max([x.shape[1] - T2_n_steps - 1, 0])
        if len(v.shape) == 2:
            v_current = v[1, 0]
        else:
            v_current = v[1]
        delta_v = v[0, i_T1] - v[1, i_T1]
        delta_x = x[0, i_T2] - x[1, i_T2]
        true_acc = self.C * delta_v / delta_x ** (self.gamma + 1) + self.A / self.T_r * (
                    self.V(self.car_length / delta_x) - v_current)
        #         return np.min([self.max_acc,true_acc])
        return true_acc

    def first_car_acc(self, v):
        return self.A / self.T_r * (self.V(0) - v)