import numpy as np
import scipy as sp
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Concatenate, Add, Multiply, Average, Subtract
from tensorflow.keras import Model
import traffic_data_generator
import traffic_util
import random as rn


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
rand_seed = 116
rn.seed(rand_seed)
np.random.seed(rand_seed)
tf.random.set_seed(rand_seed)


# In[]
# =========================== #
# Data generation coefficient #
# =========================== #
delta_gen = 0.05
n_traj = 25000
n_samples = 100000


# =========================== #
#   parameter domain set up   #
# =========================== #
combinations = np.array([[64, .25], [40, .25], [24, .25], [16, .25], [8, .25],
                         [32, .5], [20, .5], [12, .5], [8, .5], [4, .5],
                         [16, 1], [10, 1], [6, 1], [4, 1], [2, 1],
                         [8, 2], [5, 2], [3, 2], [2, 2]])
n_new_combo = len(combinations)


# In[]
# ===========================
#     Output directories
# ===========================
ith_run = 12
example_name = 'traffic'
data_dir = traffic_data_generator.set_up(example_name)
experiment_name = f'_no_share_{n_samples}_Delta_{delta_gen}'
model_dir = example_name + '/' + experiment_name
plot_dir = model_dir + '/Plots'
checkpoint_dir = model_dir + '/check_points'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)


# In[]
# inits=['Orthogonal', 'glorot_normal','he_normal' ,'he_uniform', 'glorot_uniform']
# n_starts = 6
# n_starts = 4
# np.savetxt(data_dir + '/combinations' + str(len(combinations)) + '.txt', combinations)
# print('ith_run', ith_run)
inits = ['glorot_normal', 'he_normal', 'he_uniform']
rand_seed_shift = 0
n_epochs = 2000
validation_split = .2
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=1, mode='min',
                                              baseline=None, restore_best_weights=True)


# In[]
# # below is for i_combo=0 and ith_run=0
#   for i_start=1 we use the hard_sigmoid function as our second to last activation funciton (for intial guess)
#   for i_start=2 we multiply the input of hard_sigmoid by 10 to get it close to heavyiside (didn't work)
#   for i_start=3 we try to use L-1 norm
#   for i_start=4 we use NN data and revert to tanh and L-2 norm
#   for i_start=5 we use polyfit data (degree 5)
# # below is for ith run=1 i_combo=0
#   for i_start=1 we run with no lr deca
# # below is for ith_run=2 (uniform data generation) with 20 sec traj len
# # below is for ith_run=3 uniform data gen with updated interaction rules and 100 sec traj len
#   for i_start=1 we use abs val weight
# # for ith_run=5
#   i_start=0 is "normal"
#   I_start=1 we use sigmoid
#   I_start=2 we use polyfit
#   I_start=3 we use abs weighted

for i_combo in range(len(combinations)):
    for i_start in (0, 2, 3):
        opt_model_path = checkpoint_dir + f'/opt_model_first_try{i_start}_{i_combo}_{ith_run}.h5'
        if not os.path.exists(opt_model_path):
            print([i_combo, i_start, ith_run])
            i_hist_MZ = combinations[i_combo, 0]
            Delta_t = combinations[i_combo, 1]
            len_hist_MZ = int(i_hist_MZ)
            if i_start == 2:
                x = np.load(
                    f'{data_dir}/poly_deg=5_x_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}.npy')
                y_temp = np.load(
                    f'{data_dir}/poly_deg=5_y_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}.npy')
            elif i_start == 3:
                x = np.load(
                    f'{data_dir}/abs_weight_smoothed_x_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}.npy')
                y_temp = np.load(
                    f'{data_dir}/abs_weight_smoothed_y_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}.npy')
            else:
                x = np.load(f'{data_dir}/x_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}.npy')
                y_temp = np.load(f'{data_dir}/y_delta_t={Delta_t}MZ_hist={len_hist_MZ}ith run{ith_run}_{i_combo}.npy')
            exec(f'y_shift{i_combo}_{i_start}=traffic_util.min_max_scale()')
            exec(f'y=y_shift{i_combo}_{i_start}.fit_transform((y_temp-x[:,0,:]))')
            tf.random.set_seed(rand_seed_shift + i_start + i_combo * len(combinations))
            if x.shape[0] == y.shape[0]:
                n_data = x.shape[0]
            else:
                break
            x = x.reshape([n_data, -1])
            x_test = x[:int(validation_split * len(x))]
            x_train = x[int(validation_split * len(x)):]
            y_test = y[:int(validation_split * len(y))]
            y_train = y[int(validation_split * len(y)):]

            states = Input(shape=(x.shape[1],))
            x1 = states
            x2 = Dense(150, activation='relu', kernel_initializer=inits[i_start % (len(inits) - 1)])(x1)
            x3 = Dense(100, activation='relu', kernel_initializer=inits[i_start % (len(inits) - 1)])(x2)
            if i_start == 1:
                x4 = Dense(70, activation='hard_sigmoid', kernel_initializer=inits[i_start % (len(inits) - 1)])(x3 * 10)
            else:
                x4 = Dense(70, activation='tanh', kernel_initializer=inits[i_start % (len(inits) - 1)])(x3)
            x_output = Dense(y.shape[1], activation='linear')(x4)
            outputs = x_output
            model = Model(inputs=states, outputs=outputs)
            learning_rate = 0.001
            batch_size = 30
            adam = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.998, epsilon=10 ** -8, decay=0.0,
                                            amsgrad=False)
            model.compile(optimizer=adam, loss='mean_squared_error')
            savebest = tf.keras.callbacks.ModelCheckpoint(filepath=opt_model_path, monitor='val_loss',
                                                          verbose=i_combo == 0, save_best_only=True,
                                                          save_weights_only=False, mode='auto', save_freq='epoch')
            #     lr_decrease=bursting_util.ReduceLROnPlateau(
            #                         monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto',
            #                         min_delta=0, cooldown=0, min_lr=10**-12,filepath=opt_model_path
            #                         )
            lr_decrease = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50,
                                                               verbose=i_combo == 0, mode='auto', min_delta=0,
                                                               cooldown=0, min_lr=10 ** -15)
            hist = model.fit(x_train, y_train,
                             batch_size=batch_size,
                             epochs=n_epochs,
                             verbose=i_combo == 0,
                             callbacks=[savebest, lr_decrease, early_stop],
                             validation_data=(x_test, y_test),)
            val_loss_log = np.concatenate(
                (hist.history['val_loss'], 100 * np.ones(n_epochs - len(hist.history['val_loss']))))
            loss_log = np.concatenate((hist.history['loss'], 100 * np.ones(n_epochs - len(hist.history['loss']))))
            np.save(checkpoint_dir + f'/loss_opt_model_first_try{i_start}_{i_combo}_{ith_run}.npy', loss_log)
            np.save(checkpoint_dir + f'/val_loss_opt_model_first_try{i_start}_{i_combo}_{ith_run}.npy', val_loss_log)
