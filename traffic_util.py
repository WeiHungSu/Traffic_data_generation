import numpy as np
import scipy as sp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


def reinitialize_weights(model):
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)


def legendre_approx(n_terms, x):
    coeff = np.arange(n_terms+1)
    coeff = (-1)**coeff*(4.*coeff+3.)*sp.special.factorial(2.*coeff, exact=True)\
    /(2.**(2.*coeff+1)*sp.special.factorial(coeff+1, exact=True)*sp.special.factorial(coeff, exact=True))

    c=np.zeros(2*n_terms+2)
    index=np.arange(len(c))
    c[np.mod(index, 2)==1]=coeff

    return np.polynomial.legendre.legval(x, c), c


def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true), axis=-1)


def traj_MSE(model,y_shift,Delta,len_hist_MZ,x_init,data_dir,t):
    if t+len_hist_MZ*Delta>160:
        print('needs shorter t')
        return
    delta_gen=.02
    delta_int=int(Delta/delta_gen)
    y_ref=np.loadtxt(data_dir+'/y_ref_c='+str(x_init)+'.txt') 
    y_predict=np.zeros(int(t/Delta)+len_hist_MZ)
    y_predict[0:len_hist_MZ]=y_ref[0:len_hist_MZ*delta_int:delta_int,2]
    for i in range(int(t/Delta)):
        y_predict[i+len_hist_MZ]=y_predict[i+len_hist_MZ-1]+y_shift.inverse(model.predict(np.array([y_predict[i:i+len_hist_MZ]])))
    MSE=mean_squared_error(y_ref[len_hist_MZ*delta_int:len_hist_MZ*delta_int+int(t/Delta)*delta_int:delta_int,2],
                                         y_predict[len_hist_MZ:len_hist_MZ+int(t/Delta)])
    return MSE


def Avg_traj_MSE(model_path,y_shift,Delta,x_init,data_dir,t):
    model=tf.keras.models.load_model(model_path)
    len_hist_MZ=model.get_weights()[0].shape[0]
    MSE_mat=np.zeros(x_init.shape[0])
    for i_init in range(x_init.shape[0]):
        MSE_mat[i_init]=traj_MSE(model,y_shift,Delta,len_hist_MZ,x_init[i_init],data_dir,t)
    return np.mean(MSE_mat)


def error_plot(hist,a):
    plt.figure(figsize=(14,5))
    x=range(len(hist.history['val_loss']))
    plt.plot(x,np.log(hist.history['val_loss']), label='val_loss')
    plt.plot(x,np.log(hist.history['loss']), label='loss')
    plt.plot(x,np.min(np.log(hist.history['val_loss']))*np.ones(len(x)), label='min_val_loss')
    plt.plot(np.argmin(hist.history['val_loss']),np.min(np.log(hist.history['val_loss'])),'rx', label='min val_loss point')
    plt.ylim(np.min(np.log(hist.history['val_loss']))-a,np.min(np.log(hist.history['val_loss']))+a)
    plt.legend()
    

class min_max_scale:
    def __inint__(self):
        self.dimension=0
        self.min_max=[]
        
    def fit_transform(self, x):
        output=x
        self.dimension=x.shape[1]
        self.min_max=np.zeros(self.dimension)
        for i in range(self.dimension):
            self.min_max[i]=np.max(x[:,i])-np.min(x[:,i])
            if self.min_max[i]!=0:
                output[:,i]=x[:,i]/self.min_max[i]
        return output
    
    def transform(self,x):
        if self.dimension != x.shape[1]:
            print('dimension does not align of fit_transform must be run first')
            return
        output=x
        for i in range(self.dimension):
            if self.min_max[i]!=0:
                output[:,i]=x[:,i]/self.min_max[i]
        return output
    
    def inverse(self,y):
        if self.dimension != y.shape[1]:
            print('dimension does not align of fit_transform must be run first')
            return
        output=y
        for i in range(self.dimension):
            if self.min_max[i]!=0:
                output[:,i]=y[:,i]*self.min_max[i]
        return output


class ReduceLROnPlateau(tf.keras.callbacks.Callback):
    def __init__(self,
                   monitor='val_loss',
                   factor=0.1,
                   patience=10,
                   verbose=0,
                   mode='auto',
                   min_delta=1e-4,
                   cooldown=0,
                   min_lr=0,
                   filepath=None,
                   **kwargs):
        super(ReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.filepath = filepath
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            logging.warning('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
            (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Reduce LR on plateau conditioned on metric `%s` '
                          'which is not available. Available metrics are: %s',
                          self.monitor, ','.join(list(logs.keys())))

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.model=tf.keras.models.load_model(self.filepath)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                  'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


class MinLossModel_batch(tf.keras.callbacks.Callback):
    def __init__(self, filepath,
                 val_input, val_output,
                 last_best,validation_batch_size=32, monitor='val_loss', 
                 verbose=0,
                 save_best_only=True, 
                 save_weights_only=False,
                 mode='auto', 
                 period=1):
        #This is a subclass of the Callback class
        super(MinLossModel_batch, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.val_input=val_input
        self.val_output=val_output
        self.best=last_best
        self.monitor_op=np.less
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.period = period
        self.batches_since_last_save = 0
        self.validation_batch_size=validation_batch_size

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batches_since_last_save += 1
        if self.batches_since_last_save >= self.period:
            self.batches_since_last_save = 0
            filepath = self.filepath.format(batch + 1, **logs)
            current=self.model.evaluate(self.val_input, self.val_output, verbose=0,batch_size=self.validation_batch_size)

            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if self.monitor_op(current, self.best):                      
                    self.best = current
                    self.model.save(filepath, overwrite=True)
    
