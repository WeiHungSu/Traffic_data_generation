import numpy as np
from scipy.integrate import odeint
import os


def set_up(example_name):
    cwd = os.getcwd()
    example_dir = cwd + f'/{example_name}'
    if not os.path.exists(example_dir):
        os.mkdir(example_dir)
    data_dir=example_dir+'/Data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)    
    
    return data_dir


def rhs_f_linear(x, t):
    v1, v2, v3, v4 = -1.2, 18, 12, 17.4
    gca, gk, gl, gkca = 4., 8.0, 2.0, .25
    vk, vl, vca = -84, -60, 120.0
    phi, eps, mu = .23, .005, .02
    I, k = 45, 1
#     minf=(.5*(1.+np.tanh((x[0]-v1)/v2)))
#     winf=(.5*(1.+np.tanh((x[0]-v3)/v4)))
#     tauw=1./np.cosh((x[0]-v3)/(2*v4))
#     z=x[1]/(1.0+x[1])
    y=np.zeros(x.shape)
    #Example2
    y[0]=-(gca*(.5*(1.+np.tanh((x[0]-v1)/v2)))*(x[0]-vca)+gk*x[2]*(x[0]-vk)+gl*(x[0]-vl)+gkca*(x[1]/(1.0+x[1]))*(x[0]-vk))+I
#     y[1]= 0
    y[1]= 20*eps*(-mu*gca*(.5*(1.+np.tanh((x[0]-v1)/v2)))*(x[0]-vca)-k*x[1])
    y[2]= 20*phi*((.5*(1.+np.tanh((x[0]-v3)/v4)))-x[2])*np.cosh((x[0]-v3)/(2*v4))
    return y


def generate_u(f, x, delta_train, t0=0):
    y=np.zeros(x.shape)
    n=x.shape[0]
    
    for i in np.arange(n):
        t_grid=np.linspace(t0, delta_train[i], 20)
        traj=odeint(f,  x[i, :], t_grid)
        y[i,:]=traj[-1]
    return y


def generate_t(f, x, delta_train, t0=0):
    
    y=np.zeros(x.shape)
    n=x.shape[0]
    
    for i in np.arange(n-1):
        t_grid=np.linspace(t0, delta_train[i], 20)
        traj=odeint(f,  x[i, :], t_grid)
        y[i,:]=traj[-1]
        x[i+1,:]=y[i,:]
    return x, y


def generate_data_rand(Delta, n_data, n_var, domain_u, example_name):
    
    x_train=np.zeros([n_data, n_var])
    y_train=np.zeros([n_data, n_var])
    
    for i_var in np.arange(n_var):
        x_train[:, i_var]=np.random.uniform(domain_u[i_var, 0], domain_u[i_var, 1],n_data)
     
    delta_train=Delta*np.ones(n_data)                                              

    y_train=generate_u(rhs_f_linear, x_train, delta_train)

    return x_train, y_train    


def generate_data_traj(Delta, n_data, n_var, domain_u, example_name,length):
    
    x_train=np.zeros([n_data, n_var])
    y_train=np.zeros([n_data, n_var])
    
    n_iter=n_data//length
    
    for i_var in np.arange(n_var):
        x_train[0:n_data:length, i_var]=np.random.uniform(domain_u[i_var, 0], domain_u[i_var, 1],n_iter)
     
    delta_train=Delta*np.ones(n_data)
    
#     x_train[n_data//2,:]=[-30,1,0]
#     y_train[0:n_data//2,:]=generate_u(rhs_f_linear, x_train[n_data//2:n_data,:], delta_train)
#     x_train[n_data//2:n_data,:],y_train[n_data//2:n_data,:]=generate_t(rhs_f_linear, x_train[n_data//2:n_data,:], delta_train)
                                                                      
                                                                       
    for i in range(n_iter):
        start=i*length
        end=start+length
        x_train[start:end,:],y_train[start:end,:]=generate_t(rhs_f_linear, x_train[start:end,:], delta_train)
            
    return x_train, y_train  
