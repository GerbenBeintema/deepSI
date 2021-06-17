import torch
from torch import nn, optim


class feed_forward_nn(nn.Module): #for encoding
    def __init__(self,n_in=6,n_out=5,n_nodes_per_layer=64,n_hidden_layers=2,activation=nn.Tanh):
        super(feed_forward_nn,self).__init__()
        seq = [nn.Linear(n_in,n_nodes_per_layer),activation()]
        assert n_hidden_layers>0
        for i in range(n_hidden_layers-1):
            seq.append(nn.Linear(n_nodes_per_layer,n_nodes_per_layer))
            seq.append(activation())
        seq.append(nn.Linear(n_nodes_per_layer,n_out))
        self.net = nn.Sequential(*seq)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, val=0) #bias
    def forward(self,X):
        return self.net(X)        


class simple_res_net(nn.Module):
    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        #linear + non-linear part 
        super(simple_res_net,self).__init__()
        self.net_lin = nn.Linear(n_in,n_out)
        if n_hidden_layers>0:
            self.net_non_lin = feed_forward_nn(n_in,n_out,n_nodes_per_layer=n_nodes_per_layer,n_hidden_layers=n_hidden_layers,activation=activation)
        else:
            self.net_non_lin = None

    def forward(self,x):
        if self.net_non_lin is not None:
            return self.net_lin(x) + self.net_non_lin(x)
        else: #linear
            return self.net_lin(x)


class affine_input_net(nn.Module):
    # implementation of: y = g(z)*u
    def __init__(self, output_dim=7, input_dim=2, affine_dim=3, g_net=simple_res_net, g_net_kwargs={}):
        super(affine_input_net, self).__init__()
        self.g_net_now = g_net(n_in=affine_dim,n_out=output_dim*input_dim,**g_net_kwargs)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.affine_dim = affine_dim

    def forward(self,z,u):
        gnow = self.g_net_now(z).view(-1,self.output_dim,self.input_dim)
        return torch.einsum('nij,nj->ni', gnow, u)

class affine_forward_layer(nn.Module):
    """
    Implantation of

    x_k+1 = A@x_k + g(x_k)@u_k

    where @ is matrix multiply
    """
    def __init__(self, nx, nu, g_net=simple_res_net, g_net_kwargs={}):
        super(affine_forward_layer, self).__init__()
        self.nu = 1 if nu is None else nu
        self.gpart = affine_input_net(output_dim=nx, input_dim=self.nu, affine_dim=nx, g_net=g_net, g_net_kwargs=g_net_kwargs)
        self.Apart = nn.Linear(nx, nx, bias=False)

    def forward(self,x,u):
        u = u.view(u.shape[0],-1) #flatten
        return self.Apart(x) + self.gpart(x,u)

class time_integrators(nn.Module):
    """docstring for time_integrators"""
    def __init__(self, deriv, dt_norm=1, dt_0=None, dt_now=None):
        '''include time normalization as dt = dt_norm*dt_now/dt_0'''
        super(time_integrators,self).__init__()
        self.dt_checked = False
        self.dt_valued = None


        self.dt_norm = dt_norm #normalized dt in units of x
                               #this a parameter?
        self.dt_0 = dt_0 #original time constant of fitted data 
                         #(set during first call of .fit using set_dt_0)
        self._dt = dt_now #the current time constant (most probably the same as dt_0)
                             #should be set using set_dt before applying any dataset
        self.deriv_base   = deriv #the deriv network

    def deriv(self,x,u):
        return self.deriv_base(x,u)*self.dt_norm/self.dt_0 if self.dt_0 is not None else self.deriv_base(x,u)*self.dt_norm


    @property
    def dt(self):
        #does this need a check?
        return self._dt
        # if not self.dt_checked:
        #     self.dt_checked = True
        #     self.dt_valued = False if self._dt is None else True
        # else:
        #     assert self.dt_valued==(self._dt is not None), 'are you mixing valued dt and none valued datasets?'
        # return 1 if self._dt is None else self._dt #could be None
        #I want to warn the user if dt is None but a dt was expected
        # if self.dt_0 is None: #no time constant set, assuming no changes in dt
        #     return self.dt_norm
        # return self.dt_norm*self.dt_now/self.dt_0

    @dt.setter
    def dt(self,dt):
        #put check here
        if not self.dt_checked:
            self.dt_checked = True
            self.dt_valued = False if dt is None else True
        else:
            assert self.dt_valued==(dt is not None), 'are you mixing valued dt and None dt valued datasets?'
        self._dt = 1. if dt is None else dt



class integrator_RK4(time_integrators):
    def forward(self, x, u): #u constant on segment, zero-order hold
        #put here a for loop
        k1 = self.dt*self.deriv(x,u) #t=0
        k2 = self.dt*self.deriv(x+k1/2,u) #t=dt/2
        k3 = self.dt*self.deriv(x+k2/2,u) #t=dt/2
        k4 = self.dt*self.deriv(x+k3,u) #t=dt
        return x + (k1 + 2*k2 + 2*k3 + k4)/6

class integrator_euler(time_integrators):
    def forward(self, x, u): #u constant on segment
        return x + self.dt*self.deriv(x,u)

if __name__ == '__main__':
    import deepSI
    import numpy as np
    from matplotlib import pyplot as plt
    import torch
    from torch import optim, nn
    from tqdm.auto import tqdm
    test, train = deepSI.datasets.CED() #switch
    train.plot()
    test.plot()
    print(train,test)
    from deepSI.utils import integrator_euler, integrator_RK4
    def get_sys_epoch(dt=0.12,nf=30,epochs=200,n_hidden_layers=1,n_nodes_per_layer=64,euler=False):
        test, train = deepSI.datasets.CED() #switch
        test.dt = None
        train.dt = None
        f_net_kwargs=dict(n_hidden_layers=n_hidden_layers,n_nodes_per_layer=n_nodes_per_layer)
        integrator = integrator_euler if euler else integrator_RK4
        sys = deepSI.fit_systems.SS_encoder_deriv_general(nx=3,na=7,nb=7,dt_norm=dt,\
                f_net_kwargs=f_net_kwargs,h_net_kwargs=dict(n_hidden_layers=2),integrator_net=integrator)
    #     while True:
        sys.fit(train,sim_val=test,concurrent_val=True,batch_size=32,\
                    epochs=epochs,verbose=2,loss_kwargs=dict(nf=nf))
    #         sys.checkpoint_load_system(name='_last')
    #         print(sys.bestfit,sys.epoch_id[np.argmin(sys.Loss_val)],sys.epoch_id[-1],sys.epoch_id[np.argmin(sys.Loss_val)]/sys.epoch_id[-1])
    #         if sys.epoch_id[np.argmin(sys.Loss_val)]/sys.epoch_id[-1]<0.5:
    #             break
        sys.checkpoint_load_system(name='_best')
        return sys

    from torch import manual_seed
    np.random.seed(5)
    manual_seed(5)
    sys = get_sys_epoch(nf=30, epochs=400, n_hidden_layers=1, n_nodes_per_layer=64)
    test.dt = None
    sys.apply_experiment(test)