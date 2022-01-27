

from deepSI.fit_systems.fit_system import System_fittable, System_torch
from deepSI.system_data import System_data
import torch
from torch import nn
import numpy as np


class IO_autoencoder(System_torch):
    """docstring for IO_autoencoder"""
    def __init__(self, nz=4, na=5, nb=5):
        super(IO_autoencoder, self).__init__()
        self.nz, self.na, self.nb = nz, na, nb
        self.k0 = max(self.na,self.nb)
        
        from deepSI.utils import simple_res_net, feed_forward_nn
        self.net = simple_res_net
        self.n_hidden_layers = 2
        self.n_nodes_per_layer = 64
        self.activation = nn.Tanh

    ########## How to fit #############
    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        return sys_data.to_hist_future_data(na=self.na,nb=self.nb,nf=1,force_multi_u=True,force_multi_y=True) #returns uhist, yhist, ufuture, yfuture

    def init_nets(self, nu, ny): # a bit weird
        ny = ny if ny is not None else 1
        nu = nu if nu is not None else 1
        #g_n decoder, g_inv_n encoder 
        self.g_n =         self.net(n_in=self.nz,                     n_out=ny,      n_nodes_per_layer=self.n_nodes_per_layer, n_hidden_layers=self.n_hidden_layers, activation=self.activation)
        self.f_n =         self.net(n_in=self.nz*self.na + self.nb*nu,n_out=self.nz, n_nodes_per_layer=self.n_nodes_per_layer, n_hidden_layers=self.n_hidden_layers, activation=self.activation)
        self.g_inv_n =     self.net(n_in=ny,                          n_out=self.nz, n_nodes_per_layer=self.n_nodes_per_layer, n_hidden_layers=self.n_hidden_layers, activation=self.activation)
        return list(self.g_n.parameters()) + list(self.f_n.parameters()) + list(self.g_inv_n.parameters())

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        #uhist (s, nb, nu)
        #yhist (s, na, ny)
        #ufuture (s, nf=1, nu) (not used)
        #yfuture (s, nf=1, ny)
        yhist_flat = yhist.reshape(-1,yhist.shape[2])  #(s*na, ny)
        zvec = self.g_inv_n(yhist_flat) #(s*na, ny) -> (s*na, nz)
        yhist_flat_predict = self.g_n(zvec) #(s*na, nz) -> (s*na, ny)
        Loss1 = torch.mean((yhist_flat-yhist_flat_predict)**2)

        zvec = zvec.reshape(yhist.shape[0], yhist.shape[1], zvec.shape[1]) #reshape it as (s, na, nz)

        f_in = torch.cat((zvec.flatten(start_dim=1),uhist.flatten(start_dim=1)),axis=1) #(s, na*nz) + (s, nb*nu) = (s, na*nz + nb*nu)
        znext = self.f_n(f_in) # (s, nz)
        ynext = self.g_n(znext) # (s, nz) -> (s, ny)
        Loss2 = torch.mean((ynext-yfuture[:,0,:])**2)

        return Loss1 + Loss2

    def init_state_and_measure_multi(self,sys_data,nf=100,stride=1):
        uhist, yhist, ufuture, yfuture = sys_data.to_hist_future_data(na=self.na,nb=self.nb,nf=nf,force_multi_u=True,force_multi_y=True)
        yhist = torch.tensor(yhist,dtype=torch.float32)
        uhist = torch.tensor(uhist,dtype=torch.float32)

        yhist_flat = yhist.reshape(-1,yhist.shape[2])  #(s*na, ny)
        self.zvec = self.g_inv_n(yhist_flat).detach().reshape(yhist.shape[0], self.na, self.nz) #(s*na, ny) -> (s*na, nz) -> (s,na,nz)
        self.uhist = uhist
        f_in = torch.cat((self.zvec.flatten(start_dim=1),self.uhist.flatten(start_dim=1)),axis=1) #(s, na*nz) + (s, nb*nu) = (s, na*nz + nb*nu)
        znext = self.f_n(f_in) # (s, nz)
        y_predict = self.g_n(znext).detach().numpy() # (s, nz) -> (s, ny)
        #advance state
        self.zvec = torch.cat((self.zvec[:,1:,:],znext[:,None,:]),axis=1) #add to time dime (1, na, nz)
        self.uhist = self.uhist[:,1:] #(1, nb-1, nu)
        return (y_predict[:,0] if self.ny is None else y_predict), max(self.na,self.nb)

    def act_measure_multi(self,action):
        action = torch.tensor(action,dtype=torch.float32) #array
        action = action[:,None] if self.nu is None else action
        self.uhist = torch.cat((self.uhist, action[:,None,:]),axis=1)
        f_in = torch.cat((self.zvec.flatten(start_dim=1),self.uhist.flatten(start_dim=1)),axis=1) #(s, na*nz) + (s, nb*nu) = (s, na*nz + nb*nu)
        znext = self.f_n(f_in) # (s, nz)
        y_predict = self.g_n(znext).detach().numpy() # (s, nz) -> (s, ny)
        self.zvec = torch.cat((self.zvec[:,1:,:],znext[:,None,:]),axis=1) #add to time dime (1, na, nz)
        self.uhist = self.uhist[:,1:] #(1, nb-1, nu)
        return (y_predict[:,0] if self.ny is None else y_predict)


if __name__ == '__main__':
    x = torch.randn(7,2,3)
    print(x.flatten(start_dim=1).shape)
    print(x.shape)
    sys = IO_autoencoder()
    sys_data = System_data(u=np.random.normal(size=(1000,2)),y=np.random.normal(size=(1000,7)))
    sys.fit(sys_data,batch_size=64,verbose=2)
    sys_data_sim = sys.apply_experiment(sys_data)
