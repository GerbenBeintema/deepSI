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
        self.net_non_lin = feed_forward_nn(n_in,n_out,n_nodes_per_layer=n_nodes_per_layer,n_hidden_layers=n_hidden_layers,activation=activation)
    def forward(self,x):
        return self.net_lin(x) + self.net_non_lin(x)


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
    def __init__(self, dt=None):
        super(time_integrators,self).__init__()
        self.dt = dt
    def deriv(self,x,u):
        raise NotImplementedError('deriv')    

class integrators_RK4(time_integrators):
    def forward(self, x, u): #almost
        assert self.dt is not None, 'error dt not set'
        k1 = self.dt*self.deriv(x,u)
        k2 = self.dt*self.deriv(x+k1/2,u)
        k3 = self.dt*self.deriv(x+k2/2,u)
        k4 = self.dt*self.deriv(x+k3,u)
        return x + (k1 + 2*k2 + 2*k3 + k4)/6

