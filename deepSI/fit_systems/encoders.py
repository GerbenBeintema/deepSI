
from deepSI.fit_systems.fit_system import System_fittable, System_torch
import deepSI
import torch
from torch import nn
import numpy as np

class SS_encoder(System_torch):
    """docstring for SS_encoder"""
    def __init__(self, nx=10, na=20, nb=20):
        super(SS_encoder, self).__init__() #where does dt come into
        self.nx, self.na, self.nb = nx, na, nb
        self.k0 = max(self.na,self.nb)
        
        from deepSI.utils import simple_res_net, feed_forward_nn
        self.e_net = simple_res_net
        self.e_n_hidden_layers = 2
        self.e_n_nodes_per_layer = 64
        self.e_activation = nn.Tanh

        self.f_net = simple_res_net
        self.f_n_hidden_layers = 2
        self.f_n_nodes_per_layer = 64
        self.f_activation = nn.Tanh

        self.h_net = simple_res_net
        self.h_n_hidden_layers = 2
        self.h_n_nodes_per_layer = 64
        self.h_activation = nn.Tanh

    ########## How to fit #############
    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        nf = Loss_kwargs.get('nf',25)
        dilation = Loss_kwargs.get('dilation',1)
        online_construct = Loss_kwargs.get('online_construct',False)
        return sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=nf,dilation=dilation,force_multi_u=True,force_multi_y=True,online_construct=online_construct) #returns np.array(hist),np.array(ufuture),np.array(yfuture)

    def init_nets(self, nu, ny): # a bit weird
        ny = ny if ny is not None else 1
        nu = nu if nu is not None else 1
        self.encoder = self.e_net(n_in=self.nb*nu+self.na*ny, n_out=self.nx, n_nodes_per_layer=self.e_n_nodes_per_layer, n_hidden_layers=self.e_n_hidden_layers, activation=self.e_activation)
        self.fn =      self.f_net(n_in=self.nx+nu,            n_out=self.nx, n_nodes_per_layer=self.f_n_nodes_per_layer, n_hidden_layers=self.f_n_hidden_layers, activation=self.f_activation)
        self.hn =      self.h_net(n_in=self.nx,               n_out=ny,      n_nodes_per_layer=self.h_n_nodes_per_layer, n_hidden_layers=self.h_n_hidden_layers, activation=self.h_activation)
        return list(self.encoder.parameters()) + list(self.fn.parameters()) + list(self.hn.parameters())

    def loss(self, hist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(hist)
        y_predict = []
        for u in torch.transpose(ufuture,0,1):
            y_predict.append(self.hn(x)) #output prediction
            fn_in = torch.cat((x,u),dim=1)
            x = self.fn(fn_in)
        return torch.mean((torch.stack(y_predict,dim=1)-yfuture)**2)

    ########## How to use ##############
    def init_state(self,sys_data): #put nf here for n-step error?
        hist = torch.tensor(sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=len(sys_data)-max(self.na,self.nb))[0][:1],dtype=torch.float32) #(1,)
        with torch.no_grad():
            self.state = self.encoder(hist) #detach here?
        y_predict = self.hn(self.state).detach().numpy()[0,:]
        return (y_predict[0] if self.ny is None else y_predict), max(self.na,self.nb)

    def init_state_multi(self,sys_data,nf=100,dilation=1):
        hist = torch.tensor(sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=nf,dilation=dilation)[0],dtype=torch.float32) #(1,)
        with torch.no_grad():
            self.state = self.encoder(hist)
        y_predict = self.hn(self.state).detach().numpy()
        return (y_predict[:,0] if self.ny is None else y_predict), max(self.na,self.nb)

    def reset(self): #to be able to use encoder network as a data generator
        self.state = torch.randn(1,self.nx)
        y_predict = self.hn(self.state).detach().numpy()[0,:]
        return (y_predict[0] if self.ny is None else y_predict)

    def step(self,action):
        action = torch.tensor(action,dtype=torch.float32) #number
        action = action[None,None] if self.nu is None else action[None,:]
        with torch.no_grad():
            self.state = self.fn(torch.cat((self.state,action),axis=1))
        y_predict = self.hn(self.state).detach().numpy()[0,:]
        return (y_predict[0] if self.ny is None else y_predict)

    def step_multi(self,action):
        action = torch.tensor(action,dtype=torch.float32) #array
        action = action[:,None] if self.nu is None else action
        with torch.no_grad():
            self.state = self.fn(torch.cat((self.state,action),axis=1))
        y_predict = self.hn(self.state).detach().numpy()
        return (y_predict[:,0] if self.ny is None else y_predict)

    def get_state(self):
        return self.state[0].numpy()

class default_encoder_net(nn.Module):
    def __init__(self, nb, nu, na, ny, nx, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(default_encoder_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.net = simple_res_net(n_in=nb*np.prod(self.nu,dtype=int) + na*np.prod(self.ny,dtype=int), n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, upast, ypast):
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1)
        return self.net(net_in)

class default_state_net(nn.Module):
    def __init__(self, nx, nu, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(default_state_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.net = simple_res_net(n_in=nx+np.prod(self.nu,dtype=int), n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, x, u):
        net_in = torch.cat([x,u.view(u.shape[0],-1)],axis=1)
        return self.net(net_in)

class default_output_net(nn.Module):
    def __init__(self, nx, ny, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(default_output_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.net = simple_res_net(n_in=nx, n_out=np.prod(self.ny,dtype=int), n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, x):
        return self.net(x).view(*((x.shape[0],)+self.ny))
    

class SS_encoder_general(System_torch):
    """docstring for SS_encoder_general"""
    def __init__(self, nx=10, na=20, nb=20, e_net=default_encoder_net, f_net=default_state_net, h_net=default_output_net, e_net_kwargs={}, f_net_kwargs={}, h_net_kwargs={}):
        super(SS_encoder_general, self).__init__()
        self.nx, self.na, self.nb = nx, na, nb
        self.k0 = max(self.na,self.nb)
        
        self.e_net = e_net
        self.e_net_kwargs = e_net_kwargs

        self.f_net = f_net
        self.f_net_kwargs = f_net_kwargs

        self.h_net = h_net
        self.h_net_kwargs = h_net_kwargs

    ########## How to fit #############
    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        nf = Loss_kwargs.get('nf',25)
        dilation = Loss_kwargs.get('dilation',1)
        online_construct = Loss_kwargs.get('online_construct',False)
        return sys_data.to_hist_future_data(na=self.na,nb=self.nb,nf=nf,dilation=dilation,online_construct=online_construct) #uhist, yhist, ufuture, yfuture

    def init_nets(self, nu, ny): # a bit weird
        self.encoder = self.e_net(nb=self.nb, nu=nu, na=self.na, ny=ny, nx=self.nx, **self.e_net_kwargs)
        self.fn =      self.f_net(nx=self.nx, nu=nu,                                **self.f_net_kwargs)
        self.hn =      self.h_net(nx=self.nx, ny=ny,                                **self.h_net_kwargs) 
        return list(self.encoder.parameters()) + list(self.fn.parameters()) + list(self.hn.parameters())

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(uhist, yhist)
        y_predict = []
        for u in torch.transpose(ufuture,0,1): #iterate over time
            y_predict.append(self.hn(x)) 
            x = self.fn(x,u)
        return torch.mean((torch.stack(y_predict,dim=1)-yfuture)**2)

    ########## How to use ##############
    def init_state(self,sys_data): #put nf here for n-step error?
        uhist, yhist = sys_data[:self.k0].to_hist_future_data(na=self.na,nb=self.nb,nf=0)[:2]
        uhist = torch.tensor(uhist,dtype=torch.float32)
        yhist = torch.tensor(yhist,dtype=torch.float32)
        with torch.no_grad():
            self.state = self.encoder(uhist, yhist) #detach here?
            y_predict = self.hn(self.state).numpy()[0]
        return y_predict, max(self.na,self.nb)

    def init_state_multi(self,sys_data,nf=100,dilation=1):
        uhist, yhist = sys_data.to_hist_future_data(na=self.na,nb=self.nb,nf=nf,dilation=dilation)[:2] #(1,)
        uhist = torch.tensor(uhist,dtype=torch.float32)
        yhist = torch.tensor(yhist,dtype=torch.float32)
        with torch.no_grad():
            self.state = self.encoder(uhist,yhist)
            y_predict = self.hn(self.state).numpy()
        return y_predict, max(self.na,self.nb)

    def reset(self): #to be able to use encoder network as a data generator
        self.state = torch.zeros(1,self.nx)
        with torch.no_grad():
            y_predict = self.hn(self.state).numpy()[0]
        return y_predict

    def step(self,action):
        action = torch.tensor(action,dtype=torch.float32)[None] #(1,...)
        with torch.no_grad():
            self.state = self.fn(self.state,action)
            y_predict = self.hn(self.state).numpy()[0]
        return y_predict

    def step_multi(self,action):
        action = torch.tensor(action,dtype=torch.float32) #(N,...)
        with torch.no_grad():
            self.state = self.fn(self.state,action)
            y_predict = self.hn(self.state).numpy()
        return y_predict

    def get_state(self):
        return self.state[0].numpy()


############## Continuous time ##################
from deepSI.utils import integrator_RK4, integrator_euler
class SS_encoder_deriv_general(SS_encoder_general):
    """For backwards compatibility fn is the advance function"""
    def __init__(self, nx=10, na=20, nb=20, f_norm=0.1, dt_base=1., cutt_off=1.5, \
                 e_net=default_encoder_net, f_net=default_state_net, integrator_net=integrator_RK4, h_net=default_output_net, \
                 e_net_kwargs={},           f_net_kwargs={},         integrator_net_kwargs={},       h_net_kwargs={}):
        super(SS_encoder_deriv_general, self).__init__(nx=nx, na=na, nb=nb, e_net=e_net, f_net=f_net, h_net=h_net, e_net_kwargs=e_net_kwargs, f_net_kwargs=f_net_kwargs, h_net_kwargs=h_net_kwargs)
        self.integrator_net = integrator_net
        self.integrator_net_kwargs = integrator_net_kwargs
        self.f_norm = f_norm
        self.dt_base = dt_base #freal = f_norm/dt_base simple rescale factor which is often used
        self.cutt_off = cutt_off

    def init_nets(self, nu, ny): # a bit weird
        par = super(SS_encoder_deriv_general, self).init_nets(nu,ny) 
        self.derivn = self.fn  #move fn to become the deriviative net
        self.fn = self.integrator_net(self.derivn, f_norm=self.f_norm, dt_base=self.dt_base, **self.integrator_net_kwargs) #has no torch parameters?
        return par

    @property
    def dt(self):
        return self._dt 

    @dt.setter
    def dt(self,dt):
        self._dt = dt
        self.fn.dt = dt

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(uhist, yhist) #this fails if dt starts to change
        diff = []
        for u,y in zip(torch.transpose(ufuture,0,1), torch.transpose(yfuture,0,1)): #iterate over time
            yhat = self.hn(x)
            dy = yhat-y # (Nbatch, ny)
            with torch.no_grad(): #break if the 
                if torch.mean(dy**2).item()**0.5>self.cutt_off:
                    break
            diff.append(dy)
            x = self.fn(x,u)
        return torch.mean((torch.stack(diff,dim=1))**2)

class SS_encoder_rnn(System_torch):
    """docstring for SS_encoder_rnn"""
    def __init__(self, hidden_size=10, num_layers=2, na=20, nb=20):
        super(SS_encoder_rnn, self).__init__(None,None)
        self.na = na
        self.nb = nb
        from deepSI.utils import simple_res_net, feed_forward_nn
        self.net = simple_res_net
        self.n_hidden_layers = 2
        self.n_nodes_per_layer = 64
        self.activation = nn.Tanh

        #RNN parameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        nf = Loss_kwargs.get('nf',25)
        online_construct = Loss_kwargs.get('online_construct',False)
        return sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=nf,online_construct=online_construct) #returns np.array(hist),np.array(ufuture),np.array(yfuture)

    def init_nets(self, nu, ny): # a bit weird
        # print(nu,ny)
        assert ny==None and nu==None
        ny = 1
        nu = 1

        self.rnn = nn.RNN(input_size=nu,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True) #batch_first yes
        # output, h_n = self.RNN(input,h_0) #input = (batch, seq_len, input_size), 
        #h_0 = (num_layers, batch, hidden_size)
        #outputs = (batch, seq_len, hidden_size) #last layer

        #encoder: self.nb + self.na -> h_0 = (num_layers, batch, hidden_size)
        #hn: (batch*seq_len, hidden_size) -> (batch*seq_len, ny)
        self.encoder = self.net(n_in=self.nb+self.na, n_out=self.hidden_size*self.num_layers, n_nodes_per_layer=self.n_nodes_per_layer, n_hidden_layers=self.n_hidden_layers, activation=self.activation)
        self.hn =      self.net(n_in=self.hidden_size,n_out=ny,      n_nodes_per_layer=self.n_nodes_per_layer, n_hidden_layers=self.n_hidden_layers, activation=self.activation)
        return list(self.encoder.parameters()) + list(self.rnn.parameters()) + list(self.hn.parameters())

    def loss(self, hist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(hist) # (s, nhist = nb + na) -> (s, hidden_size*num_layers)
        h_0 = x.view(-1, self.num_layers, self.hidden_size).permute(1,0,2)   # to (num_layers, s, hidden_size)

        # ufuture (batch, seq_len)
        output, h_n = self.rnn(ufuture[:,:-1,None], h_0) #do not use the last u
        #print(output.shape) #has shape (s, seq_len-1, hidden_size)
        output = torch.cat((h_0[-1][:,None,:],output),dim=1)
        #outputs = (batch, seq_len, hidden_size) -> (batch*seq_len, hidden_size)
        h_in = output.reshape(-1,self.hidden_size)
        y_predict = self.hn(h_in).view(output.shape[0], output.shape[1])
        return torch.mean((y_predict-yfuture)**2)

    def init_state(self,sys_data): #put nf here for n-step error?
        hist = torch.tensor(sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=len(sys_data)-max(self.na,self.nb))[0][:1],dtype=torch.float32) #(1,)
        self.state = self.encoder(hist).view(-1, self.num_layers, self.hidden_size).permute(1,0,2)
        return self.hn(self.state[-1,:,:])[0,0].item(), max(self.na,self.nb) #some error is being made here

    def init_state_multi(self,sys_data,nf=100,dilation=1):
        hist = torch.tensor(sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=nf,dilation=dilation)[0],dtype=torch.float32) #(1,)
        self.state = self.encoder(hist).view(-1, self.num_layers, self.hidden_size).permute(1,0,2)
        return self.hn(self.state[-1,:,:])[:,0].detach().numpy(), max(self.na,self.nb)

    def step(self,action):
        action = torch.tensor(action,dtype=torch.float32) #number
        output, self.state = self.rnn(action[None,None,None], self.state)
        return self.hn(output[:,0,:])[0,0].item()

    def step_multi(self,action):
        action = torch.tensor(action,dtype=torch.float32) #array
        output, self.state = self.rnn(action[:,None,None], self.state)
        return self.hn(output[:,0,:])[:,0].detach().numpy()

    def get_state(self):
        return self.state[0].numpy()

class par_start_encoder(nn.Module):
    """docstring for par_start_encoder"""
    def __init__(self, nx, nsamples):
        super(par_start_encoder, self).__init__()
        self.start_state = nn.parameter.Parameter(data=torch.as_tensor(np.random.normal(scale=0.1,size=(nsamples,nx)),dtype=torch.float32))

    def forward(self,ids):
        return self.start_state[ids]

class SS_par_start(System_torch): #this is not implemented in a nice manner, there might be bugs.
    """docstring for SS_par_start"""
    def __init__(self, nx=10):
        super(SS_par_start, self).__init__()
        self.nx = nx
        self.k0 = 0

        from deepSI.utils import simple_res_net, feed_forward_nn
        self.f_net = simple_res_net
        self.f_n_hidden_layers = 2
        self.f_n_nodes_per_layer = 64
        self.f_activation = nn.Tanh

        self.h_net = simple_res_net
        self.h_n_hidden_layers = 2
        self.h_n_nodes_per_layer = 64
        self.h_activation = nn.Tanh

    def init_optimizer(self, parameters, **optimizer_kwargs):
        '''Optionally defined in subclass to create the optimizer

        Parameters
        ----------
        parameters : list
            system torch parameters
        optimizer_kwargs : dict
            If 'optimizer' is defined than that optimizer will be used otherwise Adam will be used.
            The other parameters will be passed to the optimizer as a kwarg.
        '''
        self.optimizer_kwargs = optimizer_kwargs
        if optimizer_kwargs.get('optimizer') is not None:
            from copy import deepcopy
            optimizer_kwargs = deepcopy(optimizer_kwargs) #do not modify the original kwargs, is this necessary
            optimizer = optimizer_kwargs['optimizer']
            del optimizer_kwargs['optimizer']
        else:
            optimizer = torch.optim.Adam
        return optimizer(parameters,**optimizer_kwargs) 


    ########## How to fit #############
    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        nf = Loss_kwargs.get('nf',25)
        dilation = Loss_kwargs.get('dilation',1)
        online_construct = Loss_kwargs.get('online_construct',False)
        assert online_construct==False, 'to be implemented'
        hist, ufuture, yfuture = sys_data.to_encoder_data(na=0,nb=0,nf=nf,dilation=dilation,force_multi_u=True,force_multi_y=True)
        nsamples = hist.shape[0]
        ids = np.arange(nsamples,dtype=int)
        self.par_starter = par_start_encoder(nx=self.nx, nsamples=nsamples)
        self.optimizer = self.init_optimizer(self.parameters+list(self.par_starter.parameters()), **self.optimizer_kwargs) #no kwargs

        return ids, ufuture, yfuture #returns np.array(hist),np.array(ufuture),np.array(yfuture)

    def init_nets(self, nu, ny): # a bit weird
        ny = ny if ny is not None else 1
        nu = nu if nu is not None else 1
        self.fn =      self.f_net(n_in=self.nx+nu,            n_out=self.nx, n_nodes_per_layer=self.f_n_nodes_per_layer, n_hidden_layers=self.f_n_hidden_layers, activation=self.f_activation)
        self.hn =      self.h_net(n_in=self.nx,               n_out=ny,      n_nodes_per_layer=self.h_n_nodes_per_layer, n_hidden_layers=self.h_n_hidden_layers, activation=self.h_activation)
        return list(self.fn.parameters()) + list(self.hn.parameters())

    def loss(self, ids, ufuture, yfuture, **Loss_kwargs):
        ids = ids.numpy().astype(int)
        #hist is empty
        x = self.par_starter(ids)
        y_predict = []
        for u in torch.transpose(ufuture,0,1):
            y_predict.append(self.hn(x)) #output prediction
            fn_in = torch.cat((x,u),dim=1)
            x = self.fn(fn_in)
        return torch.mean((torch.stack(y_predict,dim=1)-yfuture)**2)

    ########## How to use ##############
    def init_state(self,sys_data): #put nf here for n-step error?
        with torch.no_grad():
            self.state = torch.as_tensor(np.random.normal(scale=0.1,size=(1,self.nx)),dtype=torch.float32) #detach here?
        y_predict = self.hn(self.state).detach().numpy()[0,:]
        return (y_predict[0] if self.ny is None else y_predict), 0

    def init_state_multi(self,sys_data,nf=100,dilation=1):
        hist = torch.tensor(sys_data.to_encoder_data(na=0,nb=0,nf=nf,dilation=dilation)[0],dtype=torch.float32) #(1,)
        with torch.no_grad():
            self.state = torch.as_tensor(np.random.normal(scale=0.1,size=(hist.shape[0],self.nx)),dtype=torch.float32) #detach here?
        y_predict = self.hn(self.state).detach().numpy()
        return (y_predict[:,0] if self.ny is None else y_predict), 0

    def reset(self): #to be able to use encoder network as a data generator
        self.state = torch.randn(1,self.nx)
        y_predict = self.hn(self.state).detach().numpy()[0,:]
        return (y_predict[0] if self.ny is None else y_predict)

    def step(self,action):
        action = torch.tensor(action,dtype=torch.float32) #number
        action = action[None,None] if self.nu is None else action[None,:]
        with torch.no_grad():
            self.state = self.fn(torch.cat((self.state,action),axis=1))
        y_predict = self.hn(self.state).detach().numpy()[0,:]
        return (y_predict[0] if self.ny is None else y_predict)

    def step_multi(self,action):
        action = torch.tensor(action,dtype=torch.float32) #array
        action = action[:,None] if self.nu is None else action
        with torch.no_grad():
            self.state = self.fn(torch.cat((self.state,action),axis=1))
        y_predict = self.hn(self.state).detach().numpy()
        return (y_predict[:,0] if self.ny is None else y_predict)

    def get_state(self):
        return self.state[0].numpy()


from deepSI.utils import simple_res_net, feed_forward_nn, affine_forward_layer
class SS_encoder_affine_input(SS_encoder_general):
    """
    The encoder setup with a linear transition function with an affine input (kinda like an LPV), in equations

        x_k = e(u_kpast,y_kpast) 
        x_k+1 = A@x_k + g(x_k)@u_k          #affine input here (@=matrix multiply)
        y_k = h(x_k)

    Where g is given by g_net which is by default a feedforward nn with residual (i.e. simple_res_net) called as 
        'g_net(n_in=affine_dim,n_out=output_dim*input_dim,**g_net_kwargs)'
    with affine_dim=nx, output_dim = nx, input_dim=nu

    Hence, g_net produces a vector which is reshaped into a matrix (See deepSI.utils.torch_nets.affine_forward_layer for details).
    """
    def __init__(self, nx=10, na=20, nb=20, e_net=default_encoder_net, g_net=simple_res_net, h_net=default_output_net, e_net_kwargs={}, g_net_kwargs={}, h_net_kwargs={}):
        super(SS_encoder_affine_input, self).__init__(nx=nx,na=na,nb=nb,\
            e_net=e_net,f_net=affine_forward_layer, h_net=h_net, \
            e_net_kwargs=e_net_kwargs, f_net_kwargs=dict(g_net=g_net,g_net_kwargs=g_net_kwargs), h_net_kwargs=h_net_kwargs)


class default_ino_state_net(nn.Module):
    def __init__(self, nx, nu, ny, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh): #ny here?
        super(default_ino_state_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.nx = nx
        self.net = simple_res_net(n_in=nx+np.prod(self.nu,dtype=int), n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)
        self.K = nn.Linear(np.prod(self.ny,dtype=int),nx,bias=False)

    def forward(self, x, u, eps=None):
        net_in = torch.cat([x,u.view(u.shape[0],-1)],axis=1)
        if eps==None:
            return self.net(net_in)
        else:
            epsflat = eps.view(eps.shape[0],-1)
            return self.net(net_in) + self.K(epsflat)

class SS_encoder_inovation(SS_encoder_general):
    """
    Similar to SS encoder but with the structure of 
       x_k+1 = f(x_k,u_k,eps_k)
       y_k = h(x_k) + eps_k

    During optimization eps will be set to
    eps_k = y_k - h(x_k)

    During simulation eps will be set to zero (None)
    eps_k = 0

    Futhermore this requires a specialized f net build as 
        self.fn = self.f_net(nx=self.nx, nu=nu, ny=self.ny,**self.f_net_kwargs)

    default structure for f is as follows:
    f(x_k,u_k,eps_k) = f0(x_k,u_k) + K eps_k    (with K a matrix)

    """
    def __init__(self, nx=10, na=20, nb=20, e_net=default_encoder_net, f_net=default_ino_state_net, h_net=default_output_net, e_net_kwargs={}, f_net_kwargs={}, h_net_kwargs={}):
        super(SS_encoder_inovation, self).__init__(nx=nx, na=na, nb=nb, e_net=e_net, f_net=f_net, h_net=h_net, e_net_kwargs=e_net_kwargs, f_net_kwargs=f_net_kwargs, h_net_kwargs=h_net_kwargs)

    def init_nets(self, nu, ny): # a bit weird
        self.encoder = self.e_net(nb=self.nb, nu=nu, na=self.na, ny=ny, nx=self.nx, **self.e_net_kwargs)
        self.fn =      self.f_net(nx=self.nx, nu=nu, ny=self.ny,                    **self.f_net_kwargs)
        self.hn =      self.h_net(nx=self.nx, ny=ny,                                **self.h_net_kwargs) 
        return list(self.encoder.parameters()) + list(self.fn.parameters()) + list(self.hn.parameters())

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(uhist, yhist)
        y_predict = []
        for u,y in zip(torch.transpose(ufuture,0,1),torch.transpose(yfuture,0,1)): #iterate over time
            yhat = self.hn(x)
            y_predict.append(yhat) 
            x = self.fn(x,u,eps=y-yhat)
        return torch.mean((torch.stack(y_predict,dim=1)-yfuture)**2)
        

if __name__ == '__main__':
    # sys = SS_encoder_general()
    # from deepSI.datasets.sista_database import powerplant
    # from deepSI.datasets import Silverbox
    from deepSI.datasets import Cascaded_Tanks
    
    train, test = Cascaded_Tanks()#powerplant()
    train.dt = 0.1
    test.dt = 0.1
    # train, test = train[:150], test[:50]
    # print(train, test)
    # # sys.fit(train, sim_val=test,epochs=50)
    # import deepSI
    # test2 = deepSI.system_data.System_data_list([test,test])
    # sys.fit(train, sim_val=test2, epochs=50, concurrent_val=True)

    # # fit_val_multiprocess
    # train_predict = sys.apply_experiment(train)
    # train.plot()
    # train_predict.plot(show=True)
    # from matplotlib import pyplot as plt
    # plt.plot(sys.n_step_error(train,nf=20))
    # plt.show()
    sys = SS_encoder_deriv_general_V2(nx=2,f_norm=0.025)
    # sys = SS_par_start()
    sys.fit(train, sim_val=test, epochs=50, batch_size=32, concurrent_val=True)
