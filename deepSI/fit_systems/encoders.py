
from deepSI.fit_systems.fit_system import System_fittable, System_torch
from deepSI.system_data.system_data import System_data_norm, System_data, System_data_list
import deepSI
import torch
from torch import nn
import numpy as np
import time

class SS_encoder(System_torch):
    '''The basic implementation of the subspace encoder, with neural networks.

    Attributes
    ----------
    nx : int
        order of the system
    na : int
        length of the past outputs (y) considered as input for the encoder
    nb : int
        length of the past inputs (u) considered as input for the encoder
    k0 : int
        length of the encoder max(na,nb)
    e_net : 


    observation_space : gym.space or None
        The input shape of output y. (None is a single unbounded float)
    norm : instance of System_data_norm
        Used in most fittable systems to normalize the input output.
    fitted : Boole
    unique_code : str
        Some random unique 4 digit code (can be used for saving/loading)
    name : str
        concatenation of the the class name and the unique code
    seed : int
        random seed
    random : np.random.RandomState
        unique random generated initialized with seed (only created ones called)
    '''
    def __init__(self, nx = 10, na=20, nb=20, feedthrough=False):
        super(SS_encoder,self).__init__()
        self.na, self.nb = na, nb
        self.k0 = max(self.na, self.nb)
        self.nx = nx

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
        
        self.feedthrough = feedthrough

    def make_training_data(self, sys_data, **loss_kwargs):
        assert sys_data.normed == True
        nf = loss_kwargs.get('nf',25)
        stride = loss_kwargs.get('stride',1)
        online_construct = loss_kwargs.get('online_construct',False)
        return sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=nf,stride=stride,\
        force_multi_u=True,force_multi_y=True,online_construct=online_construct) #returns np.array(hist),np.array(ufuture),np.array(yfuture)
        
    def init_nets(self, nu, ny): # a bit weird
        ny = ny if ny is not None else 1
        nu = nu if nu is not None else 1
        self.encoder = self.e_net(self.nb*nu+self.na*ny, self.nx, n_nodes_per_layer=self.e_n_nodes_per_layer, n_hidden_layers=self.e_n_hidden_layers, activation=self.e_activation)
        self.fn =      self.f_net(self.nx+nu,            self.nx, n_nodes_per_layer=self.f_n_nodes_per_layer, n_hidden_layers=self.f_n_hidden_layers, activation=self.f_activation)
        hn_in = self.nx + nu if self.feedthrough else self.nx
        self.hn =      self.h_net(hn_in     ,            ny,      n_nodes_per_layer=self.h_n_nodes_per_layer, n_hidden_layers=self.h_n_hidden_layers, activation=self.h_activation)
    
    def loss(self, hist, ufuture, yfuture, **loss_kwargs):
        x = self.encoder(hist) #(N, nb*nu + na*ny) -> (N, nx)
        y_predicts = []
        for u in torch.transpose(ufuture,0,1):
            xu = torch.cat((x,u),dim=1)
            y_predicts.append(self.hn(xu if self.feedthrough else x)) #output prediction
            x = self.fn(xu) #calculate the next state
        y_predicts = torch.stack(y_predicts,dim=1) #[Nt, Nb, ny] to [Nb, Nt, ny]
        return torch.nn.functional.mse_loss(y_predicts, yfuture)
    
    def init_state_multi(self,sys_data,nf=100,stride=1):
        hist = torch.tensor(sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=nf,stride=stride)[0],dtype=torch.float32) #(1,)
        with torch.no_grad():
            self.state = self.encoder(hist)
        return max(self.na,self.nb)
    
    def measure_act_multi(self, actions):
        actions = torch.tensor(np.array(actions),dtype=torch.float32) 
        actions = actions[:,None] if self.nu is None else actions
        with torch.no_grad():
            xu = torch.cat([self.state, actions],dim=1)
            feedthrough = self.feedthrough if hasattr(self,'feedthrough') else False
            y_predict = self.hn(xu if feedthrough else self.state).numpy()
            self.state = self.fn(xu)
        return (y_predict[:,0] if self.ny is None else y_predict)
    
    def reset_state(self):
        self.state = torch.zeros((1,self.nx), dtype=torch.float32)
    
    def get_state(self):
        return self.state.numpy()[0]

class default_encoder_net(nn.Module):
    def __init__(self, nb, nu, na, ny, nx, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(default_encoder_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.net = simple_res_net(n_in=nb*np.prod(self.nu,dtype=int) + na*np.prod(self.ny,dtype=int), \
            n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, upast, ypast):
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast.view(ypast.shape[0],-1)],axis=1)
        return self.net(net_in)

class default_state_net(nn.Module):
    def __init__(self, nx, nu, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(default_state_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.net = simple_res_net(n_in=nx+np.prod(self.nu,dtype=int), n_out=nx, n_nodes_per_layer=n_nodes_per_layer, \
            n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, x, u):
        net_in = torch.cat([x,u.view(u.shape[0],-1)],axis=1)
        return self.net(net_in)

class default_output_net(nn.Module):
    def __init__(self, nx, ny, nu=-1, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(default_output_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.feedthrough = nu!=-1
        if self.feedthrough:
            self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
            net_in = nx + np.prod(self.nu, dtype=int)
        else:
            net_in = nx
        self.net = simple_res_net(n_in=net_in, n_out=np.prod(self.ny,dtype=int), n_nodes_per_layer=n_nodes_per_layer, \
            n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, x, u=None):
        feedthrough = self.feedthrough if hasattr(self,'feedthrough') else False
        if feedthrough:
            xu = torch.cat([x,u.view(u.shape[0],-1)],dim=1)
        else:
            xu = x
        return self.net(xu).view(*((x.shape[0],)+self.ny))

def to_torch_if_needed(*args, dtype=torch.float32):
    assert len(args)>0
    out = []
    for ar in args:
        if isinstance(ar, (list, tuple, float, int)):
            ar = np.array(ar)
        if isinstance(ar, np.ndarray):
            ar = torch.as_tensor(ar, dtype=dtype)
        out.append(ar)
    return out[0] if len(args)==1 else out

def check_shapes(tensor, shape):
    #the first dimension is a batch dimension when needed
    assert (len(tensor.shape)==len(shape)) or (len(tensor.shape)-1==len(shape))
    batched = len(tensor.shape)-1==len(shape)
    found_shape = tensor.shape[1:] if batched else tensor.shape
    assert len(found_shape)==len(shape)
    for n1,n2 in zip(shape, found_shape):
        assert n1==n2, f'target shape={shape} and found shape = {found_shape}'
    return (tensor, batched) if batched==True else (tensor[None], batched)


def ONNX_export(fun, network, filename, output_names, batched=False, **inputs):
    class Temp_function_module(nn.Module):
        def __init__(self, fun, network_name):
            super().__init__()
            self.fun = fun
            self.network = getattr(self.fun.__self__, network_name)
            
        def forward(self, *args):
            return self.fun(*args)

    f = Temp_function_module(fun,network)
    inputs_values = tuple([item for key,item in inputs.items()])
    inputs_names = tuple([key for key,item in inputs.items()])
    if not isinstance(output_names,list):
        output_names = tuple([output_names])
    
    dynamic_axes = None if batched==False else {key : {0 : 'batch_size'} for key in inputs_names + output_names}

    torch.onnx.export(f,               # model being run
                    inputs_values,                         # model input (or a tuple for multiple inputs)
                    filename,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = inputs_names,   # the model's input names
                    output_names = output_names,
                    dynamic_axes=dynamic_axes)

class SS_encoder_general(System_torch):
    """docstring for SS_encoder_general"""
    def __init__(self, nx=10, na=20, nb=20, feedthrough=False, \
        e_net=default_encoder_net, f_net=default_state_net, h_net=default_output_net, \
        e_net_kwargs={},           f_net_kwargs={},         h_net_kwargs={}, na_right=0, nb_right=0):

        super(SS_encoder_general, self).__init__()
        self.nx, self.na, self.nb = nx, na, nb
        self.k0 = max(self.na,self.nb)
        
        self.e_net = e_net
        self.e_net_kwargs = e_net_kwargs

        self.f_net = f_net
        self.f_net_kwargs = f_net_kwargs

        self.h_net = h_net
        self.h_net_kwargs = h_net_kwargs

        self.feedthrough = feedthrough
        self.na_right = na_right
        self.nb_right = nb_right

    ########## How to fit #############
    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        nf = Loss_kwargs.get('nf',25)
        stride = Loss_kwargs.get('stride',1)
        online_construct = Loss_kwargs.get('online_construct',False)
        na_right = self.na_right if hasattr(self,'na_right') else 0
        nb_right = self.nb_right if hasattr(self,'nb_right') else 0
        return sys_data.to_hist_future_data(na=self.na,nb=self.nb,nf=nf,na_right=na_right, nb_right=nb_right, \
            stride=stride,online_construct=online_construct) #uhist, yhist, ufuture, yfuture

    def init_nets(self, nu, ny): # a bit weird
        na_right = self.na_right if hasattr(self,'na_right') else 0
        nb_right = self.nb_right if hasattr(self,'nb_right') else 0
        self.encoder = self.e_net(nb=(self.nb+nb_right), nu=nu, na=(self.na+na_right), ny=ny, nx=self.nx, **self.e_net_kwargs)
        self.fn =      self.f_net(nx=self.nx, nu=nu,                                **self.f_net_kwargs)
        if self.feedthrough:
            self.hn =      self.h_net(nx=self.nx, ny=ny, nu=nu,                     **self.h_net_kwargs) 
        else:
            self.hn =      self.h_net(nx=self.nx, ny=ny,                            **self.h_net_kwargs) 

    def loss_old(self, uhist, yhist, ufuture, yfuture, loss_nf_cutoff=None, **Loss_kwargs):
        x = self.encoder(uhist, yhist) #initialize Nbatch number of states
        errors = []
        for y, u in zip(torch.transpose(yfuture,0,1), torch.transpose(ufuture,0,1)): #iterate over time
            error = nn.functional.mse_loss(y, self.hn(x,u) if self.feedthrough else self.hn(x))
            errors.append(error) #calculate error after taking n-steps
            if loss_nf_cutoff is not None and error.item()>loss_nf_cutoff:
                print(len(errors), end=' ')
                break
            x = self.fn(x,u) #advance state. 
        return torch.mean(torch.stack(errors))

    def loss(self, uhist, yhist, ufuture, yfuture, loss_nf_cutoff=None, **Loss_kwargs):
        assert loss_nf_cutoff==None
        x = self.encoder(uhist, yhist) #initialize Nbatch number of states
        X = []
        for y, u in zip(torch.transpose(yfuture,0,1), torch.transpose(ufuture,0,1)): #iterate over time
            X.append(x)
            x = self.fn(x,u)
        X = torch.stack(X,dim=1) #(Nbatch, nf, nx)

        if self.feedthrough:
            raise NotImplementedError
        else:
            yhat = self.hn(X.reshape(-1, X.shape[-1])) #image (Nbatch, nf, ...)
        yhat = yhat.view(*((uhist.shape[0], ufuture.shape[1]) + yhat.shape[1:]))
        loss = nn.functional.mse_loss(yfuture, yhat)
        return loss


    ########## How to use ##############
    def init_state_multi(self, sys_data, nf=100, stride=1):
        na_right = self.na_right if hasattr(self,'na_right') else 0
        nb_right = self.nb_right if hasattr(self,'nb_right') else 0
        uhist, yhist = sys_data.to_hist_future_data(na=self.na, nb=self.nb, nf=nf, na_right=na_right, nb_right=nb_right, stride=stride)[:2] #(1,)
        uhist = torch.tensor(uhist,dtype=torch.float32)
        yhist = torch.tensor(yhist,dtype=torch.float32)
        with torch.no_grad():
            self.state = self.encoder(uhist,yhist)
        return max(self.na,self.nb)

    def reset_state(self): #to be able to use encoder network as a data generator
        self.state = torch.zeros(1,self.nx)

    def measure_act_multi(self,action): #action is already normalized
        action = torch.tensor(np.array(action,dtype=np.float32), dtype=torch.float32) #(N,...)
        with torch.no_grad():
            feedthrough = self.feedthrough if hasattr(self,'feedthrough') else False
            y_predict = self.hn(self.state, action).numpy() if feedthrough else self.hn(self.state).numpy()
            self.state = self.fn(self.state, action)
        return y_predict

    def get_state(self):
        return self.state[0].numpy()

    def psi(self, upast, ypast, tensor_return='automatic'):
        #upast = (Nbatch, nb, nu) or (nb, nu) or (Nbatch, nb) if nu==None
        #ypast = (Nbatch, na, ny) or (na, ny) or (Nbatch, na) if ny==None
        if tensor_return=='automatic':
            tensor_return = isinstance(upast,torch.Tensor)

        upast, ypast = to_torch_if_needed(upast, ypast)

        upast, batched_u = check_shapes(upast, (self.nb,) + self.nu_tuple)
        ypast, batched_y = check_shapes(ypast, (self.na,) + self.ny_tuple)
        assert batched_u==batched_y, f'incorrect input shapes upast.shape={upast}, ypast.shape={ypast.shape}'
        
        # len(upast.shape) == 2 + len(self.nu_tuple)
        # batched_y = len(ypast.shape) == 2 + len(self.ny_tuple)
        # assert batched_u==batched_y, f'incorrect input shapes upast.shape={upast}, ypast.shape={ypast.shape}'
        # if batched_u==False:
        #     upast, ypast = upast[None], ypast[None]

        # B, nb, *nu_given = upast.shape
        # assert nb==self.nb, f'incorrect shape of upast, upast.shape={upast.shape}'
        # assert tuple(nu_given)==self.nu_tuple, f'incorrect shape of upast, upast.shape={upast.shape}'
        # B, na, *ny_given = ypast.shape
        # assert na==self.na, f'incorrect shape of ypast, ypast.shape={ypast.shape}'
        # assert tuple(ny_given)==self.ny_tuple, f'incorrect shape of ypast, ypast.shape={ypast.shape}'

        upast, ypast = self.norm.u_transform(upast), self.norm.y_transform(ypast)
        xinit = self.encoder(upast, ypast)
        if batched_u==False:
            xinit = xinit[0]
        return xinit if tensor_return else xinit.detach().numpy()

    def f(self, x, u, tensor_return='automatic'):
        if tensor_return=='automatic':
            tensor_return = isinstance(x,torch.Tensor)
        x, u = to_torch_if_needed(x, u)

        x, batched_x = check_shapes(x, (self.nx,))
        u, batched_u = check_shapes(u, self.nu_tuple)
        assert batched_u==batched_x

        u = self.norm.u_transform(u) #it should not change the floating point

        xnext = self.fn(x,u)
        if batched_x==False:
            xnext = xnext[0]
        return xnext if tensor_return else xnext.detach().numpy()

    def h(self, x, u=None, tensor_return='automatic'):
        if tensor_return=='automatic':
            tensor_return = isinstance(x,torch.Tensor)
        x = to_torch_if_needed(x)
        if self.feedthrough:
            assert u is not None, 'feedthrough is enabled thus an input need to be given for a output prediction'
            u = to_torch_if_needed(u)
            u = self.norm.u_transform(u) #it should not change the floating point
        x, batched_x = check_shapes(x,(self.nx,)) #(Nbatch, nx)
        if self.feedthrough:
            u, batched_u = check_shapes(u,self.nu_tuple)
            assert batched_x==batched_u, f'incorrect input shapes x.shape={x.shape}, u.shape={u.shape}'

        y = self.hn(x) if not self.feedthrough else self.hn(x,u)
        y = self.norm.y_inv_transform(y)
        if batched_x==False:
            y = y[0]
        return y if tensor_return else y.detach().numpy()

    def ONNX_export(self, batched = False, filename='SUBNET'):
        R = lambda *shape: torch.randn(*shape) if len(shape)>0 else torch.randn(1)[0]

        B = (1,) if batched else ()
        upast = R(*(B + (self.nb,) + self.nu_tuple))
        ypast = R(*(B + (self.na,) + self.ny_tuple))
        x = R(*(B + (self.nx,)))
        u = R(*(B + self.nu_tuple))
        y = R(*(B + self.ny_tuple))

        ONNX_export(self.f, 'fn', f'{filename}-f.ONNX', output_names='xnext', batched=batched, x=x, u=u)
        hinputs = {'x':x} if self.feedthrough==False else {'x':x,'u':u}
        ONNX_export(self.h, 'hn', f'{filename}-h.ONNX', output_names='y', batched=batched, **hinputs)
        ONNX_export(self.psi, 'encoder', f'{filename}-psi.ONNX', output_names='xinit', batched=batched, upast=upast, ypast=ypast)



############## Continuous time ##################
from deepSI.utils import integrator_RK4, integrator_euler
class SS_encoder_deriv_general(SS_encoder_general):
    '''The subspace encoder method to obtain continuous time models https://arxiv.org/abs/2204.09405

    Equations
    x_t|t = encoder(upast, ypast) 
    dxdt = 1/tau * derivn(x_t+k|t, u_t) = f_norm * derivn(x_t+k|t, u_t) #derivn is f_net on initalization
    x_t+k+1|t = fn(x_t+k|t, u_t) #integrator which uses derivn
    y_t+k|t = hn(x_t+k|t)

    Here the networks are initialized as:
    encoder = e_net(nb, nu, na, ny, nx, **e_net_kwargs)
    derivn = f_net(nx, nu, **f_net_kwargs)
    fn = integrator_net(deriv, f_norm, **integrator_net_kwargs) #normalization is part of the integrator
    hn = h_net(nx, ny, nu=-1, **h_net_kwargs)

    Furthermore you can use cut_off to stabelize the training. 
    tip: use cut_off = 5*(normalized noise level) is a good value
    '''
    def __init__(self, nx=10, na=20, nb=20, feedthrough=False, f_norm=None, tau=None, cut_off=float('inf'), \
                 e_net=default_encoder_net, f_net=default_state_net, integrator_net=integrator_RK4, h_net=default_output_net, \
                 e_net_kwargs={},           f_net_kwargs={},         integrator_net_kwargs={},       h_net_kwargs={},\
                 na_right=0, nb_right=0):
        '''The subspace encoder method to obtain continuous time models https://arxiv.org/abs/2204.09405

        Equations
        x_t|t = encoder(upast, ypast) 
        dxdt = 1/tau * derivn(x_t+k|t, u_t) = f_norm * derivn(x_t+k|t, u_t) #derivn is f_net on initalization
        x_t+k+1|t = fn(x_t+k|t, u_t) #integrator which uses derivn
        y_t+k|t = hn(x_t+k|t)

        Here the networks are initialized as:
        encoder = e_net(nb, nu, na, ny, nx, **e_net_kwargs)
        derivn = f_net(nx, nu, **f_net_kwargs)
        fn = integrator_net(deriv, f_norm, **integrator_net_kwargs) #normalization is part of the integrator
        hn = h_net(nx, ny, nu=-1, **h_net_kwargs)

        Furthermore you can use cut_off to stabelize the training. 
        tip: use cut_off = 5*(normalized noise level) is a good value
        '''
        super(SS_encoder_deriv_general, self).__init__(nx=nx, na=na, nb=nb, feedthrough=feedthrough, e_net=e_net, f_net=f_net, h_net=h_net, \
                                                       e_net_kwargs=e_net_kwargs, f_net_kwargs=f_net_kwargs, h_net_kwargs=h_net_kwargs, \
                                                       na_right=na_right, nb_right=nb_right)
        self.integrator_net = integrator_net
        self.integrator_net_kwargs = integrator_net_kwargs
        assert f_norm!=None or tau!=None
        self.f_norm = f_norm if tau==None else 1/tau
        self.cut_off = cut_off

    def init_nets(self, nu, ny): # a bit weird
        par = super(SS_encoder_deriv_general, self).init_nets(nu,ny) 
        self.derivn = self.fn  #move fn to become the derivative net
        self.excluded_nets_from_parameters = ['derivn']
        self.fn = self.integrator_net(self.derivn, f_norm=self.f_norm, **self.integrator_net_kwargs) #has no torch parameters?

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
        for i,(u,y) in enumerate(zip(torch.transpose(ufuture,0,1), torch.transpose(yfuture,0,1))): #iterate over time
            yhat = self.hn(x) if not self.feedthrough else self.hn(x,u)
            dy = (yhat - y)**2 # (Nbatch, ny)
            diff.append(dy)
            with torch.no_grad(): #break if the error is too large
                if torch.mean(dy).item()**0.5>self.cut_off:
                    break
            x = self.fn(x,u)
        return torch.mean((torch.stack(diff,dim=1)))
    def f(self, x, u, tensor_return='automatic'):
        if tensor_return=='automatic':
            tensor_return = isinstance(x,torch.Tensor)
        x, u = to_torch_if_needed(x, u)

        x, batched_x = check_shapes(x, (self.nx,))
        u, batched_u = check_shapes(u, self.nu_tuple)
        assert batched_u==batched_x

        u = self.norm.u_transform(u) #it should not change the floating point

        xdot = self.derivn(x,u)*self.f_norm
        if batched_x==False:
            xdot = xdot[0]
        return xdot if tensor_return else xdot.detach().numpy()

class hf_net_default(nn.Module):
    def __init__(self, nx, nu, ny, feedthrough=False, f_net=default_state_net, f_net_kwargs={}, h_net_kwargs={}, h_net=default_output_net):
        super(hf_net_default, self).__init__()
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.fn = f_net(nx, nu, **f_net_kwargs)
        #nx, ny, nu=-1,
        self.feedthrough = feedthrough
        self.hn = h_net(nx, ny, nu=nu if feedthrough else -1, **h_net_kwargs)
    
    def forward(self, x, u):
        y = self.hn(x,u) if self.feedthrough else self.hn(x)
        xnext = self.fn(x,u)
        return y, xnext

class SS_encoder_general_hf(SS_encoder_general):
    """The encoder function with combined h and f functions
    
    the hf_net_default has the arguments
       hf_net_default(nx, nu, ny, feedthrough=False, **hf_net_kwargs)
    and is used as 
       ynow, xnext = hfn(x,u)
    """
    def __init__(self, nx=10, na=20, nb=20, feedthrough=False, \
                 hf_net=hf_net_default, \
                 hf_net_kwargs = dict(f_net=default_state_net, f_net_kwargs={}, h_net_kwargs={}, h_net=default_output_net), \
                 e_net=default_encoder_net,   e_net_kwargs={}, na_right=0, nb_right=0):

        super(SS_encoder_general_hf, self).__init__(nx=nx, nb=nb, na=na, na_right=na_right, nb_right=nb_right)
        
        self.e_net = e_net
        self.e_net_kwargs = e_net_kwargs
        
        self.hf_net = hf_net
        hf_net_kwargs['feedthrough'] = feedthrough
        self.hf_net_kwargs = hf_net_kwargs

        self.feedthrough = feedthrough

    def init_nets(self, nu, ny): # a bit weird
        na_right = self.na_right if hasattr(self,'na_right') else 0
        nb_right = self.nb_right if hasattr(self,'nb_right') else 0
        self.encoder = self.e_net(nb=self.nb+nb_right, nu=nu, na=self.na+na_right, ny=ny, nx=self.nx, **self.e_net_kwargs)
        self.hfn = self.hf_net(nx=self.nx, nu=self.nu, ny=self.ny, **self.hf_net_kwargs)

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(uhist, yhist) #initialize Nbatch number of states
        errors = []
        for y, u in zip(torch.transpose(yfuture,0,1), torch.transpose(ufuture,0,1)): #iterate over time
            yhat, x = self.hfn(x, u)
            errors.append(nn.functional.mse_loss(y, yhat)) #calculate error after taking n-steps
        return torch.mean(torch.stack(errors))
    
    def measure_act_multi(self,actions):
        actions = torch.tensor(np.array(actions), dtype=torch.float32) #(N,...)
        with torch.no_grad():
            y_predict, self.state = self.hfn(self.state, actions)
        return y_predict.numpy()

    def hf(self, x, u):
        if tensor_return=='automatic':
            tensor_return = isinstance(x,torch.Tensor)
        x, u = to_torch_if_needed(x, u)

        x, batched_x = check_shapes(x, (self.nx,))
        u, batched_u = check_shapes(u, self.nu_tuple)
        assert batched_u==batched_x

        u = self.norm.u_transform(u) #it should not change the floating point

        yhat, xnext = self.hfn(x,u)
        if batched_x==False:
            xnext = xnext[0]
            yhat = yhat[0]
        return (yhat, xnext) if tensor_return else (yhat.detach().numpy(), xnext.detach().numpy())


    def ONNX_export(self, batched = False, filename='SUBNET'):
        R = lambda *shape: torch.randn(*shape) if len(shape)>0 else torch.randn(1)[0]

        B = (1,) if batched else ()
        upast = R(*(B + (self.nb,) + self.nu_tuple))
        ypast = R(*(B + (self.na,) + self.ny_tuple))
        x = R(*(B + (self.nx,)))
        u = R(*(B + self.nu_tuple))
        y = R(*(B + self.ny_tuple))
        torch.cat()

        ONNX_export(self.hf, 'hfn', f'{filename}-hfn.ONNX', output_names='xnext', batched=batched, x=x, u=u)
        ONNX_export(self.psi, 'encoder', f'{filename}-psi.ONNX', output_names='xinit', batched=batched, upast=upast, ypast=ypast)


class par_start_encoder(nn.Module):
    """A network which makes the initial states a parameter of the network"""
    def __init__(self, nx, nsamples):
        super(par_start_encoder, self).__init__()
        self.start_state = nn.parameter.Parameter(data=torch.as_tensor(np.random.normal(scale=0.1,size=(nsamples,nx)),dtype=torch.float32))

    def forward(self,ids):
        return self.start_state[ids]

class SS_par_start(SS_encoder): #this is not implemented in a nice manner, there might be bugs.
    """docstring for SS_par_start"""
    def __init__(self, nx=10, feedthrough=False, optimizer_kwargs={}):
        super(SS_par_start, self).__init__(nx=nx,na=0, nb=0, feedthrough=feedthrough)
        self.optimizer_kwargs = optimizer_kwargs

    ########## How to fit #############
    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        nf = Loss_kwargs.get('nf',25)
        stride = Loss_kwargs.get('stride',1)
        online_construct = Loss_kwargs.get('online_construct',False)
        assert online_construct==False, 'to be implemented'
        hist, ufuture, yfuture = sys_data.to_encoder_data(na=0,nb=0,nf=nf,stride=stride,force_multi_u=True,force_multi_y=True)
        nsamples = hist.shape[0]
        ids = np.arange(nsamples,dtype=int)
        self.par_starter = par_start_encoder(nx=self.nx, nsamples=nsamples)
        self.optimizer = self.init_optimizer(self.parameters, **self.optimizer_kwargs) #no kwargs
        return ids, ufuture, yfuture #returns np.array(hist),np.array(ufuture),np.array(yfuture)

    def init_nets(self, nu, ny): # a bit weird
        ny = ny if ny is not None else 1
        nu = nu if nu is not None else 1
        self.fn =      self.f_net(self.nx+nu,            self.nx, n_nodes_per_layer=self.f_n_nodes_per_layer, n_hidden_layers=self.f_n_hidden_layers, activation=self.f_activation)
        hn_in = self.nx + nu if self.feedthrough else self.nx
        self.hn =      self.h_net(hn_in     ,            ny,      n_nodes_per_layer=self.h_n_nodes_per_layer, n_hidden_layers=self.h_n_hidden_layers, activation=self.h_activation)

    def loss(self, ids, ufuture, yfuture, **Loss_kwargs):
        ids = ids.numpy().astype(int)
        #hist is empty
        x = self.par_starter(ids)
        y_predict = []
        for u in torch.transpose(ufuture,0,1):
            xu = torch.cat((x,u),dim=1)
            y_predict.append(self.hn(x) if not self.feedthrough else self.hn(xu))
            x = self.fn(xu)
        return torch.mean((torch.stack(y_predict,dim=1)-yfuture)**2)

    ########## How to use ##############
    def init_state_multi(self,sys_data,nf=100,stride=1):
        hist = torch.tensor(sys_data.to_encoder_data(na=0,nb=0,nf=nf,stride=stride)[0],dtype=torch.float32) #(1,)
        with torch.no_grad():
            self.state = torch.as_tensor(np.random.normal(scale=0.1,size=(hist.shape[0],self.nx)),dtype=torch.float32) #detach here?
        return 0

from deepSI.utils import simple_res_net, feed_forward_nn, general_koopman_forward_layer
class SS_encoder_general_koopman(SS_encoder_general):
    """
    The encoder setup with a linear transition function with an affine input (kinda like an LPV), in equations
        x_k = e(u_kpast,y_kpast) 
        x_k+1 = A@x_k + g(x_k,u_k)@u_k          #affine input here (@=matrix multiply)
        y_k = h(x_k)
    Where g is given by g_net which is by default a feedforward nn with residual (i.e. simple_res_net) called as 
        'g_net(n_in=affine_dim,n_out=output_dim*input_dim,**g_net_kwargs)'
    with affine_dim=nx + nu, output_dim = nx, input_dim=nu
    Hence, g_net produces a vector which is reshaped into a matrix \
    (See deepSI.utils.torch_nets.general_koopman_forward_layer for details).
    """
    def __init__(self, nx=10, na=20, nb=20, feedthrough=False, include_u_in_g=True, 
                e_net=default_encoder_net, g_net=simple_res_net, h_net=default_output_net, \
                    e_net_kwargs={}, g_net_kwargs={}, h_net_kwargs={}):
        f_net_kwargs = dict(include_u_in_g=include_u_in_g,g_net=g_net,g_net_kwargs=g_net_kwargs)
        
        super(SS_encoder_general_koopman, self).__init__(nx=nx,na=na,nb=nb, feedthrough=feedthrough, \
            e_net=e_net,               f_net=general_koopman_forward_layer, h_net=h_net, \
            e_net_kwargs=e_net_kwargs, f_net_kwargs=f_net_kwargs,           h_net_kwargs=h_net_kwargs)


from deepSI.utils import CNN_chained_upscales, CNN_encoder
class SS_encoder_CNN_video_input(SS_encoder_general):
    """The subspace encoder convolutonal neural network 

    Notes
    -----
    The subspace encoder

    """
    def __init__(self, nx=10, na=20, nb=20, feedthrough=True, e_net=CNN_encoder, f_net=CNN_encoder, h_net=CNN_encoder, \
                                            e_net_kwargs={}, f_net_kwargs={}, h_net_kwargs={}):
        super(SS_encoder_CNN_video_input, self).__init__(nx=nx,na=na,nb=nb, feedthrough=feedthrough, \
            e_net=e_net,               f_net=f_net,                h_net=h_net, \
            e_net_kwargs=e_net_kwargs, f_net_kwargs=f_net_kwargs,  h_net_kwargs=h_net_kwargs)

from deepSI.utils import CNN_chained_upscales, CNN_encoder
class SS_encoder_CNN_video(SS_encoder_general):
    """The subspace encoder convolutonal neural network 

    Notes
    -----
    The subspace encoder

    """
    def __init__(self, nx=10, na=20, nb=20, feedthrough=False, e_net=CNN_encoder, f_net=default_state_net, h_net=CNN_chained_upscales, \
                                            e_net_kwargs={}, f_net_kwargs={}, h_net_kwargs={}):
        super(SS_encoder_CNN_video, self).__init__(nx=nx,na=na,nb=nb, feedthrough=feedthrough, \
            e_net=e_net,               f_net=f_net,                h_net=h_net, \
            e_net_kwargs=e_net_kwargs, f_net_kwargs=f_net_kwargs,  h_net_kwargs=h_net_kwargs)


from deepSI.utils import Shotgun_MLP
class SS_encoder_shotgun_MLP(SS_encoder_general): #this is basicly a Neural radience field thing
    def __init__(self, nx=10, na=20, nb=20, e_net=CNN_encoder, f_net=default_state_net, h_net=Shotgun_MLP, \
                            e_net_kwargs={}, f_net_kwargs={}, h_net_kwargs={}):
        '''Todo: fix cuda with all the arrays'''
        raise NotImplementedError('not yet updated to 0.3 go back to 0.2 to use this model')
        super(SS_encoder_shotgun_MLP, self).__init__(nx=nx,na=na,nb=nb,\
            e_net=e_net,               f_net=f_net,                h_net=h_net, \
            e_net_kwargs=e_net_kwargs, f_net_kwargs=f_net_kwargs,  h_net_kwargs=h_net_kwargs)
        self.encoder_time = 0
        self.forward_time = 0
    
    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        # I can pre-sample it or sample it when passed. Which one is faster?
        # I'm doing it here for now, maybe later I will do it in the dataset on a shuffle or something.
        if len(self.ny)==3:
            C,H,W = self.ny
        else:
            H,W = self.ny
            C = None
        Nb = uhist.shape[0]
        Nsamp = Loss_kwargs.get('Nsamp',100) #int(800/Nb) is approx the best for speed for CPU
        batchselector = torch.broadcast_to(torch.arange(Nb)[:,None],(Nb,Nsamp))
        time.time()

        t_start = time.time()
        x = self.encoder(uhist, yhist)
        self.encoder_time += time.time() - t_start

        t_start = time.time()
        mse_losses = []
        for y, u in zip(torch.transpose(yfuture,0,1), torch.transpose(ufuture,0,1)): #iterate over time
            h = torch.randint(low=0, high=H, size=(Nb,Nsamp))
            w = torch.randint(low=0, high=W, size=(Nb,Nsamp))
            ysamps = y[batchselector,:,h,w] if C!=None else y[batchselector,h,w]
            yhat = self.hn.sampler(x, h, w)
            mse_losses.append(nn.functional.mse_loss(yhat, ysamps))
            x = self.fn(x,u)
        self.forward_time += time.time() - t_start
        return torch.mean(torch.stack(mse_losses))

    def apply_experiment(self, sys_data, save_state=False): #can put this in apply controller
        '''Does an experiment with for a given system data (fixed u)

        Parameters
        ----------
        sys_data : System_data or System_data_list (or list or tuple)
            The experiment which should be applied

        Notes
        -----
        This will initialize the state using self.init_state if sys_data.y (and u)
        is not None and skip the appropriate number of steps associated with it.
        If either is missing than self.reset_state() is used to initialize the state. 
        Afterwards this state is advanced using sys_data.u and the output is saved at each step.
        Lastly, the number of skipped/copied steps in init_state is saved as sys_data.cheat_n such 
        that it can be accounted for later.
        '''
        if isinstance(sys_data,(tuple,list,System_data_list)):
            return System_data_list([self.apply_experiment(sd, save_state=save_state) for sd in sys_data])
        #check if sys_data.x holds the 
        #u = (Ns)
        #x = (Ns, C, H, W) or (Ns, H, W)
        #y = (Ns, Np, C), (Ns, Np, C)
        #h = (Ns, Np)
        #w = (Ns, Np)
        if not (hasattr(sys_data,'h') and hasattr(sys_data,'w')):
            return super(SS_encoder_shotgun_MLP, self).apply_experiment(sys_data, save_state=save_state)
        h, w = sys_data.h, sys_data.w

        Y = []
        sys_data.x, sys_data.y = sys_data.y, sys_data.x #move image to y
        sys_data_norm = self.norm.transform(sys_data) #transform image if needed
        sys_data.x, sys_data.y = sys_data.y, sys_data.x #move image back to x
        sys_data_norm.x, sys_data_norm.y = sys_data_norm.y, sys_data_norm.x #move image back to x
        sys_data_norm.h, sys_data_norm.w = h, w #set h and w on the normed version

        U = sys_data_norm.u #get the input
        Images = sys_data_norm.x #get the images

        assert sys_data_norm.y is not None, 'not implemented' #if y is not None than init state
        obs, k0 = self.init_state(sys_data_norm) #normed obs in the shape of y in the last step. 
        Y.extend(sys_data_norm.y[:k0]) #h(x_{k0-1})

        if save_state:
            X = [self.get_state()]*(k0+1)

        for k in range(k0,len(U)):
            Y.append(obs)
            if k < len(U)-1: #skip last step
                obs = self.step(U[k], h=h[k+1], w=w[k+1])
                if save_state:
                    X.append(self.get_state())

        #how the norm? the Y need to be transformed from (Ns, Np, C) with the norm
        #norm.y0 has shape (C, W, H) or (C, 1, 1) or similar
        Y = np.array(Y) #(Ns, Np, C)
        # if self.norm.y0 is 1:
            # return System_data(u=sys_data.u, y=Y, x=np.array(X) if save_state else None,normed=False,cheat_n=k0)
        #has the shape of a constant or (1, 1) or (C, 1, 1) the possiblity of (C, H, W) I will exclude for now. 
        from copy import deepcopy
        norm_sampler = deepcopy(self.norm)
        if isinstance(self.norm.y0,(int,float)):
            pass
        elif self.norm.y0.shape==(1,1):
            norm_sampler.y0 = norm_sampler.y0[0,0]
            norm_sampler.ystd = norm_sampler.ystd[0,0] #ystd to a float
        elif self.norm.y0.shape==(sys_data.x.shape[0],1,1):
            norm_sampler.y0 = norm_sampler.y0[:,0,0]
            norm_sampler.ystd = norm_sampler.ystd[:,0,0] #ystd to (C,) such that it can divide #(Ns, Np, C)
        else:
            raise NotImplementedError(f'norm of {self.norm} is not yet implemented for sampled simulations')
        sys_data_sim =  norm_sampler.inverse_transform(System_data(u=np.array(U),y=np.array(Y),x=np.array(X) if save_state else None,normed=True,cheat_n=k0))
        sys_data_sim.h, sys_data_sim.w = sys_data.h, sys_data.w
        return sys_data_sim

    def init_state(self, sys_data):
        #sys_data is already normed
        if not hasattr(sys_data,'h'):
            return super(SS_encoder_shotgun_MLP, self).init_state(sys_data)

        sys_data.x, sys_data.y = sys_data.y, sys_data.x #switch image to be y
        uhist, yhist = sys_data[:self.k0].to_hist_future_data(na=self.na,nb=self.nb,nf=0)[:2]
        sys_data.x, sys_data.y = sys_data.y, sys_data.x #switch image to be x

        uhist = torch.tensor(uhist, dtype=torch.float32)
        yhist = torch.tensor(yhist, dtype=torch.float32)
        h,w = torch.as_tensor(sys_data.h[self.k0]), torch.as_tensor(sys_data.w[self.k0]) #needs dtype?
        with torch.no_grad():
            self.state = self.encoder(uhist, yhist)
            # h = (Np)
            # w = (Np)
            # state = (1, nx)
            # sampler(self, x, h, w) goes to (Nb, Nsamp, C)
            y_predict = self.hn.sampler(x=self.state, h=h[None], w=w[None])[0].numpy() #output: (Nb, Nsamp, C) -> (Nsamp, C)
        return y_predict, max(self.na,self.nb)

    def step(self,action, h=None, w=None):
        if h is None:
            return super(SS_encoder_shotgun_MLP, self).step(action)
        action = torch.tensor(action,dtype=torch.float32)[None] #(1,...)
        h,w = torch.as_tensor(h), torch.as_tensor(w) #needs dtype?
        with torch.no_grad():
            self.state = self.fn(self.state,action)
            y_predict = self.hn.sampler(x=self.state, h=h[None], w=w[None])[0].numpy() #output: (Nb, Nsamp, C) -> (Nsamp, C)
        return y_predict

    def sys_data_sampler(self, sys_data, Ndots_per_image):
        u, images = sys_data.u, sys_data.y
        #images has shape (Ns, C, H, W) for (Ns, H, W)
        if len(images.shape)==4:
            Ns, C, W, H = images.shape
        elif len(images.shape)==3:
            Ns, W, H = images.shape
            C = None
        else:
            assert False, 'check images.shape'
        sampleselector = torch.broadcast_to(torch.arange(Ns)[:,None],(Ns,Ndots_per_image))
        h = np.random.randint(low=0, high=H, size=(Ns,Ndots_per_image))
        w = np.random.randint(low=0, high=W, size=(Ns,Ndots_per_image))
        images_shots = images[sampleselector,:,h,w] if C!=None else images[sampleselector,h,w] #what shape does this have? I hope it has (Ns, Nshots, C)
        sys_data = System_data(u=u, y=images_shots, x=images)
        sys_data.h, sys_data.w = h, w
        return sys_data

class nonlin_ino_state_net(nn.Module):
    def __init__(self, nx, nu, ny, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh): 
        super(nonlin_ino_state_net, self).__init__()
        from deepSI.utils import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.nx = nx
        self.net = simple_res_net(n_in=nx+np.prod(self.nu,dtype=int)+np.prod(self.ny,dtype=int), \
                                  n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, \
                                  activation=activation)

    def forward(self, x, u, eps=None):
        if eps==None:
            eps = torch.zeros((u.shape[0],np.prod(self.ny,dtype=int)),dtype=torch.float32)
        net_in = torch.cat([x,u.view(u.shape[0],-1),eps.view(u.shape[0],-1)],axis=1)            
        return self.net(net_in)

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

class SS_encoder_innovation(SS_encoder_general):
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
    def __init__(self, nx=10, na=20, nb=20, na_right=1, nb_right=0, feedthrough=False, \
        e_net=default_encoder_net, f_net=default_ino_state_net, h_net=default_output_net, e_net_kwargs={}, f_net_kwargs={}, h_net_kwargs={}):
        super(SS_encoder_innovation, self).__init__(nx=nx, na=na, nb=nb, na_right=na_right, nb_right=nb_right, \
            feedthrough=False, e_net=e_net, f_net=f_net, h_net=h_net, e_net_kwargs=e_net_kwargs, f_net_kwargs=f_net_kwargs, h_net_kwargs=h_net_kwargs)

    def init_nets(self, nu, ny):
        self.encoder = self.e_net(nb=self.nb+self.nb_right, nu=nu, na=self.na+self.na_right,\
             ny=ny, nx=self.nx, **self.e_net_kwargs)
        self.fn =      self.f_net(nx=self.nx, nu=nu, ny=self.ny,                    **self.f_net_kwargs)
        if self.feedthrough:
            self.hn =      self.h_net(nx=self.nx, ny=ny, nu=nu,                     **self.h_net_kwargs) 
        else:
            self.hn =      self.h_net(nx=self.nx, ny=ny,                            **self.h_net_kwargs) 

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(uhist, yhist)
        y_predict = []
        for u,y in zip(torch.transpose(ufuture,0,1),torch.transpose(yfuture,0,1)): #iterate over time
            yhat = self.hn(x) if not self.feedthrough else self.hn(x,u)
            y_predict.append(yhat) 
            x = self.fn(x,u,eps=y-yhat)
        return torch.mean((torch.stack(y_predict,dim=1)-yfuture)**2)
        
class LPV_with_schedul_net(nn.Module):
    def __init__(self, nx, nu, ny, scheduling_dim, scheduling_network, scheduling_network_dependent_on_current_input=False, F=10):
        super(LPV_with_schedul_net,self).__init__()
        #without the 0.1 it sometimes was unstable? Might need to be adressed in the future
        make_mat = lambda n_in, n_out: 1/F*(torch.rand(n_out,n_in)*2-1)/n_in**0.5 #returns a matrix of size (n_out, n_in) with uniform 
        
        self.nx = nx
        self.nu = nu
        self.ny = ny #=nx if it is the state equation
        self.scheduling_dim = scheduling_dim #np=0 is linear
        nu = 1 if nu==None else nu
        ny = 1 if ny==None else ny
        
        self.A = nn.Parameter(make_mat(nx, ny))
        self.B = nn.Parameter(make_mat(nu, ny))
        
        self.scheduling_network_dependent_on_current_input = scheduling_network_dependent_on_current_input
        self.scheduling_network = scheduling_network
        self.As = nn.Parameter(torch.stack([make_mat(nx, ny) for _ in range(scheduling_dim)]))
        self.Bs = nn.Parameter(torch.stack([make_mat(nu, ny) for _ in range(scheduling_dim)]))
    
    def forward(self, x, u=None, xenc=None):
        if self.nu==None:
            u = u[:,None] #(Nb,1)
        xp = x if xenc is None else xenc
        pin = torch.cat([xp,u],dim=1) if self.scheduling_network_dependent_on_current_input else xp
        pout = self.scheduling_network(pin)
        
        ylin = torch.einsum('ij,bj->bi',self.A,x) + torch.einsum('ij,bj->bi',self.B,u)
        ynonlin = torch.einsum('pij,bp,bj->bi',self.As,pout,x) + torch.einsum('pij,bp,bj->bi',self.Bs,pout,u)
        yout = ylin + ynonlin
        return yout[:,0] if self.ny==None else yout

    def parameters(self): #exclude scheduling_network from the parameters
        return nn.ParameterList([self.A, self.B, self.As, self.Bs])


from deepSI.fit_systems.encoders import default_encoder_net
from deepSI.utils import simple_res_net
class LPV_SUBNET_internally_scheduled(SS_encoder_general):
    def __init__(self, nx=10, na=20, nb=20, scheduling_dim=2,           feedthrough=True,  scheduling_network_dependent_on_current_input=True, \
        e_net=default_encoder_net, f_net=LPV_with_schedul_net,           h_net=LPV_with_schedul_net,   p_net=simple_res_net, \
        e_net_kwargs={},           f_net_kwargs={},         h_net_kwargs={}, p_net_kwargs={}):
        assert feedthrough==True, 'non-feedthrough has not been implemented for this system yet'
        super(LPV_SUBNET_internally_scheduled, self).__init__(nx=nx,na=na,nb=nb, feedthrough=feedthrough, \
            e_net=e_net,f_net=f_net, h_net=h_net, \
            e_net_kwargs=e_net_kwargs, f_net_kwargs=f_net_kwargs, \
            h_net_kwargs=h_net_kwargs)
        self.p_net = p_net
        self.p_net_kwargs = p_net_kwargs
        self.scheduling_network_dependent_on_current_input = scheduling_network_dependent_on_current_input
        self.scheduling_dim = scheduling_dim
    
    def init_nets(self, nu, ny):
        nuval = 1 if nu==None else nu
        nyval = 1 if ny==None else ny
        self.scheduling_network = self.p_net(n_in=self.nx + (nuval if self.scheduling_network_dependent_on_current_input else 0), n_out = self.scheduling_dim, **self.p_net_kwargs)
        self.encoder = self.e_net(nb=self.nb, nu=nu, na=self.na, ny=ny, nx=self.nx, **self.e_net_kwargs)
        #nx, nu, ny, np, scheduling_network
        self.fn = self.f_net(nx=self.nx, nu=nu, ny=self.nx, scheduling_dim=self.scheduling_dim, scheduling_network=self.scheduling_network, scheduling_network_dependent_on_current_input=self.scheduling_network_dependent_on_current_input, **self.f_net_kwargs)
        self.hn = self.h_net(nx=self.nx, nu=nu, ny=ny,      scheduling_dim=self.scheduling_dim, scheduling_network=self.scheduling_network, scheduling_network_dependent_on_current_input=self.scheduling_network_dependent_on_current_input, **self.h_net_kwargs)

class LPV_SUBNET_externally_scheduled(LPV_SUBNET_internally_scheduled):
    def __init__(self, nx=10, na=20, nb=20, scheduling_dim=2, feedthrough=True,  scheduling_network_dependent_on_current_input=True, use_predicted_output_for_encoder=False, \
        e_net=default_encoder_net, f_net=LPV_with_schedul_net,  h_net=LPV_with_schedul_net, p_net=simple_res_net, \
        e_net_kwargs={},           f_net_kwargs={},         h_net_kwargs={}, p_net_kwargs={}):
        
        super(LPV_SUBNET_externally_scheduled, self).__init__(nx=nx, na=na, nb=nb, scheduling_dim=scheduling_dim, feedthrough=feedthrough,  scheduling_network_dependent_on_current_input=scheduling_network_dependent_on_current_input, \
        e_net=e_net, f_net=f_net,           h_net=h_net,   p_net=p_net, \
        e_net_kwargs=e_net_kwargs,           f_net_kwargs=f_net_kwargs,         h_net_kwargs=h_net_kwargs, p_net_kwargs=p_net_kwargs)
        
        self.use_predicted_output_for_encoder = use_predicted_output_for_encoder
    
    def loss(self, uhist, yhist, ufuture, yfuture, loss_nf_cutoff=None, **Loss_kwargs):
        x = self.encoder(uhist, yhist) #initialize Nbatch number of states
        errors = []
        for y, u in zip(torch.transpose(yfuture,0,1), torch.transpose(ufuture,0,1)): #iterate over time
            xenc = self.encoder(uhist, yhist)
            yhat = self.hn(x=x,u=u,xenc=xenc) if self.feedthrough else self.hn(x,u=None,xenc=xenc)
            error = nn.functional.mse_loss(y, yhat)
            errors.append(error) #calculate error after taking n-steps
            if loss_nf_cutoff is not None and error.item()>loss_nf_cutoff:
                print(len(errors), end=' ')
                break
            
            #uhist = (Nb, nb, nu)
            #yhist = (Nb, na, ny)
            uhist = torch.cat([uhist[:,1:],u[:,None]],dim=1)
            if self.use_predicted_output_for_encoder:
                yhist = torch.cat([yhist[:,1:],yhat[:,None]],dim=1)
            else:
                yhist = torch.cat([yhist[:,1:],y[:,None]],dim=1)
            
            x = self.fn(x, u, xenc=xenc) 
        return torch.mean(torch.stack(errors))
    
    def apply_experiment(self, data, use_predicted_output_for_encoder=False, verbose=True):
        if verbose:
            print(f'INFO: applying the input the LPV SUBNET model with {use_predicted_output_for_encoder=}')
        if use_predicted_output_for_encoder:
            return super().apply_experiment(data)
        else:
            return self.apply_experiment_with_measured_outputs_in_encoder(data)
    
    def apply_experiment_with_measured_outputs_in_encoder(self, sys_data): 
        if isinstance(sys_data, System_data_list):
            return System_data_list([self.apply_experiment_with_measured_outputs_in_encoder(s) for s in sys_data])
        
        sys_data_norm = self.norm.transform(sys_data)
        
        A = sys_data_norm.to_hist_future_data(na=self.na, nb=self.nb, nf=len(sys_data)-self.k0)
        uhist, yhist, ufuture, yfuture = [torch.as_tensor(a,dtype=torch.float32) for a in A]
        x = self.encoder(uhist, yhist) #initialize Nbatch number of states
        yhats = []
        for y, u in zip(torch.transpose(yfuture,0,1), torch.transpose(ufuture,0,1)): #iterate over time
            xenc = self.encoder(uhist, yhist)
            yhat = self.hn(x=x,u=u,xenc=xenc) if self.feedthrough else self.hn(x,u=None,xenc=xenc)
            yhats.append(yhat.detach().numpy()[0]) #calculate error after taking n-steps
            
            uhist = torch.cat([uhist[:,1:],u[:,None]],dim=1)
            yhist = torch.cat([yhist[:,1:],y[:,None]],dim=1)
            x = self.fn(x,u, xenc=xenc) #advance state. 
        yhats = np.concatenate([sys_data_norm.y[:self.k0], np.array(yhats)],axis=0)
        return self.norm.inverse_transform(System_data(u=sys_data_norm.u, y=yhats,cheat_n=self.k0,normed=True))


if __name__ == '__main__':
    sys = SS_encoder_general()
    # from deepSI.datasets.sista_database import powerplant
    # from deepSI.datasets import Silverbox
    from deepSI.datasets import Cascaded_Tanks
    
    train, test = Cascaded_Tanks()#powerplant()
    # train.dt = 0.1
    # test.dt = 0.1
    # train, test = train[:150], test[:50]
    # print(train, test)
    sys.fit(train,test,epochs=1)
    test2 = sys.apply_experiment(test)
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
    # sys = SS_encoder_deriv_general(nx=2,f_norm=0.025)
    # sys = SS_par_start()
    # sys.fit(train, sim_val=test, epochs=50, batch_size=32, concurrent_val=True)