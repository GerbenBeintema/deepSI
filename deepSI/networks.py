import torch
from torch.nn import Sequential
from torch import nn

###########################################################
#### Multi layer peceptron/feed forward neural network ####
###########################################################

class MLP_res_net(nn.Module):
    '''Multi-Layer Perceptron with Residual Connection (MLP_res_net) as follows:
              y_pred = net(input) = net_MLP(input) + A * input
              where net_MLP(input) is a simple Multi-Layer Perceptron, e.g.:
                h_1 = input
                h-2 = activation(A_1 h_1 + b_1) #A_1.shape = n_hidden_nodes x input_size
                h_3 = activation(A_2 h_2 + b_2) #A_2.shape = n_hidden_nodes x n_hidden_nodes
                ...
                h_n_hidden_layers = activation(A_n-1 h_n-1 + b_n-1)
                return h_n_hidden_layers
    '''
    def __init__(self, input_size: str | int | list, output_size: str | int | list, n_hidden_layers = 2, n_hidden_nodes = 64, \
                 activation=nn.Tanh, zero_bias=True):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()
        self.scalar_output = output_size=='scalar'
        #convert input shape:
        def to_num(s):
            if isinstance(s, int):
                return s
            if s=='scalar':
                return 1
            a = 1
            for si in s:
                a = a*(1 if si=='scalar' else si)
            return a
        if isinstance(input_size, list):
            input_size = sum(to_num(s) for s in input_size)
        
        output_size = 1 if self.scalar_output else output_size
        self.net_res = nn.Linear(input_size, output_size)

        seq = [nn.Linear(input_size,n_hidden_nodes),activation()]
        for i in range(n_hidden_layers-1):
            seq.append(nn.Linear(n_hidden_nodes,n_hidden_nodes))
            seq.append(activation())
        seq.append(nn.Linear(n_hidden_nodes,output_size))
        self.net_nonlin = nn.Sequential(*seq)

        if zero_bias:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, val=0) #bias
        
    def forward(self, *ars):
        if len(ars)==1:
            net_in = ars[0]
            net_in = net_in.view(net_in.shape[0], -1) #adds a dim when needed
        else:
            net_in = torch.cat([a.view(a.shape[0], -1) for a in ars],dim=1) #flattens everything
        out = self.net_nonlin(net_in) + self.net_res(net_in)
        return out[:,0] if self.scalar_output else out
    
###########################
###### Integrators ########
###########################

def euler_integrator(f, x, u, dt, n_steps=1):
    dtp = (dt/n_steps)[:,None]
    for _ in range(n_steps): #f(x,u) has shape (nbatch, nx)
        x = x + f(x,u)*dtp
    return x

def rk4_integrator(f, x, u, dt, n_steps=1):
    dtp = (dt/n_steps)[:,None]
    for _ in range(n_steps): #f(x,u) has shape (nbatch, nx)
        k1 = dtp * f(x,u)
        k2 = dtp * f(x+k1*0.5,u)
        k3 = dtp * f(x+k2*0.5,u)
        k4 = dtp * f(x+k3,u)
        x = x + (k1+2*k2+2*k3+k4)/6
    return x

def rk45_integrator(f, x, u, dt, n_steps=1):
    dtp = (dt/n_steps)[:,None]
    for _ in range(n_steps): #f(x,u) has shape (nbatch, nx)
        k1 = dtp * f(x, u)
        k2 = dtp * f(x + k1 / 4, u)
        k3 = dtp * f(x + 3 * k1 / 32 + 9 * k2 / 32, u)
        k4 = dtp * f(x + 1932 * k1 / 2197 - 7200 * k2 / 2197 + 7296 * k3 / 2197, u)
        k5 = dtp * f(x + 439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104, u)
        k6 = dtp * f(x - 8 * k1 / 27 + 2 * k2 - 3544 * k3 / 2565 + 1859 * k4 / 4104 - 11 * k5 / 40, u)
        
        x = x + (16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430 - 9 * k5 / 50 + 2 * k6 / 55)
    return x

##################################
##### LPV SUBNET networks ########
##################################

import numpy as np
class Bilinear(nn.Module):
    '''A(p) = A_0 + A_1 p_1 + A_2 p_2 + ... + A_n_schedual p_n_schedual'''
    def __init__(self, n_in, n_out, n_schedual, std_output=None, std_input=None, scale_fac=None):
        super().__init__()
        scale_fac = (n_in*(n_schedual+1))**0.5*10 if scale_fac is None else scale_fac
        self.Alin = nn.Parameter(torch.randn((n_out, n_in))/scale_fac)
        self.Anlin = nn.Parameter(torch.randn((n_schedual, n_out, n_in))/scale_fac)
        self.std_output = torch.as_tensor(std_output,dtype=torch.float32) if std_output is not None else torch.ones((n_out,), dtype=torch.float32)
        assert self.std_output.shape == (n_out,)
        self.std_input = torch.as_tensor(std_input,dtype=torch.float32) if std_input is not None else torch.ones((n_in,), dtype=torch.float32)
        assert self.std_input.shape == (n_in,), f'{self.std_input.shape} == {(n_in,)}'
    
    def forward(self, p):
        #p (Nb, np) 
        #Anlin (np, n_out, n_in) -> (1, np, n_out, n_in)
        #self.Alin (n_out, n_int) -> (None, n_out, n_in)
        A = (self.Alin[None] + (self.Anlin[None]*p[:,:,None,None]).sum(1)) #nbatch, n_out, n_in
        return self.std_output[:,None]*A/self.std_input[None,:]

####################
###  CNN SUBNET ####
####################

class ConvShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', upscale_factor=2, \
        padding_mode='zeros'):
        super(ConvShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels*upscale_factor**2, kernel_size, padding=padding, \
            padding_mode=padding_mode)
    
    def forward(self, X):
        X = self.conv(X) #(N, Cout*upscale**2, H, W)
        return nn.functional.pixel_shuffle(X, self.upscale_factor) #(N, Cin, H*r, W*r)


class Upscale_Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', \
                 upscale_factor=2, main_upscale=ConvShuffle, shortcut=ConvShuffle, \
                 padding_mode='zeros', activation=nn.functional.relu, Ch=0, Cw=0):
        assert isinstance(upscale_factor, int)
        super(Upscale_Conv_block, self).__init__()
        #padding='valid' is weird????
        self.shortcut = shortcut(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode, upscale_factor=upscale_factor)
        self.activation = activation
        self.upscale = main_upscale(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode, upscale_factor=upscale_factor)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode)
        self.Ch = Ch
        self.Cw = Cw
        
    def forward(self, X):
        #shortcut
        X_shortcut = self.shortcut(X) # (N, Cout, H*r, W*r)
        
        #main line
        X = self.activation(X) # (N, Cin, H, W)
        X = self.upscale(X)    # (N, Cout, H*r, W*r)
        X = self.activation(X) # (N, Cout, H*r, W*r)
        X = self.conv(X)       # (N, Cout, H*r, W*r)
        
        #combine
        # X.shape[:,Cout,H,W]
        H,W = X.shape[2:]
        H2,W2 = X_shortcut.shape[2:]
        if H2>H or W2>W:
            padding_height = (H2-H)//2
            padding_width = (W2-W)//2
            X = X + X_shortcut[:,:,padding_height:padding_height+H,padding_width:padding_width+W]
        else:
            X = X + X_shortcut
        return X[:,:,self.Ch:,self.Cw:] #slice if needed
        #Nnodes = W*H*N(Cout*4*r**2 + Cin)

class CNN_vec_to_image(nn.Module):
    def __init__(self, nx, ny, nu=-1, features_out = 1, kernel_size=3, padding='same', \
                 upscale_factor=2, feature_scale_factor=2, final_padding=4, main_upscale=ConvShuffle, shortcut=ConvShuffle, \
                 padding_mode='zeros', activation=nn.functional.relu):
        super(CNN_vec_to_image, self).__init__()
        self.feedthrough = nu!=-1
        if self.feedthrough:
            self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
            FCnet_in = nx + np.prod(self.nu, dtype=int)
        else:
            FCnet_in = nx
        
        self.activation  = activation
        assert isinstance(ny,(list,tuple)) and (len(ny)==2 or len(ny)==3), 'ny should have 2 or 3 dimentions in the form (nchannels, height, width) or (height, width)'
        if len(ny)==2:
            self.nchannels = 1
            self.None_nchannels = True
            self.height_target, self.width_target = ny
        else:
            self.None_nchannels = False
            self.nchannels, self.height_target, self.width_target = ny
        
        if self.nchannels>self.width_target or self.nchannels>self.height_target:
            import warnings
            text = f"Interpreting shape of data as (Nnchannels={self.nchannels}, Nheight={self.height_target}, Nwidth={self.width_target}), This might not be what you intended!"
            warnings.warn(text)

        #work backwards
        features_out = int(features_out*self.nchannels)
        self.final_padding = final_padding
        height_now = self.height_target + 2*self.final_padding
        width_now  = self.width_target  + 2*self.final_padding
        features_now = features_out
        
        self.upblocks = []
        while height_now>=2*upscale_factor+1 and width_now>=2*upscale_factor+1:
            
            Ch = (-height_now)%upscale_factor
            Cw = (-width_now)%upscale_factor
            # print(height_now, width_now, features_now, Ch, Cw)
            B = Upscale_Conv_block(int(features_now*feature_scale_factor), int(features_now), kernel_size, padding=padding, \
                 upscale_factor=upscale_factor, main_upscale=main_upscale, shortcut=shortcut, \
                 padding_mode=padding_mode, activation=activation, Cw=Cw, Ch=Ch)
            self.upblocks.append(B)
            features_now *= feature_scale_factor
            #implement slicing 
            
            height_now += Ch
            width_now += Cw
            height_now //= upscale_factor
            width_now //= upscale_factor
        # print(height_now, width_now, features_now)
        self.width0 = width_now
        self.height0 = height_now
        self.features0 = int(features_now)
        
        self.upblocks = nn.Sequential(*list(reversed(self.upblocks)))
        self.FC = MLP_res_net(input_size=FCnet_in,output_size=self.width0*self.height0*self.features0, n_hidden_layers=1)
        self.final_conv = nn.Conv2d(features_out, self.nchannels, kernel_size=3, padding=padding, padding_mode='zeros')
        
    def forward(self, x, u=None):
        if self.feedthrough:
            xu = torch.cat([x,u.view(u.shape[0],-1)],dim=1)
        else:
            xu = x
        X = self.FC(xu).view(-1, self.features0, self.height0, self.width0) 
        X = self.upblocks(X)
        X = self.activation(X)
        Xout = self.final_conv(X)
        if self.final_padding>0:
            Xout = Xout[:,:,self.final_padding:-self.final_padding,self.final_padding:-self.final_padding]
        return Xout[:,0,:,:] if self.None_nchannels else Xout

class ShuffleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', upscale_factor=2, padding_mode='zeros'):
        super(ShuffleConv, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode) #kernal larger?
    
    def forward(self, X):
        X = torch.cat([X]*self.upscale_factor**2,dim=1) #(N, Cin*r**2, H, W)
        X = nn.functional.pixel_shuffle(X, self.upscale_factor)  #(N, Cin, H*r, W*r)
        return self.conv(X)
        
class ClassicUpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', upscale_factor=2, padding_mode='zeros'):
        super(ClassicUpConv, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode) #kernal larger?
        self.up = nn.Upsample(size=None,scale_factor=upscale_factor,mode='bicubic',align_corners=False)

    def forward(self, X):
        X = self.up(X) #(N, Cin, H*r, W*r)
        return self.conv(X)

class Down_Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', \
                 downscale_factor=2, padding_mode='zeros', activation=nn.functional.relu):
        assert isinstance(downscale_factor, int)
        super(Down_Conv_block, self).__init__()
        #padding='valid' is weird????
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode, stride=downscale_factor)
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', padding_mode='zeros')
        self.downscale = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=padding_mode, stride=downscale_factor)
        
    def forward(self, X):
        #shortcut
        X_shortcut = self.shortcut(X) # (N, Cout, H/r, W/r)
        
        #main line
        X = self.activation(X)  # (N, Cin, H, W)
        X = self.conv(X)        # (N, Cout, H, W)
        X = self.activation(X)  # (N, Cout, H, W)
        X = self.downscale(X)   # (N, Cout, H/r, W/r)
        
        #combine
        X = X + X_shortcut
        return X

class CNN_chained_downscales(nn.Module):
    def __init__(self, ny, kernel_size=3, padding='valid', features_ups_factor=1.5, \
                 downscale_factor=2, padding_mode='zeros', activation=nn.functional.relu):

        super(CNN_chained_downscales, self).__init__()
        self.activation  = activation
        assert isinstance(ny,(list,tuple)) and (len(ny)==2 or len(ny)==3), 'ny should have 2 or 3 dimentions in the form (nchannels, height, width) or (height, width)'
        if len(ny)==2:
            self.nchannels = 1
            self.None_nchannels = True
            self.height, self.width = ny
        else:
            self.None_nchannels = False
            self.nchannels, self.height, self.width = ny
        
        #work backwards
        Y = torch.randn((1,self.nchannels,self.height,self.width))
        _, features_now, height_now, width_now = Y.shape
        
        self.downblocks = []
        features_now_base = features_now
        while height_now>=2*downscale_factor+1 and width_now>=2*downscale_factor+1:
            features_now_base *= features_ups_factor
            B = Down_Conv_block(features_now, int(features_now_base), kernel_size, padding=padding, \
                 downscale_factor=downscale_factor, padding_mode=padding_mode, activation=activation)
            
            self.downblocks.append(B)
            with torch.no_grad():
                Y = B(Y)
            _, features_now, height_now, width_now = Y.shape #i'm lazy sorry

        self.width0 = width_now
        self.height0 = height_now
        self.features0 = features_now
        self.nout = self.width0*self.height0*self.features0
        # print('CNN output size=',self.nout)
        self.downblocks = nn.Sequential(*self.downblocks)
        
    def forward(self, Y):
        if self.None_nchannels:
            Y = Y[:,None,:,:]
        return self.downblocks(Y).view(Y.shape[0],-1)
    
class CNN_encoder(nn.Module):
    def __init__(self, nb, nu, na, ny, nx, n_hidden_nodes=64, n_hidden_layers=2, activation=nn.Tanh, features_ups_factor=1.5):
        super(CNN_encoder, self).__init__()
        self.nx = nx
        self.nu = tuple() if nu=='scalar' else ((nu,) if isinstance(nu,int) else nu)
        assert isinstance(ny,(list,tuple)) and (len(ny)==2 or len(ny)==3), 'ny should have 2 or 3 dimentions in the form (nchannels, height, width) or (height, width)'
        ny = (ny[0]*na, ny[1], ny[2]) if len(ny)==3 else (na, ny[0], ny[1])
        # print('ny=',ny)

        self.CNN = CNN_chained_downscales(ny, features_ups_factor=features_ups_factor) 
        self.net = MLP_res_net(input_size=nb*np.prod(self.nu,dtype=int) + self.CNN.nout, \
            output_size=nx, n_hidden_nodes=n_hidden_nodes, n_hidden_layers=n_hidden_layers, activation=activation)


    def forward(self, upast, ypast):
        #ypast = (samples, na, W, H) or (samples, na, C, W, H) to (samples, na*C, W, H)
        ypast = ypast.view(ypast.shape[0],-1,ypast.shape[-2],ypast.shape[-1])
        # print('ypast.shape=',ypast.shape)
        ypast_encode = self.CNN(ypast)
        # print('ypast_encode.shape=',ypast_encode.shape)
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast_encode.view(ypast.shape[0],-1)],axis=1)
        return self.net(net_in)


############################################################
################ HNN SUBNET function #######################
############################################################

class ELU_lower_bound(nn.Module): 
    '''Set a lower bound on a function using a ELU using:
       torch.nn.functional.elu(y - b) + b'''
    def __init__(self, net, lower_bound=-10): #-10 such that the gradient is not suppressed near zero
        super(ELU_lower_bound, self).__init__()
        self.net = net
        self.lower_bound = lower_bound
        
    def forward(self, *args, **kwargs):    
        y = self.net(*args, **kwargs)
        b = self.lower_bound + 1
        return torch.nn.functional.elu(y - b) + b

class Ham_converter(nn.Module): #rescales the output such that the std of dH/dx = 1
    '''Converts a H(x) to a hamiltonian by multipying the output by sqrt(nx) which gives approximately dH/dx_i = 1'''
    def __init__(self, net, norm='auto'):
        super().__init__()
        self.net = net
        self.norm = norm
    
    def forward(self, x):
        if self.norm=='auto':
            return self.net(x)*x.shape[1]**0.5
        else:
            return self.net(x)*self.norm

class Matrix_converter(nn.Module):
    '''
    Converts a net(x) vector to a matrix using a reshape
    '''
    def __init__(self, net, nrows, ncols, norm='auto'):
        super().__init__()
        self.net = net
        self.norm = norm
        self.nrows = nrows
        self.ncols = ncols

    def forward(self, *x):
        A = self.net(*x).view(x[0].shape[0], self.nrows, self.ncols)
        if self.norm=='auto':
            A = A/(self.ncols**0.5) #this can be improved with some additional math 
        else:
            A = A/self.norm #this can be improved with some additional math 
        return A

class Skew_sym_converter(nn.Module):
    '''converts a net(x) vector to a skew-symtreic matrix (J = -J^T) using
        A = shape_to_matrix(net(x))
        return A - A^T
    '''
    def __init__(self, net, norm='auto'):
        super().__init__()
        self.net = net
        self.norm = norm

    def forward(self, x):
        z = self.net(x)
        #z.shape = (Nbatch, nx*nx)
        nx = int(round(z.shape[1]**0.5))
        assert nx*nx==z.shape[1], 'the output of net needs to have a sqaure number of elements to be reshaped to a square matrix'
        J = z.view(z.shape[0], nx, nx)
        if self.norm=='auto':
            J = J/(((nx-1)*2)**0.5) #this can be improved with some additional math 
        else:
            J = J/self.norm #this can be improved with some additional math 
        return J - J.permute(0,2,1)

class Sym_pos_semidef_converter(nn.Module):
    '''converts a net(x) vector to a semi-positive definite matrix using 
        A = shape_to_matrix(net(x))
        return A^T A
    '''
    def __init__(self, net, norm='auto'):
        super().__init__()
        self.norm = norm
        self.net = net

    def forward(self, x):
        z = self.net(x)
        nx = int(round(z.shape[1]**0.5))
        assert nx*nx==z.shape[1], 'the output of net needs to have a sqaure number of elements to be reshaped to a square matrix'
        A = z.view(z.shape[0], nx, nx)
        if self.norm=='auto':
            A = A/(((nx+2)*nx**2)**0.25) #this might not be entirely correct
        else:
            A = A/self.norm
        R = torch.einsum('bik,bjk->bij', A, A)
        return R

class Bias_net(nn.Module): 
    '''f(x)=b is a bias (trainable)'''
    def __init__(self, num_pars, requires_grad=True):
        super().__init__()
        self.pars = nn.Parameter(torch.randn(num_pars), requires_grad=requires_grad)

    def forward(self, *args, **kwargs):
        return torch.broadcast_to(self.pars, (args[0].shape[0], self.pars.shape[0]))  

class Contant_net(nn.Module):  #todo documentation
    '''f(x)=c is a constant given by c'''
    def __init__(self, c):
        super().__init__()
        assert isinstance(c, torch.Tensor)
        self.c = c 

    def forward(self, *args, **kwargs):
        return torch.broadcast_to(self.c, (args[0].shape[0],) + self.c.shape) 


class Sum_net(nn.Module):
    '''f_1(x) + f_2(x) + f_3(x) + ... + f_n(x)'''
    def __init__(self, nets, scaling_factors='auto'):
        super().__init__()
        self.nets = nn.ParameterList(nets)
        self.scaling_factors = [1/len(nets)**0.5]*len(nets) if scaling_factors=='auto' else scaling_factors

    def forward(self, *args, **kwargs):
        outputs = [scaling*net(*args, **kwargs) for scaling, net in zip(self.scaling_factors, self.nets)]
        return torch.stack(outputs,dim=0).sum(0)


class Quadratic_net(nn.Module): 
    '''x^T Q X'''
    def __init__(self, nx):
        super().__init__()
        self.net = Skew_sym_converter(Bias_net(nx*nx))

    def forward(self, x):
        Q = self.net(x)
        return torch.einsum('bi,bij,bj->b', x, Q, x)
