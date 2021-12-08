import torch
from torch import nn, optim
import numpy as np


class feed_forward_nn(nn.Module): #a simple MLP
    def __init__(self,n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
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

class MLP_res_block(nn.Module):
    def __init__(self, n_in=64, n_out=64, activation=nn.Tanh, force_linear_res=False):
        super(MLP_res_block,self).__init__()
        if n_in==n_out and not force_linear_res:
            self.skipper = True
        else:
            self.skipper = False
            self.res = nn.Linear(n_in, n_out) 
        self.nonlin = nn.Linear(n_in, n_out)
        self.activation = activation()

    def forward(self,x):
        if self.skipper:
            return x + self.activation(self.nonlin(x))
        else:
            return self.res(x) + self.activation(self.nonlin(x))

class complete_MLP_res_net(nn.Module):
    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh, force_linear_res=False):
        super(complete_MLP_res_net,self).__init__()
        assert n_hidden_layers>0, 'just use nn.Linear lol'
        seq = [MLP_res_block(n_in, n_nodes_per_layer, activation=activation, force_linear_res=force_linear_res)]
        for i in range(n_hidden_layers-1):
            seq.append(MLP_res_block(n_nodes_per_layer, n_nodes_per_layer, activation=activation, force_linear_res=force_linear_res))
        seq.append(nn.Linear(n_nodes_per_layer, n_out))
        self.net = nn.Sequential(*seq)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, val=0) #bias zero
    def forward(self,X):
        return self.net(X)  


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


#todo:
# class ConvTranspose(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding='same', upscale_factor=2, padding_mode='replicate'):
#         super(ConvTranspose, self).__init__()
#         self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=upscale_factor, padding=padding, padding_mode='zeros')

#     def forward(self, X):
#         print(X.shape)
#         X = self.conv(X) #padding is weird
#         print(X.shape)
#         return X #(N, Cin, H*r, W*r)

#todo figure out valid padding and compare to zero padding implemention on cnntesting project.

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

class CNN_chained_upscales(nn.Module):
    def __init__(self, nx, ny, features_out = 1, kernel_size=3, padding='same', \
                 upscale_factor=2, feature_scale_factor=2, final_padding=4, main_upscale=ConvShuffle, shortcut=ConvShuffle, \
                 padding_mode='zeros', activation=nn.functional.relu):

        super(CNN_chained_upscales, self).__init__()
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
        self.FC = simple_res_net(n_in=nx,n_out=self.width0*self.height0*self.features0, n_hidden_layers=1)
        self.final_conv = nn.Conv2d(features_out, self.nchannels, kernel_size=3, padding=padding, padding_mode='zeros')
        
    def forward(self, x):
        X = self.FC(x).view(-1, self.features0, self.height0, self.width0) 
        X = self.upblocks(X)
        X = self.activation(X)
        Xout = self.final_conv(X)
        if self.final_padding>0:
            Xout = Xout[:,:,self.final_padding:-self.final_padding,self.final_padding:-self.final_padding]
        return Xout[:,0,:,:] if self.None_nchannels else Xout

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
        X = self.activation(X)  # (N, Cout, H, W/r)
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
    def __init__(self, nb, nu, na, ny, nx, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh, features_ups_factor=1.33):
        super(CNN_encoder, self).__init__()
        self.nx = nx
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        assert isinstance(ny,(list,tuple)) and (len(ny)==2 or len(ny)==3), 'ny should have 2 or 3 dimentions in the form (nchannels, height, width) or (height, width)'
        ny = (ny[0]*na, ny[1], ny[2]) if len(ny)==3 else (na, ny[0], ny[1])
        # print('ny=',ny)

        self.CNN = CNN_chained_downscales(ny, features_ups_factor=features_ups_factor) 
        self.net = simple_res_net(n_in=nb*np.prod(self.nu,dtype=int) + self.CNN.nout, \
            n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)


    def forward(self, upast, ypast):
        #ypast = (samples, na, W, H) or (samples, na, C, W, H) to (samples, na*C, W, H)
        ypast = ypast.view(ypast.shape[0],-1,ypast.shape[-2],ypast.shape[-1])
        # print('ypast.shape=',ypast.shape)
        ypast_encode = self.CNN(ypast)
        # print('ypast_encode.shape=',ypast_encode.shape)
        net_in = torch.cat([upast.view(upast.shape[0],-1),ypast_encode.view(ypast.shape[0],-1)],axis=1)
        return self.net(net_in)

class Shotgun_MLP(nn.Module):
    def __init__(self, nx, ny, positional_encoding=1.3, n_nodes_per_layer=256, n_hidden_layers=3):
        super(Shotgun_MLP,self).__init__()
        if len(ny)==2:
            self.H, self.W = ny
            self.C = 1
            self.Cnone = True
        else:
            self.C, self.H, self.W = ny
            self.Cnone = False
        
        self.positional_encoding = positional_encoding
        if positional_encoding:
            assert positional_encoding>1
            nh = int(np.ceil((np.log(self.H) - np.log(1))/np.log(positional_encoding)))
            nw = int(np.ceil((np.log(self.W) - np.log(1))/np.log(positional_encoding)))
            self.kh = nn.Parameter(positional_encoding**torch.arange(0,nh),requires_grad=False)
            self.kw = nn.Parameter(positional_encoding**torch.arange(0,nw),requires_grad=False)
        else:
            self.kh, self.kw = [], []
            
        self.net = complete_MLP_res_net(n_in=nx+len(self.kh)*2+len(self.kw)*2+2, n_out=self.C, \
                                        n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers)
        
        
    def forward(self, x): #produces the full image
        h = torch.arange(start=0, end=self.H,device=x.device)
        w = torch.arange(start=0, end=self.W,device=x.device)
        h,w = torch.meshgrid(h,w)
        h,w = h.flatten(), w.flatten()
        Nb = x.shape[0]
        Ngrid = len(h)
        h = torch.broadcast_to(h[None,:], (Nb, Ngrid))
        w = torch.broadcast_to(w[None,:], (Nb, Ngrid))
        out = self.sampler(x, h, w) #(Nb, H*W, C or None)
        if self.Cnone:
            return out.reshape(Nb,self.H,self.W)
        else:
            return out.swapaxes(1,2).reshape(Nb,self.C,self.H,self.W)
        
    
    def sampler(self, x, h, w):
        # x = (Nb, nx)    -> (Nb, 1,    nx)
        # w = (Nb, Nsamp) -> (Nb, Nsamp, 1) or (Nb, Nsamp, 1 + log2(W)) if positional encoding
        # h = (Nb, Nsamp) -> (Nb, Nsamp, 1) or (Nb, Nsamp, 1 + log2(W)) if positional encoding
        # concat x:
        #                (Nb, Nsamp, nx + 2)
        #                (Nb*Nsamp, nx + 2) pulled through network to (Nb*Nsamp, C)
        #  Reshape back  (Nb, Nsamp, C) 
        Nb, nx = x.shape
        _, Nsamp = w.shape
        S = (Nb, Nsamp, nx)
        
        h = h/(self.H-1) #0 to 1
        w = w/(self.W-1) #0 to 1
        if self.positional_encoding:
            sincosargh = np.pi*(self.kh[None,None,:]*h[:,:,None])
            sincosargw = np.pi*(self.kw[None,None,:]*w[:,:,None])
            h = torch.cat([torch.sin(sincosargh), torch.cos(sincosargh), h[:,:,None]],dim=2)
            w = torch.cat([torch.sin(sincosargw), torch.cos(sincosargw), w[:,:,None]],dim=2)
        else:
            h = h[:,:,None]
            w = w[:,:,None]
        
        x = torch.broadcast_to(x[:,None,:], S)
        X = torch.cat((x,h,w),dim=2).flatten(end_dim=-2)
        Y = self.net(X)
        return Y.view(Nb, Nsamp) if self.Cnone else Y.view(Nb, Nsamp, self.C)

class Shotgun_encoder(nn.Module):
    def __init__(self, nb, nu, na, ny, nx, Nsamp=256, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(Shotgun_encoder, self).__init__()
        self.nx = nx
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        assert isinstance(ny,(list,tuple)) and (len(ny)==2 or len(ny)==3), 'ny should have 2 or 3 dimentions in the form (nchannels, height, width) or (height, width)'
        ny = (ny[0]*na, ny[1], ny[2]) if len(ny)==3 else (na, ny[0], ny[1])
        #ny = (C, H, W)
        self.c = nn.Parameter(torch.randint(0,ny[0],size=(Nsamp,)),requires_grad=False)
        self.h = nn.Parameter(torch.randint(0,ny[1],size=(Nsamp,)),requires_grad=False)
        self.w = nn.Parameter(torch.randint(0,ny[2],size=(Nsamp,)),requires_grad=False)
        # self.CNN = CNN_chained_downscales(ny, features_ups_factor=features_ups_factor) 
        self.net = simple_res_net(n_in=nb*np.prod(self.nu,dtype=int) + Nsamp, \
            n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)


    def forward(self, upast, ypast):
        #ypast = (samples, na, H, W) or (samples, na, C, H, W) to (samples, na*C, H, W)
        ypast = ypast.view(ypast.shape[0],-1,ypast.shape[-2],ypast.shape[-1])
        ysamps = ypast[:,self.c, self.h, self.w]
        net_in = torch.cat([upast.view(upast.shape[0],-1),ysamps.view(ysamps.shape[0],-1)],axis=1)
        return self.net(net_in)



if __name__ == '__main__':
    F = CNN_chained_downscales((2,177,177), features_ups_factor=1.5)
    # print(F.downblocks)   
    print(F(torch.rand(4,2,177,177)).shape)
    nb = 4
    nu = None
    na = 3
    ny = (100,100)
    nx = 10
    N = 2
    F = CNN_encoder(nb=nb,nu=nu,na=na,ny=ny, nx=nx)
    print(F(torch.rand(N,nb), torch.rand(N,na,*ny)).shape)

# class FC_video(nn.Module): #use general encoder
#     def __init__(self, nx, ny, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
#         super(FC_video, self).__init__()
#         assert isinstance(ny,(list,tuple)) and (len(ny)==2 or len(ny)==3), 'ny should have 2 or 3 dimentions in the form (nchannels, height, width) or (height, width)'
#         if len(ny)==2:
#             self.nchannels = 1
#             self.None_nchannels = True
#             self.height_target, self.width_target = ny
#         else:
#             self.None_nchannels = False
#             self.nchannels, self.height_target, self.width_target = ny

#         self.FC = simple_res_net(n_in=nx, n_out=self.nchannels*self.height_target*self.width_target, \
#                   n_hidden_layers=n_hidden_layers, activation=activation, n_nodes_per_layer=n_nodes_per_layer)

#     def forward(self,x):
#         Xout = self.FC(x).view(-1, self.nchannels, self.height_target, self.width_target)
#         return Xout[:,0,:,:] if self.None_nchannels else Xout

if __name__ == '__main__':
    ny = (20,20)
    nx = 8
    upscale_factor = 2
    features_out = 2
    for up in [ConvShuffle, ShuffleConv, ClassicUpConv]:
        padding_mode='replicate' #or zero
        activation = nn.functional.relu # or torch.tanh
        f = CNN_chained_upscales(nx, ny, features_out=features_out, \
                             shortcut=up, main_upscale=up, padding_mode=padding_mode, \
                            activation=activation, upscale_factor=upscale_factor)
        print(f(torch.randn(1,nx)).shape)

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
    def __init__(self, deriv, f_norm=1, dt_base=1, dt=None):
        '''include time normalization as dt = f_norm*dt, f_norm is often dx/dt'''
        super(time_integrators,self).__init__()
        self.dt_checked = False
        self.dt_valued = None

        self.f_norm = f_norm/dt_base if dt_base is not None else f_norm #normalized dt in units of x
                                     #this a parameter?
        self._dt = dt #the current time constant (most probably the same as dt_0)
                             #should be set using set_dt before applying any dataset
        self.deriv_base   = deriv #the deriv network

    def deriv(self,x,u):
        return self.f_norm*self.deriv_base(x,u)

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self,dt):
        if not self.dt_checked: #checking for mixing of None valued dt and valued dt 
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
    def get_sys_epoch(f_norm=0.12,nf=30,epochs=200,n_hidden_layers=1,n_nodes_per_layer=64,euler=False):
        test, train = deepSI.datasets.CED() #switch
        test.dt = None
        train.dt = None
        f_net_kwargs=dict(n_hidden_layers=n_hidden_layers,n_nodes_per_layer=n_nodes_per_layer)
        integrator = integrator_euler if euler else integrator_RK4
        sys = deepSI.fit_systems.SS_encoder_deriv_general(nx=3,na=7,nb=7,f_norm=f_norm,dt_base=train.dt,\
                f_net_kwargs=f_net_kwargs,h_net_kwargs=dict(n_hidden_layers=2),integrator_net=integrator)
        sys.fit(train,sim_val=test,concurrent_val=True,batch_size=32,\
                    epochs=epochs,verbose=2,loss_kwargs=dict(nf=nf))
        sys.checkpoint_load_system(name='_best')
        return sys

    from torch import manual_seed
    np.random.seed(5)
    manual_seed(5)
    sys = get_sys_epoch(nf=30, epochs=400, n_hidden_layers=1, n_nodes_per_layer=64,euler=True)
    # test.dt = 4
    sys.apply_experiment(test)