
from deepSI.fit_systems.fit_system import System_fittable, random_search, grid_search, System_pytorch
from deepSI.systems.System import System_io
import deepSI
import torch
from torch import nn


class System_IO_pytorch(System_pytorch, System_io):
    def __init__(self,na=5,nb=5):
        super(System_IO_pytorch,self).__init__(na=na,nb=nb)
        
        from deepSI.utils import simple_res_net, feed_forward_nn
        self.net = simple_res_net
        self.n_hidden_layers = 2
        self.n_nodes_per_layer = 64
        self.activation = nn.Tanh

    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        nf = Loss_kwargs.get('nf',25)
        return sys_data.to_hist_future_data(na=self.na,nb=self.nb, nf=nf, force_multi_u=True, force_multi_y=True) #returns np.array(uhist),np.array(yhist),np.array(ufuture),np.array(yfuture)
    

    def init_nets(self, nu, ny): # a bit weird
        self.ny_real = ny
        self.nu, self.ny = nu if nu is not None else 1, ny if ny is not None else 1 #multi y multi u
        self.gn = self.net(n_in=self.nb*self.nu+self.na*self.ny, n_out=self.ny, n_nodes_per_layer=self.n_nodes_per_layer, n_hidden_layers=self.n_hidden_layers, activation=self.activation)
        return list(self.gn.parameters())

    def CallLoss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        Loss = torch.zeros(1,dtype=yhist.dtype,device=yhist.device)[0]
        for unow, ynow in zip(torch.transpose(ufuture,0,1), torch.transpose(yfuture,0,1)): #unow = (Nsamples, nu), ynow = (Nsamples, ny)
            g_in = torch.cat([torch.flatten(uhist, start_dim=1), torch.flatten(yhist, start_dim=1)],axis=1)
            yout = self.gn(g_in) #(Nsamples, ny)
            Loss += torch.mean((yout - ynow)**2)**0.5
            uhist = torch.cat((uhist[:,1:,:],unow[:,None,:]),dim=1)
            yhist = torch.cat((yhist[:,1:,:],yout[:,None,:]),dim=1)
        Loss /= ufuture.shape[1]
        return Loss

    def multi_IO_step(self,uy):
        # uy #(N_samples, uy len)
        with torch.no_grad():
            yout = self.gn(torch.tensor(uy,dtype=torch.float32)).detach().numpy()
            if self.ny_real is None:
                return yout[:,0]
            else:
                return yout

    def IO_step(self,uy):
        # uy #(uy len)
        with torch.no_grad():
            yout = self.gn(torch.tensor(uy[None,:],dtype=torch.float32))[0].detach().numpy()
            if self.ny_real is None:
                return yout[0]
            else:
                return yout


class System_IO_SISO_pytorch(System_pytorch, System_io):
    def __init__(self,na,nb):
        super(System_IO_SISO_pytorch, self).__init__(na,nb)

    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        return sys_data.to_IO_data(na=self.na,nb=self.nb) #np.array(hist), np.array(Y)

    def init_nets(self, nu, ny):
        assert ny==None
        #returns parameters
        nu = 1 if nu is None else nu
        one_out = ny==None
        ny = 1 if ny is None else ny
        n_in = nu*self.nb + ny*self.na
        IN = [nn.Linear(n_in,64),nn.Tanh(),nn.Linear(64,ny),nn.Flatten()]
        self.net = nn.Sequential(*IN)
        return self.net.parameters()

    def CallLoss(self,hist,Y, **kwargs):
        return torch.mean((self.net(hist)[:,0]-Y)**2)

    def IO_step(self,uy):
        uy = torch.tensor(uy,dtype=torch.float32)
        if uy.ndim==1:
            uy = uy[None,:]
            return self.net(uy)[0,0].item()
        else:
            return self.net(uy)[:,0].detach().numpy()


if __name__ == '__main__':
    sys = System_IO_pytorch()
    train, test = deepSI.datasets.SISTA_Database.winding()
    print(train.nu,test.ny)
    sys.fit(train,sim_val=test,batch_size=16)
    sys.save_system('../../testing/pytorchIO.system')
    sys = deepSI.load_system('../../testing/pytorchIO.system')
    test_predict = sys.apply_experiment(test)
    test.plot()
    test_predict.plot(show=True)