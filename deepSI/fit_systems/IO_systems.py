
from deepSI.fit_systems.Fit_system import System_IO_fit_sklearn, System_fittable, fit_system_tuner, System_Torch_IO, System_PyTorch
from deepSI.systems.System import System_IO
import deepSI
import torch
from torch import nn


class System_IO_pytorch(System_PyTorch, System_IO):
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
            yout = self.gn(g_in                                                                                                     ) #(Nsamples, ny)
            Loss += torch.mean((yout - ynow)**2)**0.5
            self.uhist = torch.cat((uhist[:,1:,:],unow[:,None,:]),dim=1)
            self.yhist = torch.cat((yhist[:,1:,:],yout[:,None,:]),dim=1)
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