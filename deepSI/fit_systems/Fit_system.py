
from deepSI.systems.System import System, System_IO, System_data, load_system
import numpy as np
import deepSI
import torch
from tqdm.auto import tqdm

class System_fittable(System):
    """docstring for System_fit"""
    def fit(self,sys_data,**kwargs):
        if self.fitted==False:
            self.norm.fit(sys_data)
            self.nu = sys_data.nu
            self.ny = sys_data.ny
        self._fit(self.norm.transform(sys_data),**kwargs) #transfrom data to fittable data?
        self.fitted = True        

class System_IO_fit_sklearn(System_fittable, System_IO): #name?
    def __init__(self, na, nb, reg):
        super(System_IO_fit_sklearn, self).__init__(na, nb)
        self.reg = reg

    def _fit(self,sys_data):
        #sys_data #is already normed fitted on 
        hist,y = sys_data.to_IO_data(na=self.na,nb=self.nb)
        self.reg.fit(hist,y)

    def IO_step(self,uy):
        return self.reg.predict([uy])[0] if uy.ndim==1 else reg.predict(uy)

class System_PyTorch(System_fittable):
    """docstring for System_PyTorch"""
    # def __init__(self, arg):
    #     super(System_PyTorch, self).__init__()
    #     self.arg = arg

    def init_nets(self,nu,ny):
        #returns parameters
        raise NotImplementedError

    def init_optimizer(self,parameters):
        #return the optimizer with a optimizer.zero_grad and optimizer.step method
        return torch.optim.Adam(parameters,lr=3e-3) 

    def make_training_data(self,sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        raise NotImplementedError


    def fit(self, sys_data, epochs=30, batch_size=256, Loss_kwargs={}, sim_val=True, n_val=10000, verbose=1):
        #1. init funcs already happened
        #2. init optimizer
        #3. training data
        #4. optimization

        if self.fitted==False:
            self.norm.fit(sys_data)
            self.nu = sys_data.nu
            self.ny = sys_data.ny
            self.paremters = list(self.init_nets(self.nu,self.ny))
            self.optimizer = self.init_optimizer(self.paremters)

        sys_data, sys_data0 = self.norm.transform(sys_data), sys_data
        data_full = self.make_training_data(sys_data,**Loss_kwargs)
        data_full = [torch.tensor(dat,dtype=torch.float32) for dat in data_full]

        data_train = data_full #later validation


        N_training_samples = len(data_train[0])
        ids = np.arange(0,N_training_samples,dtype=int)
        for epoch in tqdm(range(epochs)):
            np.random.shuffle(ids)

            Losses = 0
            for i in range(batch_size,N_training_samples+1,batch_size):
                ids_batch = ids[i-batch_size:i]
                train_batch = [part[ids_batch] for part in data_train]
                Loss = self.CallLoss(*train_batch,**Loss_kwargs)
                self.optimizer.zero_grad()
                Loss.backward()
                self.optimizer.step()
                Losses += Loss.item()
            Losses /= N_training_samples//batch_size

            data_val = self.apply_experiment(sys_data0[-n_val:])
            Loss_val = data_val.NRMS(sys_data0[-n_val:])

            print(f'epoch={epoch} NRMS={Loss_val:9.4%} Loss={Losses:.5f}')

        self.fitted = True       

    def CallLoss(*args,**kwargs):
        #kwargs are the settings
        #args is the data
        raise NotImplementedError
    
    def _pytorch_save_system(self,vars=[],dir_placement=None,name=''):
        self._save_system_torch(dir_placement=dir_placement,name=name) #error here if you have 
        if self.fitted:
            this_vars = self.fitted,self.bestfit,self.Loss_val,self.Loss_train,self.batch_id,self.time
        else:
            this_vars = self.fitted,self.bestfit
        self._save_system_np(this_vars,dir_placement=dir_placement,name=name) #also could save 
        self._save_system_pickle([self.norm_sys_data,self.norm]+vars,dir_placement=dir_placement,name=name)

    def _pytorch_load_system(self,dir_placement=None,name=''):
        self._load_system_torch(dir_placement=dir_placement,name=name)
        vars = self._load_system_np(dir_placement=dir_placement,name=name)
        self.fitted = vars[0]
        if self.fitted:
            self.fitted, self.bestfit, self.Loss_val, self.Loss_train,self.batch_id,self.time = vars
            self.Loss_val, self.Loss_train, self.batch_id, self.time = self.Loss_val.tolist(), self.Loss_train.tolist(), self.batch_id.tolist(), self.time.tolist()
        else:
            self.fitted,self.bestfit = vars
        self.norm_sys_data,self.norm = self._load_system_pickle(dir_placement=dir_placement,name=name)[:2]
        


import torch
from torch import nn

class System_Torch_IO(System_PyTorch, System_IO):
    def __init__(self,na,nb):
        super(System_Torch_IO, self).__init__(na,nb)

    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        return sys_data.to_IO_data(na=self.na,nb=self.nb) #np.array(hist), np.array(Y)

    def init_nets(self, nu, ny):
        print(nu,ny)
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
        uy = torch.tensor(uy,dtype=torch.float32)[None,:]
        return self.net(uy)[0,0].item()

class System_encoder(System_PyTorch):
    """docstring for System_encoder"""
    def __init__(self, nx=10, na=20, nb=20):
        super(System_encoder, self).__init__(None,None)
        self.nx = nx
        self.na = na
        self.nb = nb

    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        nf = Loss_kwargs.get('nf',25)
        return sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=nf) #np.array(hist),np.array(ufuture),np.array(yfuture)

    def init_nets(self, nu, ny): # a bit weird
        print(nu,ny)
        assert ny==None and nu==None
        ny = 1
        nu = 1
        self.encoder = nn.Sequential(nn.Linear(self.na+self.nb,64),nn.Tanh(),nn.Linear(64,self.nx))
        self.fn = nn.Sequential(nn.Linear(self.nx+nu,64),nn.Tanh(),nn.Linear(64,self.nx))
        self.hn = nn.Sequential(nn.Linear(self.nx,64),nn.Tanh(),nn.Linear(64,ny),nn.Flatten())
        return list(self.encoder.parameters()) + list(self.fn.parameters()) + list(self.hn.parameters())

    def CallLoss(self, hist, ufuture, yfuture, **Loss_kwargs):
        x = self.encoder(hist)
        y_predict = []
        for u in torch.transpose(ufuture,0,1):
            y_predict.append(self.hn(x)[:,0]) #output prediction
            fn_in = torch.cat((x,u[:,None]),dim=1)
            x = self.fn(fn_in)
        return torch.mean((torch.stack(y_predict,dim=1)-yfuture)**2)

    def init_state(self,sys_data): #put nf here for n-step error?
        hist = torch.tensor(sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=len(sys_data)-max(self.na,self.nb))[0][:1],dtype=torch.float32) #(1,)
        self.state = self.encoder(hist)
        return self.hn(self.state)[0].item(), max(self.na,self.nb)

    def step(self,action):
        action = torch.tensor(action,dtype=torch.float32) #number
        self.state = self.fn(torch.cat((self.state,action[None,None]),axis=1))
        return self.hn(self.state)[0].item()



def fit_system_tuner(fit_system, sys_data, search_dict, verbose=1):
    import copy
    #example use: print(hyper_parameter_tunner(System_IO_fit_linear,dict(na=[1,2,3],nb=[1,2,3]),sys_data))
    def itter(itter_dict, k=0, dict_now=None, best_score=float('inf'), best_sys=None, best_dict=None):
        if dict_now is None:
            dict_now = dict()
        if k==len(itter_dict):
            sys = fit_system(**dict_now)
            sys.fit(sys_data)
            try:
                score = sys.apply_experiment(sys_data).NRMS(sys_data)
            except ValueError:
                score = float('inf')
            if verbose>0: print(score, dict_now)
            if score<=best_score:
                return score, sys, copy.deepcopy(dict_now)
            else:
                return best_score, best_sys, best_dict
        else:
            for item in itter_dict[k][1]:
                dict_now[itter_dict[k][0]] = item
                best_score, best_sys, best_dict = itter(itter_dict, k=k+1, dict_now=dict_now, best_score=best_score, best_sys=best_sys, best_dict=best_dict)
            return best_score, best_sys, best_dict

    itter_dict = [list(a) for a in search_dict.items()]
    for k in range(len(itter_dict)):
        I = itter_dict[k][1]
        if isinstance(I,range):
            itter_dict[k][1] = list(I)
        elif not isinstance(I,(tuple,list)):
            itter_dict[k][1] = list([I])
    best_score, best_sys, best_dict = itter(itter_dict)
    if verbose>0: print('Result:', best_score, best_sys, best_dict)
    return best_sys, best_score, best_dict

if __name__ == '__main__':
    from sklearn import linear_model 
    class System_IO_fit_linear(System_IO_fit_sklearn):
        def __init__(self,na,nb):
            super(System_IO_fit_linear,self).__init__(na,nb,linear_model.LinearRegression())

    train, test = deepSI.datasets.WienerHammerBenchMark()

    # sys0 = deepSI.systems.sys_ss_test()
    # sys_data = sys0.apply_experiment(System_data(u=np.random.normal(size=10000)))

    # sys = System_IO_fit_linear(7,3)
    sys = System_encoder(nx=8,na=20,nb=20)
    # sys_data = System_data(u=np.random.normal(size=100),y=np.random.normal(size=100))
    sys.fit(train,epochs=100,Loss_kwargs=dict(nf=40))

    sys_data_predict = sys.apply_experiment(test)
    print(sys_data_predict.NRMS(sys_data))
    # sys.save_system('../../testing/test-fit.p')
    # del sys
    # sys = load_system('../../testing/test-fit.p')
    # sys_data_predict2 = sys.apply_experiment(sys_data)

    sys_data.plot()
    sys_data_predict.plot(show=True)
    # sys_data_predict2.plot(show=True)

