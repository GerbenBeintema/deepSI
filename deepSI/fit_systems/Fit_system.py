
from deepSI.systems.System import System, System_IO, System_data, load_system
import numpy as np
from deepSI.system_data.datasets import get_work_dirs
import deepSI
import torch
from torch import nn, optim
from tqdm.auto import tqdm
import time

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
        return self.reg.predict([uy])[0] if uy.ndim==1 else self.reg.predict(uy)


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


    def fit(self, sys_data, epochs=30, batch_size=256, Loss_kwargs={}, sim_val=None, verbose=1, val_frac = 0.2):
        #1. init funcs already happened
        #2. init optimizer
        #3. training data
        #4. optimization

        def validation():
            if sim_val is not None:
                sim_val_data = self.apply_experiment(sim_val)
                Loss_val = sim_val_data.NRMS(sim_val)
            else:
                with torch.no_grad():
                    Loss_val = self.CallLoss(*data_val,**Loss_kwargs).item()
            self.Loss_val.append(Loss_val)
            if self.bestfit>Loss_val:
                print('########## new best ###########')
                self.checkpoint_save_system()
                self.bestfit = Loss_val
            return Loss_val

        if self.fitted==False:
            self.norm.fit(sys_data)
            self.nu = sys_data.nu
            self.ny = sys_data.ny
            self.paremters = list(self.init_nets(self.nu,self.ny))
            self.optimizer = self.init_optimizer(self.paremters)
            self.bestfit = float('inf')
            self.Loss_val,self.Loss_train,self.batch_id,self.time = [],[],[],[]
            self.batch_counter = 0
            extra_t = 0
            self.fitted = True
        else:
            self.batch_counter = 0 if len(self.batch_id)==0 else self.batch_id[-1]
            extra_t = 0 if len(self.time)==0 else self.time[-1]


        sys_data, sys_data0 = self.norm.transform(sys_data), sys_data
        data_full = self.make_training_data(sys_data,**Loss_kwargs)
        data_full = [torch.tensor(dat,dtype=torch.float32) for dat in data_full]


        if sim_val is not None:
            data_train = data_full #later validation
        else:
            split = int(len(data_full)*(1-val_frac))
            data_train = [dat[:split] for dat in data_full]
            data_val = [dat[split:] for dat in data_full]

        Loss_val = validation()
        N_training_samples = len(data_train[0])
        batch_size = min(batch_size,N_training_samples)
        ids = np.arange(0,N_training_samples,dtype=int)
        try:
            self.start_t = time.time()
            for epoch in tqdm(range(epochs)):
                np.random.shuffle(ids)

                Loss_acc = 0
                for i in range(batch_size,N_training_samples+1,batch_size):
                    ids_batch = ids[i-batch_size:i]
                    train_batch = [part[ids_batch] for part in data_train]
                    Loss = self.CallLoss(*train_batch,**Loss_kwargs)
                    self.optimizer.zero_grad()
                    Loss.backward()
                    self.optimizer.step()
                    Loss_acc += Loss.item()
                Loss_acc /= N_training_samples//batch_size
                self.batch_counter += N_training_samples//batch_size
                self.batch_id.append(self.batch_counter)
                self.Loss_train.append(Loss_acc)
                self.time.append(time.time()-self.start_t+extra_t)

                Loss_val = validation()
                print(f'epoch={epoch} NRMS={Loss_val:9.4%} Loss={Loss_acc:.5f}')
        except KeyboardInterrupt:
            print('stopping early due to KeyboardInterrupt')
        self.checkpoint_load_system()

    def CallLoss(*args,**kwargs):
        #kwargs are the settings
        #args is the data
        raise NotImplementedError
    
    def checkpoint_save_system(self):
        from pathlib import Path
        import os.path
        directory  = get_work_dirs()['checkpoints']
        self._save_system_torch(file=os.path.join(directory,self.name+'_best'+'.pth')) #error here if you have 
        vars = self.norm.u0, self.norm.ustd, self.norm.y0, self.norm.ystd, self.fitted, self.bestfit, self.Loss_val, self.Loss_train, self.batch_id, self.time
        np.savez(os.path.join(directory,self.name+'_best'+'.npz'),*vars)

    def checkpoint_load_system(self):
        from pathlib import Path
        import os.path
        directory  = get_work_dirs()['checkpoints'] 
        self._load_system_torch(file=os.path.join(directory,self.name+'_best'+'.pth'))
        out = np.load(os.path.join(directory,self.name+'_best'+'.npz'))
        out_real = [(a[1].tolist() if a[1].ndim==0 else a[1]) for a in out.items()]

        # out_real = []
        # for a in out.items(): #if it is a ndim number it will remove the array
        #     a = a[1] #keys only
        #     if a.ndim==0:
        #         out_real.append(a.tolist())
        #     else:
        #         out_real.append(a)
        self.norm.u0, self.norm.ustd, self.norm.y0, self.norm.ystd, self.fitted, self.bestfit, self.Loss_val, self.Loss_train, self.batch_id, self.time = out_real
        
    #torch variant
    def _save_system_torch(self,file):
        # save_dir = tempfile.gettempdir() if dir_placement is None else Path(dir_placement)
        # if os.path.isdir(save_dir) is False:
        #     os.mkdir(save_dir)
        save_dict = {}
        for d in dir(self):
            attribute = self.__getattribute__(d)
            if isinstance(attribute,(nn.Module,optim.Optimizer)):
                save_dict[d] = attribute.state_dict()
        # name_file = os.path.join(save_dir,self.name + name + '.pth')
        torch.save(save_dict,file)
    def _load_system_torch(self,file):
        # save_dir = tempfile.gettempdir() if dir_placement is None else Path(dir_placement)
        # assert os.path.isdir(save_dir)
        # name_file = os.path.join(save_dir,self.name + name + '.pth')
        save_dict = torch.load(file)
        for key in save_dict:
            attribute = self.__getattribute__(key)
            try:
                attribute.load_state_dict(save_dict[key])
            except (AttributeError, ValueError):
                print('Error loading key',key)


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

    train, test = deepSI.datasets.Cascaded_Tanks()
    sys = System_Torch_IO(na=5,nb=5)
    # sys0 = deepSI.systems.sys_ss_test()
    # sys_data = sys0.apply_experiment(System_data(u=np.random.normal(size=10000)))

    # sys = System_IO_fit_linear(7,3)
    # sys = System_encoder(nx=8,na=20,nb=20)
    # sys_data = System_data(u=np.random.normal(size=100),y=np.random.normal(size=100))
    sys.fit(train,epochs=100,Loss_kwargs=dict(nf=40),sim_val=test)

    sys_data_predict = sys.apply_experiment(test)
    print(sys_data_predict.NRMS(test))
    # sys.save_system('../../testing/test-fit.p')
    # del sys
    # sys = load_system('../../testing/test-fit.p')
    # sys_data_predict2 = sys.apply_experiment(sys_data)

    test.plot()
    sys_data_predict.plot(show=True)
    # sys_data_predict2.plot(show=True)

