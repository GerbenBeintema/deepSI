
from deepSI.systems.System import System, System_IO, System_data, load_system,System_BJ
import numpy as np
from deepSI.system_data.datasets import get_work_dirs
import deepSI
import torch
from torch import nn, optim
from tqdm.auto import tqdm
import time

class System_fittable(System):
    """docstring for System_fit"""
    def fit(self, sys_data, **kwargs):
        if self.fitted==False:
            if self.use_norm: #if the norm is not used you can also manually initialize it.
                #you may consider not using the norm if you have constant values in your training data which can change. They are known to cause quite a number of bugs and errors. 
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
    def init_nets(self,nu,ny):
        #returns parameters
        raise NotImplementedError

    def init_optimizer(self,parameters,**optimizer_kwargs):
        #return the optimizer with a optimizer.zero_grad and optimizer.step method
        if optimizer_kwargs.get('optimizer') is not None:
            from copy import deepcopy
            optimizer_kwargs = deepcopy(optimizer_kwargs) #do not modify the original kwargs, is this necessary
            optimizer = optimizer_kwargs['optimizer']
            del optimizer_kwargs['optimizer']
        else:
            optimizer = torch.optim.Adam
        return optimizer(parameters,**optimizer_kwargs) 

    def make_training_data(self,sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        #should be implemented in child
        raise NotImplementedError


    def fit(self, sys_data, epochs=30, batch_size=256, Loss_kwargs={}, optimizer_kwargs={}, sim_val=None, verbose=1, val_frac=0.2, sim_val_fun='NRMS'):
        #todo implement verbose

        #1. init funcs already happened
        #2. init optimizer
        #3. training data
        #4. optimization

        def validation(append=True):
            global time_val
            t_start_val = time.time()
            if sim_val is not None:
                sim_val_predict = self.apply_experiment(sim_val)
                #I can transform sim_val_predict and sim_val with self.norm for a controllable NRMS
                
                Loss_val = sim_val_predict.__getattribute__(sim_val_fun)(sim_val)
            else:
                with torch.no_grad():
                    Loss_val = self.CallLoss(*data_val,**Loss_kwargs).item()
            time_val += time.time() - t_start_val
            if append: self.Loss_val.append(Loss_val) 
            if self.bestfit>Loss_val:
                if verbose: print(f'########## new best ########### {Loss_val}')
                self.checkpoint_save_system()
                self.bestfit = Loss_val
            return Loss_val

        if self.fitted==False:
            if self.use_norm: #if the norm is not used you can also manually initialize it.
                #you may consider not using the norm if you have constant values in your training data which can change. They are known to cause quite a number of bugs and errors. 
                self.norm.fit(sys_data)
            self.nu = sys_data.nu
            self.ny = sys_data.ny
            self.paremters = list(self.init_nets(self.nu,self.ny))
            self.optimizer = self.init_optimizer(self.paremters,**optimizer_kwargs)
            self.bestfit = float('inf')
            self.Loss_val, self.Loss_train, self.batch_id, self.time = [], [], [], []
            self.batch_counter = 0
            extra_t = 0
            self.fitted = True
        else:
            self.batch_counter = 0 if len(self.batch_id)==0 else self.batch_id[-1]
            extra_t = 0 if len(self.time)==0 else self.time[-1]


        sys_data, sys_data0 = self.norm.transform(sys_data), sys_data
        data_full = self.make_training_data(sys_data, **Loss_kwargs)
        data_full = [torch.tensor(dat, dtype=torch.float32) for dat in data_full] #add cuda?


        if sim_val is not None:
            data_train = data_full
        else: #is not used that often, could use sklearn to split data
            split = int(len(data_full[0])*(1-val_frac))
            data_train = [dat[:split] for dat in data_full]
            data_val = [dat[split:] for dat in data_full]


        global time_val
        time_val = time_back = time_loss = 0
        Loss_val = validation(append=False) #add to cpu/to cuda in validation?
        time_val = 0 #reset
        N_training_samples = len(data_train[0])
        batch_size = min(batch_size, N_training_samples)
        N_batch_updates_per_epoch = N_training_samples//batch_size
        if verbose>0: print(f'N_training_samples={N_training_samples}, batch_size={batch_size}, N_batch_updates_per_epoch={N_batch_updates_per_epoch}')
        ids = np.arange(0, N_training_samples, dtype=int)
        try:
            self.start_t = time.time()
            for epoch in tqdm(range(epochs)):
                np.random.shuffle(ids)

                Loss_acc = 0
                for i in range(batch_size, N_training_samples + 1, batch_size):
                    ids_batch = ids[i-batch_size:i]
                    train_batch = [part[ids_batch] for part in data_train] #add cuda?
                    start_t_loss = time.time()
                    Loss = self.CallLoss(*train_batch, **Loss_kwargs)
                    time_loss += time.time() - start_t_loss

                    self.optimizer.zero_grad()

                    start_t_back = time.time()
                    Loss.backward()
                    time_back += time.time() - start_t_back

                    self.optimizer.step()
                    Loss_acc += Loss.item()
                self.batch_counter += N_batch_updates_per_epoch
                Loss_acc /= N_batch_updates_per_epoch
                self.Loss_train.append(Loss_acc)
                self.time.append(time.time()-self.start_t+extra_t)
                self.batch_id.append(self.batch_counter)
                Loss_val = validation()
                if verbose>0: 
                    time_elapsed = time.time()-self.start_t
                    # print('train Loss:',self.CallLoss(*data_train, **Loss_kwargs).item(), 'val:',self.CallLoss(*data_val, **Loss_kwargs).item() )
                    print(f'Epoch: {epoch+1:4} Training loss: {self.Loss_train[-1]:7.4} Validation loss = {Loss_val:6.4}, time Loss: {time_loss/time_elapsed:.1%}, back: {time_back/time_elapsed:.1%}, val: {time_val/time_elapsed:.1%}')
                # print(f'epoch={epoch} NRMS={Loss_val:9.4%} Loss={Loss_acc:.5f}')
        except KeyboardInterrupt:
            print('stopping early due to KeyboardInterrupt')
        self.checkpoint_save_system(name='_last')
        self.checkpoint_load_system()

    def CallLoss(*args,**kwargs):
        #kwargs are the settings
        #args is the data
        raise NotImplementedError
    
    ########## Saving and loading ############
    def checkpoint_save_system(self,name='_best'):
        from pathlib import Path
        import os.path
        directory  = get_work_dirs()['checkpoints']
        self._save_system_torch(file=os.path.join(directory,self.name+name+'.pth')) #error here if you have 
        vars = self.norm.u0, self.norm.ustd, self.norm.y0, self.norm.ystd, self.fitted, self.bestfit, self.Loss_val, self.Loss_train, self.batch_id, self.time
        np.savez(os.path.join(directory,self.name+name+'.npz'),*vars)
    def checkpoint_load_system(self,name='_best'):
        from pathlib import Path
        import os.path
        directory  = get_work_dirs()['checkpoints'] 
        self._load_system_torch(file=os.path.join(directory,self.name+name+'.pth'))
        out = np.load(os.path.join(directory,self.name+name+'.npz'))
        out_real = [(a[1].tolist() if a[1].ndim==0 else a[1]) for a in out.items()]
        self.norm.u0, self.norm.ustd, self.norm.y0, self.norm.ystd, self.fitted, self.bestfit, self.Loss_val, self.Loss_train, self.batch_id, self.time = out_real
        self.Loss_val, self.Loss_train, self.batch_id, self.time = self.Loss_val.tolist(), self.Loss_train.tolist(), self.batch_id.tolist(), self.time.tolist()
        
    def _save_system_torch(self,file):
        save_dict = {}
        for d in dir(self):
            attribute = self.__getattribute__(d)
            if isinstance(attribute,(nn.Module,optim.Optimizer)):
                save_dict[d] = attribute.state_dict()
        torch.save(save_dict,file)
    def _load_system_torch(self,file):
        save_dict = torch.load(file)
        for key in save_dict:
            attribute = self.__getattribute__(key)
            try:
                attribute.load_state_dict(save_dict[key])
            except (AttributeError, ValueError):
                print('Error loading key',key)


def System_BJ_fittable(System_BJ,System_PyTorch):
    #make data
    #call Loss
    def CallLoss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        #order: u,yhat,yreal
        Loss = torch.zeros(1,dtype=yhist.dtype,device=yhist.device)[0]
        yhisthat = yhist[:,yhist.shape[1]-self.nb:] #nb
        yhistreal = yhist[:,yhist.shape[1]-self.nc:] #nc

        for unow, ynow in zip(torch.transpose(ufuture,0,1), torch.transpose(yfuture,0,1)): #unow = (Nsamples, nu), ynow = (Nsamples, ny)
            g_in = torch.cat([torch.flatten(uhist, start_dim=1), 
                            torch.flatten(yhisthat, start_dim=1), 
                            torch.flatten(yhistreal, start_dim=1)],axis=1)
            yout = self.gn(g_in) #(Nsamples, ny)
            Loss += torch.mean((yout - ynow)**2)**0.5 #possible to skip here
            uhist = torch.cat((uhist[:,1:,:],unow[:,None,:]),dim=1)
            yhistreal = torch.cat((yhistreal[:,1:,:],ynow[:,None,:]),dim=1)
            yhisthat = torch.cat((yhisthat[:,1:,:],yout[:,None,:]),dim=1)
        Loss /= ufuture.shape[1]
        return Loss

    def make_training_data(self, sys_data, **Loss_kwargs):
        assert sys_data.normed == True
        nf = Loss_kwargs.get('nf',25)
        return sys_data.to_hist_future_data(na=self.na,nb=max(self.nc,self.nb), nf=nf, force_multi_u=True, force_multi_y=True) #returns np.array(uhist),np.array(yhist),np.array(ufuture),np.array(yfuture)

    def init_nets(self,nu,ny):
        #returns parameters
        raise NotImplementedError

def System_BJ_full(System_BJ_fittable):
    #y = B/F u + C/D v
    #yhat = BD u + (C-D)*yreal + (1-CF)*yhat
    #F = 1 + .. #nF
    #C = 1 + ..
    #D = 1 + ..
    #B = 0 + ..
    #
    #C monic
    #F monic
    def __init__(self,nB,nF,nC,nD):
        #monic?
        #na = length of y hat = nC*nF - 1
        #nb = length of u = nB*nD
        #nc = length of y real = max(nC,nB)
        super(System_BJ_full,self).__init__(nC*nF-1 , nB*nD,max(nC,nB))

        




import torch
from torch import nn

def process_dict(search_dict):
    #this is not foul proof so watch out. (e.g. passing lists as items when the whole list should be passed to the function)
    itter_dict = [list(a) for a in search_dict.items()]
    for k in range(len(itter_dict)):
        I = itter_dict[k][1]
        if isinstance(I,range):
            itter_dict[k][1] = list(I)
        elif not isinstance(I,(tuple,list,np.ndarray)):
            itter_dict[k][1] = [I]
    return itter_dict


def grid_search(fit_system, sys_data, sys_dict_choices={}, fit_dict_choices={}, sim_val=None, RMS=False, verbose=2):
    import copy
    sim_val = sys_data if sim_val is None else sim_val
    #example use: print(hyper_parameter_tunner(System_IO_fit_linear,dict(na=[1,2,3],nb=[1,2,3]),sys_data))
    def itter(sys_dict_choices, fit_dict_choices, sys_dict=None, fit_dict=None, bests=None, best_score=float('inf'), best_sys=None, best_sys_dict=None, best_fit_dict=None):
        if sys_dict is None:
            sys_dict,fit_dict = dict(), dict()
        length_sys_dict, length_fit_dict = len(sys_dict), len(fit_dict)

        if length_sys_dict!=len(sys_dict_choices):
            key,items = sys_dict_choices[length_sys_dict][0], sys_dict_choices[length_sys_dict][1]
            for item in items:
                sys_dict[key] = item
                best_score, best_sys, best_sys_dict, best_fit_dict = itter(sys_dict_choices, fit_dict_choices, sys_dict=sys_dict, fit_dict=fit_dict, best_score=best_score, best_sys=best_sys, best_sys_dict=best_sys_dict, best_fit_dict=best_fit_dict)
            del sys_dict[key]
            return best_score, best_sys, best_sys_dict, best_fit_dict
        elif length_fit_dict!=len(fit_dict_choices):
            key,items = fit_dict_choices[length_fit_dict][0], fit_dict_choices[length_fit_dict][1]
            for item in items:
                fit_dict[key] = item
                best_score, best_sys, best_sys_dict, best_fit_dict = itter(fit_dict_choices, fit_dict_choices, sys_dict=sys_dict, fit_dict=fit_dict, best_score=best_score, best_sys=best_sys, best_sys_dict=best_sys_dict, best_fit_dict=best_fit_dict)
            del fit_dict[key]
            return best_score, best_sys, best_sys_dict, best_fit_dict
        else:

            try: #fit the system if possible
                sys = fit_system(**sys_dict)
                sys.fit(sys_data,**fit_dict)
            except Exception as inst:
                print('error',inst)
                print('for sys_dict=', sys_dict)
                print('for fit_dict=', fit_dict)
            try:
                score = sys.apply_experiment(sim_val).RMS(sim_val) if RMS is None else sys.apply_experiment(sim_val).NRMS(sim_val)
            except ValueError:
                score = float('inf')
            if verbose>1: print(score, sys_dict, fit_dict)
            if score<=best_score:
                return score, sys, copy.deepcopy(sys_dict), copy.deepcopy(fit_dict)
            else:
                return best_score, best_sys, best_sys_dict, best_fit_dict

    sys_dict_choices, fit_dict_choices = process_dict(sys_dict_choices), process_dict(fit_dict_choices)
    best_score, best_sys, best_sys_dict, best_fit_dict = itter(sys_dict_choices, fit_dict_choices)
    if verbose>0: print('Result:', best_score, best_sys, best_sys_dict, best_fit_dict)
    return best_score, best_sys, best_sys_dict, best_fit_dict

from random import choice
def sample_dict(dict_now):
    #dict = {'k':[1,2,3],'a':[True,False],}
    #goes to a random choice from the given list
    return dict([(key,choice(item)) for key,item in dict_now.items()])

# I can multi process this function in its entirety in the future
def random_search(fit_system, sys_data, sys_dict_choices={}, fit_dict_choices={}, sim_val=None, RMS=False, budget=20):
    sim_val = sys_data if sim_val is None else sim_val
    def test(sys_dict, fit_dict):
        sys = fit_system(**sys_dict)
        try:
            sys.fit(sys_data,**fit_dict)
        except Exception as inst:
            print('error',inst)
            print('for sys_dict=', sys_dict)
            print('for fit_dict=', fit_dict)
        try:
            score = sys.apply_experiment(sim_val).RMS(sim_val) if RMS is None else sys.apply_experiment(sim_val).NRMS(sim_val)
        except ValueError:
            score = float('inf')
        return sys, score

    all_results = []
    sys_dict_choices, fit_dict_choices = dict(process_dict(sys_dict_choices)), dict(process_dict(fit_dict_choices))
    for trial in tqdm(range(budget)):
        sys_dict, fit_dict = sample_dict(sys_dict_choices), sample_dict(fit_dict_choices)
        sys, score = test(sys_dict, fit_dict)
        all_results.append(dict(sys_dict=sys_dict,fit_dict=fit_dict,sys=sys,score=score))
    return all_results


#GP search may be possible



if __name__ == '__main__':
    from sklearn import linear_model 
    from matplotlib import pyplot as plt
    class System_IO_fit_linear(System_IO_fit_sklearn):
        def __init__(self,na,nb):
            super(System_IO_fit_linear,self).__init__(na,nb,linear_model.LinearRegression())
    # class System_IO_fit_linear(System_IO_fit_sklearn):
    #     def __init__(self,na,nb,alpha=0.1):
    #         super(System_IO_fit_linear,self).__init__(na,nb,linear_model.Lasso(alpha=alpha))

    train, test = deepSI.datasets.Cascaded_Tanks()
    # sys = deepSI.fit_systems.System_IO_pytorch(na=5,nb=5)

    # all_results = random_search(fit_system=deepSI.fit_systems.System_IO_pytorch, sys_data=train, sys_dict_choices=dict(na=[1,2,3,4,5],nb=[1,2,3,4]),fit_dict_choices=dict(sim_val=[train],verbose=[1]),budget=5)
    # print(all_results)

    # result = grid_search(System_IO_fit_linear, train, sys_dict_choices=dict(na=range(1,10),nb=range(1,10)), fit_dict_choices={}, sim_val=test, RMS=False, verbose=1)
    result = random_search(System_IO_fit_linear, train, sys_dict_choices=dict(na=range(1,10),nb=np.arange(1,10)), fit_dict_choices={}, sim_val=test, RMS=False, budget=200)
    #0.30908989649451607 {'na': 6, 'nb': 8, 'alpha': 0.0004804870439655134}
    #0.29513191231139657

    # fit_system, sys_data, sys_dict_choices={}, fit_dict_choices={}, sim_val=None, RMS=False, budget=20
    print(min(result,key=lambda x: x['score']))
    print(result)
    # sys = System_encoder(nx=8, na=50, nb=50)
    # sys0 = deepSI.systems.sys_ss_test()
    # sys_data = sys0.apply_experiment(System_data(u=np.random.normal(size=10000)))

    # sys = System_IO_fit_linear(7,3)
    # sys = System_encoder(nx=8,na=20,nb=20)
    # sys_data = System_data(u=np.random.normal(size=100),y=np.random.normal(size=100))
    # sys.fit(train,epochs=1000,Loss_kwargs=dict(nf=15),batch_size=8,sim_val=None,optimizer_kwargs=dict(optimizer=optim.Adam,lr=1e-3))
    # print(sys.optimizer)
    # sys.save_system('../../testing/test-fit.p')
    # del sys
    # sys = load_system('../../testing/test-fit.p')

    # sys_data_predict = sys.apply_experiment(test)
    # print(sys_data_predict.NRMS(test))
    # plt.plot(sys.n_step_error(test))
    # plt.show()

    # sys_data_predict2 = sys.apply_experiment(sys_data)

    # test.plot()
    # sys_data_predict.plot(show=True)
    # plt.plot(test.u)
    # plt.show()
    # sys_data_predict2.plot(show=True)

