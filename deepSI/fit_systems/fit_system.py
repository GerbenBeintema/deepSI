
from deepSI.systems.system import System, System_io, System_data, load_system, System_bj
import numpy as np
from deepSI.datasets import get_work_dirs
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
        self._fit(self.norm.transform(sys_data), **kwargs) #transform data to fittable data?
        self.fitted = True


class System_torch(System_fittable):
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


    def fit(self, sys_data, epochs=30, batch_size=256, Loss_kwargs={}, \
        optimizer_kwargs={}, sim_val=None, verbose=1, cuda=False, val_frac=0.2, sim_val_fun='NRMS'):
        #todo implement verbose

        #1. init funcs already happened
        #2. init optimizer
        #3. training data
        #4. optimization

        def validation(append=True):
            self.eval(); self.cpu();
            global time_val
            t_start_val = time.time()
            if sim_val is not None:
                sim_val_predict = self.apply_experiment(sim_val)
                #I can transform sim_val_predict and sim_val with self.norm for a controllable NRMS
                
                Loss_val = sim_val_predict.__getattribute__(sim_val_fun)(sim_val)
            else:
                with torch.no_grad():
                    Loss_val = self.loss(*data_val,**Loss_kwargs).item()
            time_val += time.time() - t_start_val
            if append: self.Loss_val.append(Loss_val) 
            if self.bestfit>Loss_val:
                if verbose: print(f'########## new best ########### {Loss_val}')
                self.checkpoint_save_system()
                self.bestfit = Loss_val
            if cuda: 
                self.cuda()
            self.train()
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
            self.Loss_val, self.Loss_train, self.batch_id, self.time = np.array([]), np.array([]), np.array([]), np.array([])
            self.batch_counter = 0
            extra_t = 0
            self.fitted = True
        else:
            self.batch_counter = 0 if len(self.batch_id)==0 else self.batch_id[-1]
            extra_t = 0 if len(self.time)==0 else self.time[-1] #correct time counting after reset


        sys_data, sys_data0 = self.norm.transform(sys_data), sys_data
        data_full = self.make_training_data(sys_data, **Loss_kwargs)
        data_full = [torch.tensor(dat, dtype=torch.float32) for dat in data_full] #add cuda?


        if sim_val is not None:
            data_train = data_full
        else: #is not used that often, could use sklearn to split data
            split = int(len(data_full[0])*(1-val_frac))
            data_train = [dat[:split] for dat in data_full]
            data_val = [dat[split:] for dat in data_full]

        self.Loss_val, self.Loss_train, self.batch_id, self.time = list(self.Loss_val), list(self.Loss_train), list(self.batch_id), list(self.time)

        global time_val, time_loss, time_back #time keeping
        time_val = time_back = time_loss = 0
        Loss_val = validation(append=False) #Also switches to cuda if indicated
        time_val = 0 #reset
        N_training_samples = len(data_train[0])
        batch_size = min(batch_size, N_training_samples)
        N_batch_updates_per_epoch = N_training_samples//batch_size
        if verbose>0: print(f'N_training_samples={N_training_samples}, batch_size={batch_size}, N_batch_updates_per_epoch={N_batch_updates_per_epoch}')
        ids = np.arange(0, N_training_samples, dtype=int)
        try:
            self.start_t = time.time()
            for epoch in (tqdm(range(epochs)) if verbose>0 else range(epochs)):
                np.random.shuffle(ids)

                Loss_acc = 0
                for i in range(batch_size, N_training_samples + 1, batch_size):
                    ids_batch = ids[i-batch_size:i]
                    train_batch = [(part[ids_batch] if not cuda else part[ids_batch].cuda()) for part in data_train] #add cuda?

                    def closure(backward=True):
                        global time_loss, time_back
                        start_t_loss = time.time()
                        Loss = self.loss(*train_batch, **Loss_kwargs)
                        time_loss += time.time() - start_t_loss
                        if backward:
                            self.optimizer.zero_grad()
                            start_t_back = time.time()
                            Loss.backward()
                            time_back += time.time() - start_t_back
                        return Loss

                    Loss = self.optimizer.step(closure)
                    Loss_acc += Loss.item()
                self.batch_counter += N_batch_updates_per_epoch
                Loss_acc /= N_batch_updates_per_epoch
                self.Loss_train.append(Loss_acc)
                self.time.append(time.time()-self.start_t+extra_t)
                self.batch_id.append(self.batch_counter)
                Loss_val = validation()
                if verbose>0: 
                    time_elapsed = time.time()-self.start_t
                    # print('train Loss:',self.loss(*data_train, **Loss_kwargs).item(), 'val:',self.loss(*data_val, **Loss_kwargs).item() )
                    print(f'Epoch: {epoch+1:4} Training loss: {self.Loss_train[-1]:7.4} Validation loss = {Loss_val:6.4}, time Loss: {time_loss/time_elapsed:.1%}, back: {time_back/time_elapsed:.1%}, val: {time_val/time_elapsed:.1%}')
                # print(f'epoch={epoch} NRMS={Loss_val:9.4%} Loss={Loss_acc:.5f}')
        except KeyboardInterrupt:
            print('stopping early due to KeyboardInterrupt')
        self.train(); self.cpu();
        self.Loss_val, self.Loss_train, self.batch_id, self.time = np.array(self.Loss_val), np.array(self.Loss_train), np.array(self.batch_id), np.array(self.time)
        self.checkpoint_save_system(name='_last')
        self.checkpoint_load_system()

    def loss(*args,**kwargs):
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
        # self.Loss_val, self.Loss_train, self.batch_id, self.time = self.Loss_val, self.Loss_train, self.batch_id, self.time
        
    def _save_system_torch(self, file):
        save_dict = {}
        for d in dir(self):
            if d in ['random']: #exclude random
                continue
            attribute = self.__getattribute__(d)
            if isinstance(attribute,(nn.Module,optim.Optimizer)):
                save_dict[d] = attribute.state_dict()
        torch.save(save_dict,file)
    def _load_system_torch(self, file):
        save_dict = torch.load(file)
        for key in save_dict:
            attribute = self.__getattribute__(key)
            try:
                attribute.load_state_dict(save_dict[key])
            except (AttributeError, ValueError):
                print('Error loading key',key)

    ### CPU & CUDA ###
    def cuda(self):
        self.to_device('cuda')
    def cpu(self):
        self.to_device('cpu')
    def to_device(self,device):
        for d in dir(self):
            attribute = self.__getattribute__(d)
            if isinstance(attribute,nn.Module):
                attribute.to(device)

    def eval(self):
        for d in dir(self):
            attribute = self.__getattribute__(d)
            if isinstance(attribute,nn.Module):
                attribute.eval()
    def train(self):
        for d in dir(self):
            attribute = self.__getattribute__(d)
            if isinstance(attribute,nn.Module):
                attribute.train()

if __name__ == '__main__':
    pass