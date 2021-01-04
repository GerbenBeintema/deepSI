
from deepSI.systems.system import System, System_io, System_data, load_system, System_bj
import numpy as np
from deepSI.datasets import get_work_dirs
import deepSI
import torch
from torch import nn, optim
from tqdm.auto import tqdm
import time
from pathlib import Path
import os.path

class System_fittable(System):
    """Subclass of system which introduces a .fit method which calls ._fit to fit the systems

    Notes
    -----
    This function will automaticly fit the normalization in self.norm if self.use_norm is set to True (default). 
    Lastly it will set self.fitted to True which will keep the norm constant. 
    """
    def fit(self, sys_data, **kwargs):
        if self.fitted==False:
            if self.use_norm: #if the norm is not used you can also manually initialize it.
                #you may consider not using the norm if you have constant values in your training data which can change. They are known to cause quite a number of bugs and errors. 
                self.norm.fit(sys_data)
            self.nu = sys_data.nu
            self.ny = sys_data.ny
        self._fit(self.norm.transform(sys_data), **kwargs)
        self.fitted = True

    def _fit(self, normed_sys_data, **kwargs):
        raise NotImplementedError('_fit or fit should be implemented in subclass')


class System_torch(System_fittable):
    '''For systems that utilize torch

    Attributes
    ----------
    parameters : list
        The list of fittable network parameters returned by System_torch.init_nets(nu,ny)
    optimizer : torch.Optimizer
        The main optimizer returned by System_torch.init_optimizer
    time : numpy.ndarray
        Current runtime after each epoch
    batch_id : numpy.ndarray
        Current total number of batch optimization steps is saved after each epoch
    Loss_train : numpy.ndarray
        Average training loss for each epoch
    Loss_val : numpy.ndarray
        Validation loss for each epoch

    Notes
    -----
    subclasses should define three methods
    (i) init_nets(nu, ny) which returns the network parameters, 
    (ii) make_training_data(sys_data, **loss_kwargs)` which converts the normed sys_data into training data (list of numpy arrays),
    (iii) loss(*training_data, **loss_kwargs) which returns the loss using the current training data
    '''
    def init_nets(self, nu, ny):
        '''Defined in subclass and initializes networks and returns the parameters

        Parameters
        ----------
        nu : None, int or tuple
            The shape of the input u
        ny : None, int or tuple
            The shape of the output y

        Returns
        -------
        parameters : list
            List of the network parameters
        '''
        raise NotImplementedError('init_nets should be implemented in subclass')

    def make_training_data(self, sys_data, **loss_kwargs):
        '''Defined in subclass which converts the normed sys_data into training data

        Parameters
        ----------
        sys_data : System_data or System_data_list
            Already normalized
        loss_kwargs : dict
            loss function settings passed into .fit
        '''
        assert sys_data.normed == True
        raise NotImplementedError('make_training_data should be implemented in subclass')

    def loss(*training_data_batch, **loss_kwargs):
        '''Defined in subclass which take the batch data and calculates the loss based on loss_kwargs

        Parameters
        ----------
        training_data_batch : list
            batch of the training data returned by make_training_data and converted to torch arrays
        loss_kwargs : dict
            loss function settings passed into .fit
        '''
        raise NotImplementedError('loss should be implemented in subclass')

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
        if optimizer_kwargs.get('optimizer') is not None:
            from copy import deepcopy
            optimizer_kwargs = deepcopy(optimizer_kwargs) #do not modify the original kwargs, is this necessary
            optimizer = optimizer_kwargs['optimizer']
            del optimizer_kwargs['optimizer']
        else:
            optimizer = torch.optim.Adam
        return optimizer(parameters,**optimizer_kwargs) 

    def fit(self, sys_data, epochs=30, batch_size=256, loss_kwargs={}, \
        optimizer_kwargs={}, sim_val=None, verbose=1, cuda=False, val_frac=0.2, sim_val_fun='NRMS'):
        '''The default batch optimization method 

        Parameters
        ----------
        sys_data : System_data or System_data_list
            the system data to be fitted
        epochs : int
        batch_size : int
        loss_kwargs : dict
            kwargs to be passed on to make_training_data and init_optimizer
        optimizer_kwargs : dict
            kwargs to be passed on to init_optimizer
        sim_val : System_data or System_data_list
            the system data to be used as simulation validation using apply_experiment
        verbose : int
            set to 0 for a silent run
        cuda : bool
            if cuda will be used (often slower than not using it, be aware)
        val_frac : float
            if sim_val is absent a portion will be splitted from the training data to act as validation set using the loss method.
        sim_val_fun : str
            method on system_data invoked if sim_val is used.
        
        Notes
        -----
        This method implements a batch optimization method in the following way; each epoch the training data is scrambled and batched where each batch
        is passed to the loss method and utilized to optimize the parameters. After each epoch the systems is validated using the evaluation of a 
        simulation or a validation split and saved if a new lowest validation loss has been achieved. 
        After training (which can be stopped at any moment using a KeyboardInterrupt) the system is loaded with the lowest validation loss. 
        '''
        def validation(append=True):
            self.eval(); self.cpu();
            global time_val
            t_start_val = time.time()
            if sim_val is not None:
                sim_val_predict = self.apply_experiment(sim_val)
                Loss_val = sim_val_predict.__getattribute__(sim_val_fun)(sim_val)
            else:
                with torch.no_grad():
                    Loss_val = self.loss(*data_val,**loss_kwargs).item()
            time_val += time.time() - t_start_val
            if append: self.Loss_val.append(Loss_val) 
            if self.bestfit>Loss_val:
                if verbose: print(f'########## New lowest validation loss achieved ########### {Loss_val}')
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
            self.parameters = list(self.init_nets(self.nu,self.ny))
            self.optimizer = self.init_optimizer(self.parameters,**optimizer_kwargs)
            self.bestfit = float('inf')
            self.Loss_val, self.Loss_train, self.batch_id, self.time = np.array([]), np.array([]), np.array([]), np.array([])
            self.batch_counter = 0
            extra_t = 0
            self.fitted = True
        else:
            self.batch_counter = 0 if len(self.batch_id)==0 else self.batch_id[-1]
            extra_t = 0 if len(self.time)==0 else self.time[-1] #correct time counting after reset


        sys_data, sys_data0 = self.norm.transform(sys_data), sys_data
        data_full = self.make_training_data(sys_data, **loss_kwargs)
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
                        Loss = self.loss(*train_batch, **loss_kwargs)
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
                    print(f'Epoch: {epoch+1:4} Training loss: {self.Loss_train[-1]:7.4} Validation loss = {Loss_val:6.4}, time Loss: {time_loss/time_elapsed:.1%}, back: {time_back/time_elapsed:.1%}, val: {time_val/time_elapsed:.1%}')
        except KeyboardInterrupt:
            print('stopping early due to KeyboardInterrupt')
        self.train(); self.cpu();
        self.Loss_val, self.Loss_train, self.batch_id, self.time = np.array(self.Loss_val), np.array(self.Loss_train), np.array(self.batch_id), np.array(self.time)
        self.checkpoint_save_system(name='_last')
        self.checkpoint_load_system()
    
    ########## Saving and loading ############
    def checkpoint_save_system(self, name='_best', directory=None):
        directory  = get_work_dirs()['checkpoints'] if directory is None else directory
        self._save_system_torch(file=os.path.join(directory,self.name+name+'.pth')) #error here if you have 
        vars = self.norm.u0, self.norm.ustd, self.norm.y0, self.norm.ystd, self.fitted, self.bestfit, self.Loss_val, self.Loss_train, self.batch_id, self.time
        np.savez(os.path.join(directory,self.name+name+'.npz'),*vars)
    def checkpoint_load_system(self, name='_best', directory=None):
        directory  = get_work_dirs()['checkpoints'] if directory is None else directory
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