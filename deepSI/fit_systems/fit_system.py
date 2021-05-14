
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
from torch.utils.data import Dataset, DataLoader

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

        Returns
        -------
        data : list or torch.utils.data.Dataset
            a list of arrays (e.g. [X,Y]) or an instance of torch.utils.data.Dataset
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

    def fit(self, sys_data, epochs=30, batch_size=256, loss_kwargs={}, optimizer_kwargs={}, \
            sim_val=None, concurrent_val=False, timeout=None, verbose=1, cuda=False, val_frac=0.2, \
            sim_val_fun='NRMS', sqrt_train=True, loss_val=None, num_workers_data_loader=0, \
            print_full_time_profile=False):
        '''The batch optimization method with parallel validation, 

        Parameters
        ----------
        sys_data : System_data or System_data_list
            The system data to be fitted
        epochs : int
        batch_size : int
        loss_kwargs : dict
            Kwargs to be passed to make_training_data and loss.
        optimizer_kwargs : dict
            Kwargs to be passed on to init_optimizer.
        sim_val : System_data or System_data_list
            The system data to be used as simulation validation using apply_experiment (if absent than a portion of the training data will be used)
        concurrent_val : boole
            If set to true a subprocess will be started which concurrently evaluates the validation method selected.
            Warning: if concurrent_val is set than "if __name__=='__main__'" or import from a file if using self defined method or networks.
        timeout : None or number
            Alternative to epochs to run until a set amount of time has past. 
        verbose : int
            Set to 0 for a silent run
        cuda : bool
            if cuda will be used (often slower than not using it, be aware)
        val_frac : float
            if sim_val is absent a portion will be splitted from the training data to act as validation set using the loss method.
        sim_val_fun : str
            method on system_data invoked if sim_val is used.
        sqrt_train : boole
            will sqrt the loss while printing
        
        Notes
        -----
        This method implements a batch optimization method in the following way; each epoch the training data is scrambled and batched where each batch
        is passed to the loss method and utilized to optimize the parameters. After each epoch the systems is validated using the evaluation of a 
        simulation or a validation split and a checkpoint will be crated if a new lowest validation loss has been achieved. (or concurrently if concurrent_val is set)
        After training (which can be stopped at any moment using a KeyboardInterrupt) the system is loaded with the lowest validation loss. 

        The default checkpoint location is "C:/Users/USER/AppData/Local/deepSI/checkpoints" for windows and ~/.deepSI/checkpoints/ for unix like.
        These can be loaded manually using sys.load_checkpoint("_best") or "_last". (For this to work the sys.unique_code needs to be set to the correct string)
        '''
        def validation(append=True, train_loss=None, time_elapsed_total=None):
            self.eval(); self.cpu();
            if sim_val is not None:
                sim_val_predict = self.apply_experiment(sim_val)
                Loss_val = sim_val_predict.__getattribute__(sim_val_fun)(sim_val)
            else:
                with torch.no_grad():
                    Loss_val = self.loss(*data_val,**loss_kwargs).item()
            if append: 
                self.Loss_val.append(Loss_val) 
                self.Loss_train.append(train_loss)
                self.time.append(time_elapsed_total)
                self.batch_id.append(self.batch_counter)
                self.epoch_id.append(self.epoch_counter)
            if self.bestfit>=Loss_val:
                self.bestfit = Loss_val
                self.checkpoint_save_system()
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
            self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
            self.fitted = True

        self.epoch_counter = 0 if len(self.epoch_id)==0 else self.epoch_id[-1]
        self.batch_counter = 0 if len(self.batch_id)==0 else self.batch_id[-1]
        extra_t            = 0 if len(self.time)    ==0 else self.time[-1] #correct time counting after restart

        sys_data = self.norm.transform(sys_data)
        data_full = self.make_training_data(sys_data, **loss_kwargs) #this can return a list of numpy arrays or a torch.utils.data.Dataset instance
        if not isinstance(data_full, Dataset) and verbose:
            Dsize = sum([d.nbytes for d in data_full])
            if Dsize>2**30: 
                dstr = f'{Dsize/2**30:.1f} GB!'
                dstr += '\nConsider using pre_construct=False or let make_training_data return a Dataset to reduce data-usage'
            elif Dsize>2**20: 
                dstr = f'{Dsize/2**20:.1f} MB'
            else:
                dstr = f'{Dsize/2**10:.1f} kB'
            print('Size of the training array = ', dstr)


        if sim_val is not None:
            data_train = data_full
            data_val = None
        elif loss_val is not None: #is not used that often
            data_train = data_full
            data_val = [torch.tensor(a, dtype=torch.float32) for a in self.make_training_data(self.norm.transform(loss_val), **loss_kwargs)]
        else: #split off some data from the main dataset
            from sklearn.model_selection import train_test_split
            assert isinstance(data_full, Dataset)==False, 'not yet implemented, give a sim_val or loss_val dataset to '
            datasplitted = [torch.tensor(a, dtype=torch.float32) for a in train_test_split(*data_full,random_state=42)] # (A1_train, A1_test, A2_train, A2_test)
            data_train = [datasplitted[i] for i in range(0,len(datasplitted),2)]
            data_val = [datasplitted[i] for i in range(1,len(datasplitted),2)]

        #transforming it back to a list to be able to append.
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = list(self.Loss_val), list(self.Loss_train), list(self.batch_id), list(self.time), list(self.epoch_id)

        Loss_acc_val, N_batch_acc_val = 0, 0
        N_training_samples = len(data_train) if isinstance(data_train, Dataset) else len(data_train[0])
        batch_size = min(batch_size, N_training_samples)
        N_batch_updates_per_epoch = N_training_samples//batch_size
        if verbose>0: 
            print(f'N_training_samples = {N_training_samples}, batch_size = {batch_size}, N_batch_updates_per_epoch = {N_batch_updates_per_epoch}')
        val_counter = 0  #to print the frequency of the validation step.
        val_str = sim_val_fun if sim_val is not None else 'loss' #for correct printing
        best_epoch = 0
        batch_id_start = self.batch_counter
        
        if isinstance(data_train, Dataset):
            persistent_workers = False if num_workers_data_loader==0 else True
            data_train_loader = DataLoader(data_train, batch_size=batch_size, drop_last=True, shuffle=True, \
                                   num_workers=num_workers_data_loader, persistent_workers=persistent_workers)
        else: #add my basic DataLoader
            #slow old way
            # data_train_loader = DataLoader(default_dataset(data_train), batch_size=batch_size, drop_last=True, shuffle=True, num_workers=num_workers_data_loader)
            #fast new way
            data_train_loader = My_Simple_DataLoader(data_train, batch_size=batch_size) #is quite a bit faster for low data situations

        if concurrent_val:
            from multiprocessing import Process, Pipe
            from copy import deepcopy
            remote, work_remote = Pipe()
            remote.receiving = False
            process = Process(target=_worker, args=(work_remote, remote, sim_val, data_val, sim_val_fun, loss_kwargs))
            process.daemon = True  # if the main process crashes, we should not cause things to hang
            process.start()
            work_remote.close()
            remote.send((deepcopy(self), True, float('nan'), extra_t)) #time here does not matter
            #            sys, append, Loss_train, time_now
            remote.receiving = True
            #           sys, append, Loss_acc, time_now, epoch
            Loss_val_now = float('nan')
        else: #do it now
            Loss_val_now = validation(append=True, train_loss=float('nan'), time_elapsed_total=extra_t)
            print(f'Initial Validation {val_str}=', Loss_val_now)
        try:
            import itertools
            t = Tictoctimer()
            start_t = time.time() #time keeping
            epochsrange = range(epochs) if timeout is None else itertools.count(start=0)
            if timeout is not None and verbose>0: 
                print(f'Starting indefinite training until {timeout} seconds have passed due to provided timeout')
            for epoch in (tqdm(epochsrange) if verbose>0 else epochsrange):
                bestfit_old = self.bestfit #to check if a new lowest validation loss has been achieved
                Loss_acc_epoch = 0.

                t.start()
                t.tic('data get')
                for train_batch in data_train_loader:
                    if cuda:
                        train_batch = [b.cuda() for b in train_batch]
                    t.toc('data get')
                    def closure(backward=True):
                        t.toc('optimizer start')
                        t.tic('loss')
                        Loss = self.loss(*train_batch, **loss_kwargs)
                        t.toc('loss')
                        if backward:
                            t.tic('zero_grad')
                            self.optimizer.zero_grad()
                            t.toc('zero_grad')
                            t.tic('backward')
                            Loss.backward()
                            t.toc('backward')
                        t.tic('stepping')
                        return Loss

                    # t.tic('optimizer')
                    t.tic('optimizer start')
                    training_loss = self.optimizer.step(closure).item()
                    t.toc('stepping')
                    # t.toc('optimizer')
                    Loss_acc_val += training_loss
                    Loss_acc_epoch += training_loss
                    N_batch_acc_val += 1
                    self.batch_counter += 1
                    self.epoch_counter += 1/N_batch_updates_per_epoch

                    t.tic('val')
                    if concurrent_val and remote.poll():
                        Loss_val_now, self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id, self.bestfit = remote.recv()
                        remote.receiving = False
                        #deepcopy(self) costs time due to the arrays that are passed?
                        remote.send((deepcopy(self), True, Loss_acc_val/N_batch_acc_val, time.time() - start_t + extra_t))
                        remote.receiving = True
                        Loss_acc_val, N_batch_acc_val, val_counter = 0, 0, val_counter + 1
                    t.toc('val')
                    t.tic('data get')
                t.toc('data get')

                #end of epoch clean up
                train_loss_epoch = Loss_acc_epoch/N_batch_updates_per_epoch
                t.tic('val')
                if not concurrent_val:
                    Loss_val_now = validation(append=True,train_loss=train_loss_epoch, \
                                        time_elapsed_total=time.time()-start_t+extra_t) #updates bestfit
                t.toc('val')
                t.pause()

                #printing routine
                if verbose>0:
                    time_elapsed = time.time() - start_t
                    if bestfit_old > self.bestfit:
                        print(f'########## New lowest validation loss achieved ########### {val_str} = {self.bestfit}')
                        best_epoch = epoch+1
                    if concurrent_val: #if concurrent val then print validation freq
                        val_feq = val_counter/(epoch+1)
                        valfeqstr = f', {val_feq:4.3} vals/epoch' if (val_feq>1 or val_feq==0) else f', {1/val_feq:4.3} epochs/val'
                    else: #else print validation time use
                        valfeqstr = f''
                    trainstr = f'sqrt loss {train_loss_epoch**0.5:7.4}' if sqrt_train else f'loss {train_loss_epoch:7.4}'
                    Loss_str = f'Epoch {epoch+1:4}, Train {trainstr}, Val {val_str} {Loss_val_now:6.4}'
                    loss_time = (t.acc_times['loss'] + t.acc_times['optimizer start'] + t.acc_times['zero_grad'] + t.acc_times['backward'] + t.acc_times['stepping'])  /t.time_elapsed
                    time_str = f'Time Loss: {loss_time:.1%}, data get: {t.acc_times["data get"]/t.time_elapsed:.1%}, val: {t.acc_times["val"]/t.time_elapsed:.1%}{valfeqstr}'
                    batch_feq = (self.batch_counter - batch_id_start)/(time.time() - start_t)
                    batch_str = (f'{batch_feq:4.1f} batches/sec' if (batch_feq>1 or batch_feq==0) else f'{1/batch_feq:4.1f} sec/batch')
                    print(f'{Loss_str}, {time_str}, {batch_str}')
                    if print_full_time_profile:
                        print('Time profile:',t.percent())

                #breaking on timeout
                if timeout is not None:
                    if time.time() >= start_t+timeout:
                        break
        except KeyboardInterrupt:
            print('Stopping early due to a KeyboardInterrupt')
        del data_train_loader
        #end of training
        if concurrent_val:
            if verbose: print(f'Waiting for started validation process to finish and one last validation... (receiving = {remote.receiving})',end='')
            if remote.receiving:
                #add poll here?
                Loss_val_now, self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id, self.bestfit = remote.recv() #recv dead lock here
                if verbose: print('Recv done... ',end='')
            if N_batch_acc_val>0: #there might be some trained but not yet tested
                remote.send((self, True, Loss_acc_val/N_batch_acc_val, time.time() - start_t + extra_t))
                Loss_val_now, self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id, self.bestfit = remote.recv()
            remote.close(); process.join()
            if verbose: print('Done!')

        self.train(); self.cpu();
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array(self.Loss_val), np.array(self.Loss_train), np.array(self.batch_id), np.array(self.time), np.array(self.epoch_id)
        self.checkpoint_save_system(name='_last')
        self.checkpoint_load_system(name='_best')
        if verbose: 
            print(f'Loaded model with best known validation {val_str} of {self.bestfit:6.4} which happened on epoch {best_epoch} (epoch_id={self.epoch_id[-1] if len(self.epoch_id)>0 else 0:.2f})')

    ########## Saving and loading ############
    def checkpoint_save_system(self, name='_best', directory=None):
        directory  = get_work_dirs()['checkpoints'] if directory is None else directory
        file = os.path.join(directory,self.name + name + '.pth')
        torch.save(self.__dict__, file)
    def checkpoint_load_system(self, name='_best', directory=None):
        directory  = get_work_dirs()['checkpoints'] if directory is None else directory
        file = os.path.join(directory,self.name + name + '.pth')
        try:
            self.__dict__ = torch.load(file)
            if self.fitted:
                self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array(self.Loss_val), np.array(self.Loss_train), np.array(self.batch_id), np.array(self.time), np.array(self.epoch_id)
                for i in np.where(np.isnan(self.Loss_train))[0]:
                    if i!=len(self.Loss_train)-1: #if the last is NaN than I will leave it there. Something weird happened like breaking before one validation loop was completed. 
                        self.Loss_train[i] = self.Loss_train[i+1]

        except FileNotFoundError:
            raise FileNotFoundError(f'No such file at {file}, did you set sys.unique_code correctly?')

    def save_system(self, file):
        '''Save the system using pickle provided by torch

        Notes
        -----
        This can be quite unstable for long term storage or switching between versions of this and other modules.
        Consider manually creating a save_system function for a long term solution. (maybe utilize checkpoint_save_system)
        '''
        torch.save(self, file)

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

def _worker(remote, parent_remote, sim_val=None, data_val=None, sim_val_fun='NRMS', loss_kwargs={}):
    '''Utility function used by .fit for concurrent validation'''
    
    parent_remote.close()
    while True:
        try:
            sys, append, Loss_train, time_now = remote.recv() #gets the current network
            #put deepcopy here?
            sys.eval(); sys.cpu()
            if sim_val is not None:
                sim_val_sim = sys.apply_experiment(sim_val)
                Loss_val = sim_val_sim.__getattribute__(sim_val_fun)(sim_val)
            else:
                with torch.no_grad():
                    Loss_val = sys.loss(*data_val,**loss_kwargs).item()

            if append:
                sys.Loss_val.append(Loss_val)
                sys.Loss_train.append(Loss_train)
                sys.batch_id.append(sys.batch_counter)
                sys.time.append(time_now)
                sys.epoch_id.append(sys.epoch_counter)

            sys.train() #back to training mode
            if sys.bestfit >= Loss_val:
                sys.bestfit = Loss_val
                sys.checkpoint_save_system('_best')
            remote.send((Loss_val, sys.Loss_val, sys.Loss_train, sys.batch_id, sys.time, sys.epoch_id, sys.bestfit)) #sends back arrays
        except EOFError: #main process stopped
            break
        except Exception as err: #some other error
            import traceback
            with open('validation process crash file.txt','w') as f:
                f.write(traceback.format_exc())
            raise err

class default_dataset(Dataset):
    """docstring for default_dataset"""
    def __init__(self, data):
        super(default_dataset, self).__init__()
        self.data = [np.array(d,dtype=np.float32) for d in data]

    def __getitem__(self, i):
        return [d[i] for d in self.data]

    def __len__(self):
        return len(self.data[0])

import time
class Tictoctimer(object):
    def __init__(self):
        self.time_acc = 0
        self.timer_running = False
        self.start_times = dict()
        self.acc_times = dict()
    @property
    def time_elapsed(self):
        if self.timer_running:
            return self.time_acc + time.time() - self.start_t
        else:
            return self.time_acc
    
    def start(self):
        self.timer_running = True
        self.start_t = time.time()
        
    def pause(self):
        self.time_acc += time.time() - self.start_t
        self.timer_running = False
    
    def tic(self,name):
        self.start_times[name] = time.time()
    
    def toc(self,name):
        if self.acc_times.get(name) is None:
            self.acc_times[name] = time.time() - self.start_times[name]
        else:
            self.acc_times[name] += time.time() - self.start_times[name]

    def percent(self):
        elapsed = self.time_elapsed
        R = sum([item for key,item in self.acc_times.items()])
        return ', '.join([key + f' {item/elapsed:.1%}' for key,item in self.acc_times.items()]) +\
                f', others {1-R/elapsed:.1%}'
        
class My_Simple_DataLoader:
    def __init__(self, data, batch_size=32):
        self.data = [torch.as_tensor(d,dtype=torch.float32) for d in data] #convert to torch?
        self.ids = np.arange(len(data[0]),dtype=int)
        self.batch_size = batch_size
    
    def __iter__(self):
        np.random.shuffle(self.ids)
        return My_Simple_DataLoaderIterator(self.data, self.ids, self.batch_size)
    
class My_Simple_DataLoaderIterator:
    def __init__(self, data, ids, batch_size):
        self.ids = ids #already shuffled
        self.data = data
        self.L = len(data[0])
        self.i = 0
        self.batch_size = batch_size
    def __iter__(self):
        return self
    def __next__(self):
        self.i += self.batch_size
        if self.i>self.L:
            raise StopIteration
        ids_now = self.ids[self.i-self.batch_size:self.i]
        return [d[ids_now] for d in self.data]


if __name__ == '__main__':
    # sys = deepSI.fit_systems.SS_encoder(nx=3,na=5,nb=5)
    sys = deepSI.fit_systems.Torch_io_siso(10,10)
    train, test = deepSI.datasets.CED()
    print(train,test)
    # exit()
    # sys.fit(train,loss_val=test,epochs=500,batch_size=126,concurrent_val=True)
    sys.fit(train,sim_val=test,loss_kwargs=dict(pre_construct=True),epochs=500,batch_size=126,concurrent_val=True,num_workers_data_loader=0)
    # sys.fit(train,sim_val=test,epochs=10,batch_size=64,concurrent_val=False)
    # sys.fit(train,sim_val=test,epochs=10,batch_size=64,concurrent_val=True)
    print(sys.Loss_train)