
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

    def fit(self, sys_data, epochs=30, batch_size=256, loss_kwargs={}, optimizer_kwargs={}, \
            sim_val=None, concurrent_val=False, timeout=None, verbose=1, cuda=False, val_frac=0.2, sim_val_fun='NRMS', sqrt_train=True):
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
            global time_val
            t_start_val = time.time()
            
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
                time_val += time.time() - t_start_val
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

        self.set_dt(sys_data.dt)

        self.epoch_counter = 0 if len(self.epoch_id)==0 else self.epoch_id[-1]
        self.batch_counter = 0 if len(self.batch_id)==0 else self.batch_id[-1]
        extra_t            = 0 if len(self.time)    ==0 else self.time[-1] #correct time counting after reset

        sys_data = self.norm.transform(sys_data)
        data_full = self.make_training_data(sys_data, **loss_kwargs)

        if sim_val is not None:
            data_train = [torch.tensor(dat, dtype=torch.float32) for dat in data_full]
            data_val = None
        else: #is not used that often, could use sklearn to split data
            from sklearn.model_selection import train_test_split
            datasplitted = [torch.tensor(a, dtype=torch.float32) for a in train_test_split(*data_full,random_state=42)] # (A1_train, A1_test, A2_train, A2_test)
            data_train = [datasplitted[i] for i in range(0,len(datasplitted),2)]
            data_val = [datasplitted[i] for i in range(1,len(datasplitted),2)]


        #transforming it back to a list to be able to append.
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = list(self.Loss_val), list(self.Loss_train), list(self.batch_id), list(self.time), list(self.epoch_id)

        global time_val, time_loss, time_back #time keeping
        time_val = time_back = time_loss = 0.
        Loss_acc_val, N_batch_acc_val = 0, 0
        N_training_samples = len(data_train[0])
        batch_size = min(batch_size, N_training_samples)
        N_batch_updates_per_epoch = N_training_samples//batch_size
        if verbose>0: 
            print(f'N_training_samples = {N_training_samples}, batch_size = {batch_size}, N_batch_updates_per_epoch = {N_batch_updates_per_epoch}')
        ids = np.arange(0, N_training_samples, dtype=int)
        val_counter = 0  #to print the frequency of the validation step.
        val_str = sim_val_fun if sim_val is not None else 'loss' #for correct printing
        best_epoch = 0
        batch_id_start = self.batch_counter
        

        if concurrent_val:
            from multiprocessing import Process, Pipe
            from copy import deepcopy
            remote, work_remote = Pipe()
            remote.receiving = False
            process = Process(target=_worker, args=(work_remote, remote, sim_val, data_val, sim_val_fun, loss_kwargs))
            process.daemon = True  # if the main process crashes, we should not cause things to hang
            process.start()
            work_remote.close()
            remote.send((deepcopy(self), True, float('NaN'), extra_t)) #time here does not matter
            #            sys, append, Loss_train, time_now
            remote.receiving = True
            #           sys, append, Loss_acc, time_now, epoch
            Loss_val_now = float('nan')
        else: #do it now
            Loss_val_now = validation(append=True, train_loss=float('nan'), time_elapsed_total=extra_t)
            print(f'Initial Validation {val_str}=', Loss_val_now)
        try:
            start_t = time.time() #time keeping
            import itertools
            epochsrange = range(epochs) if timeout is None else itertools.count(start=0)
            if timeout is not None and verbose>0: print(f'Starting indefinite training until {timeout} seconds have passed due to provided timeout')
            for epoch in (tqdm(epochsrange) if verbose>0 else epochsrange):
                np.random.shuffle(ids)
                bestfit_old = self.bestfit #to check if a new lowest validation loss has been achieved
                Loss_acc_epoch = 0.
                for i in range(batch_size, N_training_samples + 1, batch_size):
                    ids_batch = ids[i-batch_size:i]
                    train_batch = [(part[ids_batch] if not cuda else part[ids_batch].cuda()) for part in data_train] 

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

                    training_loss = self.optimizer.step(closure).item()
                    Loss_acc_val += training_loss
                    Loss_acc_epoch += training_loss
                    N_batch_acc_val += 1
                    self.batch_counter += 1
                    self.epoch_counter += 1/N_batch_updates_per_epoch

                    if concurrent_val and remote.poll():
                        Loss_val_now, self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id, self.bestfit = remote.recv()
                        remote.receiving = False
                        #deepcopy(self) costs time due to the arrays that are passed.
                        remote.send((deepcopy(self), True, Loss_acc_val/N_batch_acc_val, time.time() - start_t + extra_t))
                        remote.receiving = True
                        Loss_acc_val, N_batch_acc_val, val_counter = 0, 0, val_counter + 1

                #end of epoch clean up
                train_loss_epoch = Loss_acc_epoch/N_batch_updates_per_epoch
                if not concurrent_val:
                    Loss_val_now = validation(append=True,train_loss=train_loss_epoch, time_elapsed_total=time.time()-start_t+extra_t) #updates bestfit

                #printing routine
                if verbose>0:
                    time_elapsed = time.time() - start_t
                    if bestfit_old > self.bestfit:
                        print(f'########## New lowest validation loss achieved ########### {val_str} = {self.bestfit}')
                        best_epoch = epoch+1
                    if concurrent_val: #if concurrent val then print validation freq
                        val_feq = val_counter/(epoch+1)
                        valfeqstr = (f'{val_feq:4.3} vals/epoch' if (val_feq>1 or val_feq==0) else f'{1/val_feq:4.3} epochs/val')
                    else: #else print validation time use
                        valfeqstr = f'val: {time_val/time_elapsed:.1%}'
                    trainstr = f'sqrt loss {train_loss_epoch**0.5:7.4}' if sqrt_train else f'loss {train_loss_epoch:7.4}'
                    Loss_str = f'Epoch {epoch+1:4}, Train {trainstr}, Val {val_str} {Loss_val_now:6.4}'
                    time_str = f'Time Loss: {time_loss/time_elapsed:.1%}, back: {time_back/time_elapsed:.1%}, {valfeqstr}'
                    batch_feq = (self.batch_counter - batch_id_start)/(time.time() - start_t)
                    batch_str = (f'{batch_feq:4.3} batches/sec' if (batch_feq>1 or batch_feq==0) else f'{1/batch_feq:4.3} sec/batch')
                    print(f'{Loss_str}, {time_str}, {batch_str}')

                #breaking on timeout
                if timeout is not None:
                    if time.time() >= start_t+timeout:
                        break
        except KeyboardInterrupt:
            print('Stopping early due to a KeyboardInterrupt')
        #end of training
        if concurrent_val:
            if verbose: print('Waiting for started validation process to finish and one last validation...receiving=',remote.receiving,end='')
            if remote.receiving:
                #add poll here?
                Loss_val_now, self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id, self.bestfit = remote.recv() #recv dead lock here
                if verbose: print('recv done...',end='')
            if N_batch_acc_val>0: #there might be some trained but not yet tested
                remote.send((self, True, Loss_acc_val/N_batch_acc_val, time.time() - start_t + extra_t))
                Loss_val_now, self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id, self.bestfit = remote.recv()
            remote.close(); process.join()
            if verbose: print('Done!')

        self.train(); self.cpu();
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array(self.Loss_val), np.array(self.Loss_train), np.array(self.batch_id), np.array(self.time), np.array(self.epoch_id)
        self.checkpoint_save_system(name='_last')
        try:
            self.checkpoint_load_system(name='_best')
        except FileNotFoundError:
            print('no best checkpoint found keeping last')
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

    ######### Continuous Time #########
    def set_dt(self,dt):
        if hasattr(self,'fn') and isinstance(self.fn, deepSI.utils.time_integrators):
            self.fn.dt = dt
        self.dt = dt

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
            out = (Loss_val, sys.Loss_val, sys.Loss_train, sys.batch_id, sys.time, sys.epoch_id, sys.bestfit)
            del sys
            remote.send(out) #sends back arrays
        except EOFError: #main process stopped
            break
        except Exception as err: #some other error
            import traceback
            with open('validation process crash file.txt','w') as f:
                f.write(traceback.format_exc())
            raise err



if __name__ == '__main__':
    # #check if system is continues, pass dt into 
    # class Test(object):
    #     """docstring for test"""
    #     def __init__(self, arg):
    #         super(Test, self).__init__()
    #         self.arg = arg

    #     def forward(self,**kwargs):
    #         kwargs['dt'] = 1.
    #         self.loss(**kwargs)

    #     def loss(self, nu=2):
    #         return nu
    # t = Test(1)
    # print(t.forward(nu=2))
    pass            
    # sys = deepSI.fit_systems.SS_encoder()
    # train, test = deepSI.datasets.CED()
    # sys.fit(train,sim_val=test,epochs=2,batch_size=64,concurrent_val=True)
    # # sys.fit(train,sim_val=test,epochs=10,batch_size=64,concurrent_val=False)
    # # sys.fit(train,sim_val=test,epochs=10,batch_size=64,concurrent_val=True)
    # print(sys.Loss_train)