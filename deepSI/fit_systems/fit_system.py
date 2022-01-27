
from deepSI.systems.system import System, System_io, System_data, load_system
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
import itertools
from copy import deepcopy
import warnings

class System_fittable(System):
    """Subclass of system which introduces a .fit method which calls ._fit to fit the systems

    Notes
    -----
    This function will automaticly fit the normalization in self.norm if auto_fit_norm is set to True (default). 
    Lastly it will set self.init_model_done to True which will keep the norm constant. 
    """
    def init_model(self, sys_data=None, nu=-1, ny=-1, auto_fit_norm=True):
        if auto_fit_norm: #if the norm is not used you can also manually initialize it.
            #you may consider not using the norm if you have constant values in your training data which can change. They are known to cause quite a number of bugs and errors. 
            self.norm.fit(sys_data)
        self.nu = sys_data.nu
        self.ny = sys_data.ny
        self.init_model_done = True

    def fit(self, train_sys_data, auto_fit_norm=True, **kwargs):
        if self.init_model_done==False:
            self.init_model(train_sys_data, auto_fit_norm=auto_fit_norm)            
        self._fit(self.norm.transform(train_sys_data), **kwargs)

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
        parameters : list or list of dict
            system torch parameters
        optimizer_kwargs : dict
            If 'optimizer' is defined than that optimizer will be used otherwise Adam will be used.
            The other parameters will be passed to the optimizer as a kwarg.
        '''
        if optimizer_kwargs.get('optimizer') is not None:
            optimizer_kwargs = deepcopy(optimizer_kwargs) #do not modify the original kwargs, is this necessary
            optimizer = optimizer_kwargs['optimizer']
            del optimizer_kwargs['optimizer']
        else:
            optimizer = torch.optim.Adam
        return optimizer(parameters,**optimizer_kwargs)

    def init_scheduler(self, **scheduler_kwargs):
        '''Optionally defined in subclass to create the scheduler

        Parameters
        ----------
        scheduler_kwargs : dict
            If 'scheduler' is defined than that scheduler will be used otherwise no scheduler will be used.
        '''
        if not scheduler_kwargs:
            return None
        scheduler_kwargs = deepcopy(scheduler_kwargs)
        scheduler = scheduler_kwargs['scheduler']
        del scheduler_kwargs['scheduler']
        return scheduler(self.optimizer,**scheduler_kwargs)

    def init_model(self, sys_data=None, nu=-1, ny=-1, device='cpu', auto_fit_norm=True, optimizer_kwargs={}, parameters_optimizer_kwargs={}, scheduler_kwargs={}):
        '''This function set the nu and ny, inits the network, moves parameters to device, initilizes optimizer and initilizes logging parameters'''
        if sys_data==None:
            assert nu!=-1 and ny!=-1, 'either sys_data or (nu and ny) should be provided'
            self.nu, self.ny = nu, ny
        else:
            self.nu, self.ny = sys_data.nu, sys_data.ny
            if auto_fit_norm:
                if not self.norm.is_id:
                    warnings.warn('Fitting the norm due to auto_fit_norm=True')
                self.norm.fit(sys_data)
        self.init_nets(self.nu, self.ny)
        self.to_device(device=device)
        parameters_and_optim = [{**item,**parameters_optimizer_kwargs.get(name,{})} for name,item in self.parameters_with_names.items()]
        self.optimizer = self.init_optimizer(parameters_and_optim, **optimizer_kwargs)
        self.scheduler = self.init_scheduler(**scheduler_kwargs)
        self.bestfit = float('inf')
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        self.init_model_done = True
    @property
    def parameters(self):
        return [item for key,item in self.parameters_with_names.items()]
    @property
    def parameters_with_names(self):
        nns = {d:{'params':self.__getattribute__(d).parameters()} for d in dir(self) if \
            d not in ['parameters_with_names','parameters'] and isinstance(self.__getattribute__(d),nn.Module   )}
        pars= {d:{'params':self.__getattribute__(d)}              for d in dir(self) if \
            d not in ['parameters_with_names','parameters'] and isinstance(self.__getattribute__(d),nn.Parameter)}
        return {**nns,**pars}


    def cal_validation_error(self, val_sys_data, validation_measure='sim-NRMS'):
        '''possible validation_measure are
        'sim-NRMS'
        'sim-NRMS_mean_channel'
        'sim-NRMS_per_channels' (and others defined in System_data)
        'sim-NRMS_sys_norm'
        
        '10-step-NRMS' or '10-step-average-NRMS'
        '10-step-last-NRMS'
        '10-step-last-RMS'
        '10-step-[w0,w1,w2,w3,w4,w5,w6,w7,w8,w9]-NRMS' weighted mean 10-step-error
        
        #todo;
        User given callback. (overwrite this function?)
        'loss'  #todo
        'sim-inno' #todo
        '''
        if validation_measure.find('sim')==0:
            val_sys_data_sim = self.apply_experiment(val_sys_data)
            sim_val_fun = validation_measure.split('-')[1]
            if sim_val_fun=='NRMS_sys_norm':
                return self.norm.transform(val_sys_data_sim).RMS(self.norm.transform(val_sys_data))
            else:
                return val_sys_data_sim.__getattribute__(sim_val_fun)(val_sys_data)
        elif validation_measure.find('step')!=-1:
            splitted = validation_measure.split('-')
            nstep = int(splitted[0])
            mode = splitted[-1]
            n_step_error = self.n_step_error(val_sys_data, nf=nstep, stride=1, mode=mode, mean_channels=True)

            if len(splitted)==3:
                average_method = 'average'
            else:
                average_method = splitted[2]
            if average_method[0]=='[':
                w = np.array([float(a) for a in average_method[1:-1].split(',')])
                return np.sum(w*n_step_error)/np.sum(w)
            elif average_method=='average':
                return np.mean(n_step_error)
            elif average_method=='last':
                return n_step_error[-1]
        NotImplementedError(f'validation_measure={validation_measure} not implemented, use one as "sim-NRMS", "sim-NRMS_mean_channels", "10-step-average-NRMS", ect.')

    def fit(self, train_sys_data, val_sys_data, epochs=30, batch_size=256, loss_kwargs={}, \
            auto_fit_norm=True, validation_measure='sim-NRMS', optimizer_kwargs={}, concurrent_val=False, cuda=False, \
            timeout=None, verbose=1, sqrt_train=True, num_workers_data_loader=0, print_full_time_profile=False, scheduler_kwargs={}):
        '''The batch optimization method with parallel validation, 

        Parameters
        ----------
        train_sys_data : System_data or System_data_list
            The system data to be fitted
        val_sys_data : System_data or System_data_list
            The validation system data after each used after each epoch for early stopping. Use the keyword argument validation_measure to specify which measure should be used. 
        epochs : int
        batch_size : int
        loss_kwargs : dict
            The Keyword Arguments to be passed to the self.make_training_data and self.loss of the current fit_system.
        auto_fit_norm : boole
            If true will use self.norm.fit(train_sys_data) which will fit it element wise. 
        validation_measure : str
            Specify which measure should be used for validation, e.g. 'sim-RMS', 'sim-NRMS_mean_channel', 'sim-NRMS_sys_norm', ect. See self.cal_validation_error for details.
        optimizer_kwargs : dict
            The Keyword Arguments to be passed on to init_optimizer. notes; init_optimizer['optimizer'] is the optimization function used (default torch.Adam)
            and optimizer_kwargs['parameters_optimizer_kwargs'] the learning rates and such for the different elements of the models. see https://pytorch.org/docs/stable/optim.html
        concurrent_val : boole
            If set to true a subprocess will be started which concurrently evaluates the validation method selected.
            Warning: if concurrent_val is set than "if __name__=='__main__'" or import from a file if using self defined method or networks.
        cuda : bool
            if cuda will be used (often slower than not using it, be aware)
        timeout : None or number
            Alternative to epochs to run until a set amount of time has past. 
        verbose : int
            Set to 0 for a silent run
        sqrt_train : boole
            will sqrt the loss while printing
        num_workers_data_loader : int
            see https://pytorch.org/docs/stable/data.html
        print_full_time_profile : boole
            will print the full time profile, useful for debugging and basic process optimization. 
        scheduler_kwargs : dict
            learning rate scheduals are a work in progress.
        
        Notes
        -----
        This method implements a batch optimization method in the following way; each epoch the training data is scrambled and batched where each batch
        is passed to the self.loss method and utilized to optimize the parameters. After each epoch the systems is validated using the evaluation of a 
        simulation or a validation split and a checkpoint will be crated if a new lowest validation loss has been achieved. (or concurrently if concurrent_val=True)
        After training (which can be stopped at any moment using a KeyboardInterrupt) the system is loaded with the lowest validation loss. 

        The default checkpoint location is "C:/Users/USER/AppData/Local/deepSI/checkpoints" for windows and ~/.deepSI/checkpoints/ for unix like.
        These can be loaded manually using sys.load_checkpoint("_best") or "_last". (For this to work the sys.unique_code needs to be set to the correct string)
        '''
        def validation(train_loss=None, time_elapsed_total=None):
            self.eval(); self.cpu()
            Loss_val = self.cal_validation_error(val_sys_data, validation_measure=validation_measure)
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
        
        ########## Initialization ##########
        if self.init_model_done==False:
            if verbose: print('Initilizing the model and optimizer')
            device = 'cuda' if cuda else 'cpu'
            optimizer_kwargs = deepcopy(optimizer_kwargs)
            parameters_optimizer_kwargs = optimizer_kwargs.get('parameters_optimizer_kwargs',{})
            if parameters_optimizer_kwargs:
                del optimizer_kwargs['parameters_optimizer_kwargs']
            self.init_model(sys_data=train_sys_data, device=device, auto_fit_norm=auto_fit_norm, optimizer_kwargs=optimizer_kwargs,\
                    parameters_optimizer_kwargs=parameters_optimizer_kwargs, scheduler_kwargs=scheduler_kwargs)
        else:
            if verbose: print('Model already initilized, skipping initilizing of model and optimizer')

        if self.scheduler==False and verbose:
            print('!!!! Your might be continuing from a save which had scheduler but which was removed during saving... check this !!!!!!')
        
        self.dt = train_sys_data.dt
        if cuda: 
            self.cuda()
        self.train()

        self.epoch_counter = 0 if len(self.epoch_id)==0 else self.epoch_id[-1]
        self.batch_counter = 0 if len(self.batch_id)==0 else self.batch_id[-1]
        extra_t            = 0 if len(self.time)    ==0 else self.time[-1] #correct timer after restart

        ########## Getting the data ##########
        data_train = self.make_training_data(self.norm.transform(train_sys_data), **loss_kwargs)
        if not isinstance(data_train, Dataset) and verbose: print_array_byte_size(sum([d.nbytes for d in data_train]))

        #### transforming it back to a list to be able to append. ########
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = list(self.Loss_val), list(self.Loss_train), list(self.batch_id), list(self.time), list(self.epoch_id)

        #### init monitoring values ########
        Loss_acc_val, N_batch_acc_val, val_counter, best_epoch, batch_id_start = 0, 0, 0, 0, self.batch_counter #to print the frequency of the validation step.
        N_training_samples = len(data_train) if isinstance(data_train, Dataset) else len(data_train[0])
        batch_size = min(batch_size, N_training_samples)
        N_batch_updates_per_epoch = N_training_samples//batch_size
        if verbose>0: 
            print(f'N_training_samples = {N_training_samples}, batch_size = {batch_size}, N_batch_updates_per_epoch = {N_batch_updates_per_epoch}')
        
        ### convert to dataset ###
        if isinstance(data_train, Dataset):
            persistent_workers = False if num_workers_data_loader==0 else True
            data_train_loader = DataLoader(data_train, batch_size=batch_size, drop_last=True, shuffle=True, \
                                   num_workers=num_workers_data_loader, persistent_workers=persistent_workers)
        else: #add my basic DataLoader
            data_train_loader = My_Simple_DataLoader(data_train, batch_size=batch_size) #is quite a bit faster for low data situations

        if concurrent_val:
            self.remote_start(val_sys_data, validation_measure)
            self.remote_send(float('nan'), extra_t)
        else: #start with the initial validation 
            validation(train_loss=float('nan'), time_elapsed_total=extra_t) #also sets current model to cuda
            if verbose: 
                print(f'Initial Validation {validation_measure}=', self.Loss_val[-1])

        try:
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

                    t.tic('optimizer start')
                    training_loss = self.optimizer.step(closure).item()
                    t.toc('stepping')
                    if self.scheduler:
                        t.tic('scheduler')
                        self.scheduler.step()
                        t.tic('scheduler')
                    Loss_acc_val += training_loss
                    Loss_acc_epoch += training_loss
                    N_batch_acc_val += 1
                    self.batch_counter += 1
                    self.epoch_counter += 1/N_batch_updates_per_epoch

                    t.tic('val')
                    if concurrent_val and self.remote_recv(): ####### validation #######
                        self.remote_send(Loss_acc_val/N_batch_acc_val, time.time()-start_t+extra_t)
                        Loss_acc_val, N_batch_acc_val, val_counter = 0., 0, val_counter + 1
                    t.toc('val')
                    t.tic('data get')
                t.toc('data get')

                ########## end of epoch clean up ##########
                train_loss_epoch = Loss_acc_epoch/N_batch_updates_per_epoch
                if np.isnan(train_loss_epoch):
                    if verbose>0: print(f'&&&&&&&&&&&&& Encountered a NaN value in the training loss at epoch {epoch}, breaking from loop &&&&&&&&&&')
                    break

                t.tic('val')
                if not concurrent_val:
                    validation(train_loss=train_loss_epoch, \
                               time_elapsed_total=time.time()-start_t+extra_t) #updates bestfit and goes back to cpu and back
                t.toc('val')
                t.pause()

                ######### Printing Routine ##########
                if verbose>0:
                    time_elapsed = time.time() - start_t
                    if bestfit_old > self.bestfit:
                        print(f'########## New lowest validation loss achieved ########### {validation_measure} = {self.bestfit}')
                        best_epoch = epoch+1
                    if concurrent_val: #if concurrent val than print validation freq
                        val_feq = val_counter/(epoch+1)
                        valfeqstr = f', {val_feq:4.3} vals/epoch' if (val_feq>1 or val_feq==0) else f', {1/val_feq:4.3} epochs/val'
                    else: #else print validation time use
                        valfeqstr = f''
                    trainstr = f'sqrt loss {train_loss_epoch**0.5:7.4}' if sqrt_train else f'loss {train_loss_epoch:7.4}'
                    Loss_val_now = self.Loss_val[-1] if len(self.Loss_val)!=0 else float('nan')
                    Loss_str = f'Epoch {epoch+1:4}, {trainstr}, Val {validation_measure} {Loss_val_now:6.4}'
                    loss_time = (t.acc_times['loss'] + t.acc_times['optimizer start'] + t.acc_times['zero_grad'] + t.acc_times['backward'] + t.acc_times['stepping'])  /t.time_elapsed
                    time_str = f'Time Loss: {loss_time:.1%}, data: {t.acc_times["data get"]/t.time_elapsed:.1%}, val: {t.acc_times["val"]/t.time_elapsed:.1%}{valfeqstr}'
                    self.batch_feq = (self.batch_counter - batch_id_start)/(time.time() - start_t)
                    batch_str = (f'{self.batch_feq:4.1f} batches/sec' if (self.batch_feq>1 or self.batch_feq==0) else f'{1/self.batch_feq:4.1f} sec/batch')
                    print(f'{Loss_str}, {time_str}, {batch_str}')
                    if print_full_time_profile:
                        print('Time profile:',t.percent())

                ####### Timeout Breaking ##########
                if timeout is not None:
                    if time.time() >= start_t+timeout:
                        break
        except KeyboardInterrupt:
            print('Stopping early due to a KeyboardInterrupt')

        self.train(); self.cpu()
        del data_train_loader

        ####### end of training concurrent things #####
        if concurrent_val:
            if verbose: print(f'Waiting for started validation process to finish and one last validation... (receiving = {self.remote.receiving})',end='')
            if self.remote_recv(wait=True):
                if verbose: print('Recv done... ',end='')
                if N_batch_acc_val>0:
                    self.remote_send(Loss_acc_val/N_batch_acc_val, time.time()-start_t+extra_t)
                    self.remote_recv(wait=True)
            self.remote_close()
            if verbose: print('Done!')

        
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array(self.Loss_val), np.array(self.Loss_train), np.array(self.batch_id), np.array(self.time), np.array(self.epoch_id)
        self.checkpoint_save_system(name='_last')
        try:
            self.checkpoint_load_system(name='_best')
        except FileNotFoundError:
            print('no best checkpoint found keeping last')
        if verbose: 
            print(f'Loaded model with best known validation {validation_measure} of {self.bestfit:6.4} which happened on epoch {best_epoch} (epoch_id={self.epoch_id[-1] if len(self.epoch_id)>0 else 0:.2f})')

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
            if self.init_model_done:
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

    ### CPU & CUDA Transfers ###
    def cuda(self):
        self.to_device('cuda')
    def cpu(self):
        self.to_device('cpu')
    def to_device(self,device):
        for d in dir(self):
            if d in ['parameters_with_names','parameters']:
                continue
            attribute = self.__getattribute__(d)
            if isinstance(attribute,(nn.Module,nn.Parameter)):
                attribute.to(device)
            elif isinstance(attribute, torch.optim.Optimizer):
                for key,item in attribute.state.items():
                    for name,item2 in item.items():
                        if isinstance(item2, torch.Tensor):
                            item[name] = item2.to(device)
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

    ########## Remote ##########
    def remote_start(self, val_sys_data, validation_measure):
        from multiprocessing import Process, Pipe
        self.remote, work_remote = Pipe()
        self.remote.receiving = False
        process = Process(target=_worker, args=(work_remote, self.remote, val_sys_data, validation_measure))
        process.daemon = True  # if the main process crashes, we should not cause things to hang
        process.start()
        work_remote.close()
        self.remote.process = process

    def remote_send(self, Loss_acc_val, time_optimize):
        assert self.remote.receiving==False
        remote = self.remote
        del self.remote #remote cannot be copyied by deepcopy
        copy_self = deepcopy(self)
        self.remote = remote
        copy_self.cpu(); copy_self.eval()
        import pickle
        if b'__main__' in pickle.dumps(copy_self.scheduler):
            print('setting scheduler to None for there is some main function found')
            copy_self.scheduler = False
        self.remote.send((copy_self, Loss_acc_val, time_optimize)) #time here does not matter
        self.remote.receiving = True

    def remote_recv(self,wait=False):
        if self.remote.receiving and (self.remote.poll() or wait):
            self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id, self.bestfit = self.remote.recv()
            self.remote.receiving = False
            return True
        else:
            return False

    def remote_close(self):
        self.remote.close()
        self.remote.process.join()
        del self.remote

import signal
import logging
class IgnoreKeyboardInterrupt:
    def __enter__(self):
        self.old_handler = signal.signal(signal.SIGINT, self.handler)
                
    def handler(self, sig, frame):
        print('Validation process received SIGINT but was ignored in favour of finishing computations.')
    
    def __exit__(self, type, value, traceback): #on exit it will raise the keyboard interpurt
        signal.signal(signal.SIGINT, self.old_handler)

def _worker(remote, parent_remote, val_sys_data=None, validation_measure='sim-NRMS'):
    '''Utility function used by .fit for concurrent validation'''
    
    parent_remote.close()
    while True:
        try:
            with IgnoreKeyboardInterrupt():
                sys, Loss_train, time_now = remote.recv() #gets the current network
                Loss_val = sys.cal_validation_error(val_sys_data, validation_measure)
                sys.Loss_val.append(Loss_val)
                sys.Loss_train.append(Loss_train)
                sys.batch_id.append(sys.batch_counter)
                sys.time.append(time_now)
                sys.epoch_id.append(sys.epoch_counter)

                sys.train() #back to training mode
                if sys.bestfit >= Loss_val:
                    sys.bestfit = Loss_val
                    sys.checkpoint_save_system('_best')
                remote.send((sys.Loss_val, sys.Loss_train, sys.batch_id, sys.time, sys.epoch_id, sys.bestfit)) #sends back arrays
        except EOFError: #main process stopped
            break
        except Exception as err: #some other error
            import traceback
            with open('validation process crash file.txt','w') as f:
                f.write(traceback.format_exc())
            raise err


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
        self.data = [torch.as_tensor(d,dtype=torch.float32) for d in data] #this copies the data again
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

def print_array_byte_size(Dsize):
    if Dsize>2**30: 
        dstr = f'{Dsize/2**30:.1f} GB!'
        dstr += '\nConsider using online_construct=True (in loss_kwargs) or let make_training_data return a Dataset to reduce data-usage'
    elif Dsize>2**20: 
        dstr = f'{Dsize/2**20:.1f} MB'
    else:
        dstr = f'{Dsize/2**10:.1f} kB'
    print('Size of the training array = ', dstr)

if __name__ == '__main__':
    # sys = deepSI.fit_systems.SS_encoder(nx=3,na=5,nb=5)
    sys = deepSI.fit_systems.Torch_io_siso(10,10)
    train, test = deepSI.datasets.CED()
    print(train,test)
    # exit()
    # sys.fit(train,loss_val=test,epochs=500,batch_size=126,concurrent_val=True)
    sys.fit(train,sim_val=test,loss_kwargs=dict(online_construct=False),epochs=500,batch_size=126,\
            concurrent_val=True,num_workers_data_loader=0,validation_measure='sim-NRMS')
    # sys.fit(train,sim_val=test,epochs=10,batch_size=64,concurrent_val=False)
    # sys.fit(train,sim_val=test,epochs=10,batch_size=64,concurrent_val=True)
    print(sys.Loss_train)
