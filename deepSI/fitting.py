
import numpy as np
import torch
import cloudpickle, os
from secrets import token_urlsafe
from copy import deepcopy
from tqdm.auto import tqdm
from torch import nn, optim
from nonlinear_benchmarks import Input_output_data
import time

def compute_NMSE(*A) -> torch.Tensor:
    '''Computes the Normalized Mean Squared Error. 
    Example usage: compute_NMSE(model, *xarrays, yarray) or compute_NMSE(model, upast, ypast, ufuture, yfuture)'''
    model, *xarrays, yarray = A
    yout = model(*xarrays, yarray)
    return torch.mean((yout-yarray)**2/model.norm.ystd**2)

def data_batcher(*arrays, batch_size=256, seed=0, device=None, indices=None):
    rng = np.random.default_rng(seed=seed)
    if indices is None:
        indices = np.arange(arrays[0].shape[0])
    dataset_size = len(indices)
    assert all(array.shape[0] == arrays[0].shape[0] for array in arrays)
    assert batch_size <= dataset_size
    while True:
        perm = rng.permutation(indices)
        start, end = 0, batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm].to(device) for array in arrays) #arrays are already torch arrays
            start, end = start + batch_size, end + batch_size

def fit(model: nn.Module, train:Input_output_data, val:Input_output_data, n_its:int, T:int=50, \
        batch_size:int=256, stride:int=1, val_freq:int=250, optimizer:optim.Optimizer=None, \
            device=None, compile_mode=None, loss_fun=compute_NMSE, val_fun=compute_NMSE):
    """
    Trains a PyTorch model, saving the best model and tracking training/validation progress.

    Args:
        model (nn.Module): Neural network model to be trained. The model must implement a 
            `.create_arrays(train, T, stride)` method to generate training arrays.
        train (Input_output_data): Training dataset.
        val (Input_output_data): Validation dataset.
        n_its (int): Number of training iterations (i.e., batch updates).
        T (int, optional): Sequence length considered in the loss (unroll length). Default is 50.
        batch_size (int, optional): Number of samples per batch during training. Default is 256.
        stride (int, optional): Step size for generating batches from the data. Default is 1.
        val_freq (int, optional): Frequency of validation checks (in iterations). Default is 250.
        optimizer (optim.Optimizer, optional): Optimizer for training. Default is Adam if not provided.
        device (torch.device, optional): Device to move the model and data to (e.g., 'cpu', 'cuda').
        compile_mode (optional): Optional mode for torch.compile to optimize the training step.
        loss_fun (callable, optional): Loss function used for training. Default is `compute_NMSE`.
        val_fun (callable, optional): Function used to compute validation loss. Default is `compute_NMSE`.

    Returns:
        dict: Contains the following keys:
            - 'best_model': The best model (with the lowest validation loss).
            - 'best_optimizer_state': Optimizer state when the best model was found.
            - 'last_model': The model at the end of training.
            - 'last_optimizer_state': The optimizer state at the end of training.
            - 'NRMS_train': Training loss history (normalized root mean square error).
            - 'NRMS_val': Validation loss history (normalized root mean square error).
            - 'samples/sec': Number of data samples processed per second.
            - 'val_freq': Validation frequency.
            - 'batch_size': Batch size used during training.
            - 'it_counter': List of iteration counts corresponding to each validation point.
    """

    def train_step(model, batch, optimizer):
        def closure(backward=True):
            loss = loss_fun(model, *batch)
            if backward:
                optimizer.zero_grad()
                loss.backward()
            return loss
        loss = optimizer.step(closure) #Using closure for the case that LBFGS is used.
        return loss.item()
    if compile_mode is not None:
        train_step = torch.compile(train_step, mode=compile_mode)
    
    code = token_urlsafe(4).replace('_','0').replace('-','a')
    save_filename = os.path.join(get_checkpoint_dir(), f'{model.__class__.__name__}-{code}.pth')
    fit_info = {'val_freq': val_freq, 'batch_size':batch_size}
    
    # Creat optimizer
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters()) if optimizer==None else optimizer    

    # Create training arrays
    arrays, indices = model.create_arrays(train, T=T, stride=stride) 
    print(f'Number of samples to train on = {len(indices)}')
    itter = data_batcher(*arrays, batch_size=batch_size, indices=indices, device=device)

    # Create validation arrays
    arrays_val, indices = model.create_arrays(val, T='sim')
    arrays_val = [array_val[indices].to(device) for array_val in arrays_val]
    

    # Initalize all the monitors and best found models
    best_val, best_model, best_optimizer_state, loss_acc = float('inf'), deepcopy(model), deepcopy(optimizer.state_dict()), []
    NRMS_val, NRMS_train, time_usage_train = [], [], 0. #initialize the train and val monitor
    try:
        progress_bar = tqdm(range(n_its + 1), total=n_its)
        for it_count, batch in zip(progress_bar, itter):
            ### Validation and printing step ###
            if it_count%val_freq==0: #make this an or last iteration?
                with torch.no_grad(): NRMS_val.append((val_fun(model, *arrays_val)).cpu().numpy()**0.5)
                NRMS_train.append((np.mean(loss_acc) if len(loss_acc)>0 else float('nan'))**0.5)
                loss_acc = []

                if NRMS_val[-1]<=best_val:
                    best_val, best_model, best_optimizer_state = NRMS_val[-1], deepcopy(model).cpu(), deepcopy(optimizer.state_dict()) #does this work nicely with device?
                
                #saving fit results
                samps_per_sec = it_count*batch_size/time_usage_train if time_usage_train>0 else None
                cloudpickle.dump({'best_model': best_model,            'best_optimizer_state':best_optimizer_state,\
                                  'last_model': deepcopy(model).cpu(), 'last_optimizer_state':optimizer.state_dict(),\
                                  'NRMS_train': np.array(NRMS_train),  'NRMS_val':np.array(NRMS_val),\
                                  'samples/sec': samps_per_sec, **fit_info, 'it_counter' : np.arange(len(NRMS_val))*val_freq},\
                                  open(save_filename,'wb'))
                print(f'it {it_count:7,} NRMS loss {NRMS_train[-1]:.5f} NRMS val {NRMS_val[-1]:.5f}{"!!" if NRMS_val[-1]==best_val else "  "} {(it_count*batch_size/time_usage_train if time_usage_train>0 else float("nan")):.2f} samps/sec')
            
            if it_count==n_its: break #break upon the final iteration such to skip the added iteration

            ### Train Step ###
            start_t = time.time()
            loss = train_step(model, batch, optimizer)
            time_usage_train += time.time()-start_t

            ### Post Train step ##
            if np.isnan(loss):
                print('!!!!!!!!!!!!! Loss became NaN and training will be stopped !!!!!!!!!!!!!!')
                break
            loss_acc.append(loss) # add the loss the the loss accumulator
            progress_bar.set_description(f'Sqrt loss: {loss**0.5:.5f}', refresh=False)
    except KeyboardInterrupt:
        print('Stopping early due to KeyboardInterrupt')
    d = cloudpickle.load(open(save_filename,'rb')) #save the last model to disk
    d['last_model'], d['last_optimizer_state'] = deepcopy(model).cpu(), optimizer.state_dict()
    cloudpickle.dump(d, open(save_filename,'wb'))
    model.load_state_dict(best_model.state_dict()); model.cpu()
    return d


def get_checkpoint_dir():
    '''A utility function which gets the checkpoint directory for each OS

    It creates a working directory called deepSI-checkpoints 
        in LOCALAPPDATA/deepSI-checkpoints/ for windows
        in ~/.deepSI-checkpoints/ for unix like
        in ~/Library/Application Support/deepSI-checkpoints/ for darwin

    Returns
    -------
    checkpoints_dir
    '''
    import os
    from sys import platform
    if platform == "darwin": #not tested but here it goes
        checkpoints_dir = os.path.expanduser('~/Library/Application Support/deepSI-checkpoints/')
    elif platform == "win32":
        checkpoints_dir = os.path.join(os.getenv('LOCALAPPDATA'),'deepSI-checkpoints/')
    else: #unix like, might be problematic for some weird operating systems.
        checkpoints_dir = os.path.expanduser('~/.deepSI-checkpoints/')#Path('~/.deepSI/')
    if os.path.isdir(checkpoints_dir) is False:
        os.mkdir(checkpoints_dir)
    return checkpoints_dir


def fit_minimal_implementation(model: nn.Module, train: Input_output_data,
    val: Input_output_data, n_its: int, T: int = 50, stride: int = 1, batch_size: int = 256,
    val_freq: int = 250, optimizer: optim.Optimizer = None, loss_fun=compute_NMSE):

    optimizer = optimizer or torch.optim.Adam(model.parameters())
    arrays, indices = model.create_arrays(train, T=T, stride=stride)
    itter = data_batcher(*arrays, batch_size=batch_size, indices=indices)
    arrays_val, val_indices = model.create_arrays(val, T=T)
    arrays_val = [a[val_indices] for a in arrays_val]
    
    best_val, best_model = float('inf'), deepcopy(model.state_dict())

    for it_count, batch in zip(tqdm(range(n_its)), itter):
        if it_count % val_freq == 0:  # Validation step
            val_loss = loss_fun(model, *arrays_val).sqrt().item()
            if val_loss <= best_val:
                best_val, best_model = val_loss, deepcopy(model.state_dict())
            print(f'Iter {it_count:7,}, Val Loss: {val_loss:.5f}')
        
        # Training step
        loss = loss_fun(model, *batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.load_state_dict(best_model)
    return best_model
