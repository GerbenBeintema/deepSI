
import deepSI
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset

def load_system_data(file):
    '''Load System_data from .npz file'''
    outfile = dict(np.load(file,allow_pickle=True))
    def get_sys_data(data):
        for k in data:
            if data[k].shape==tuple(): #if it is a single element
                data[k] = data[k].tolist()
        return System_data(**data)
    if outfile.get('sdl') is not None: #list of data
        return System_data_list(sys_data_list = [get_sys_data(o) for o in outfile['sdl']])
    else:
        return get_sys_data(outfile)

class IO_dataset(Dataset):
    """docstring for IO_dataset"""
    def __init__(self, u, y, na=10, nb=10):
        super(IO_dataset, self).__init__()
        self.u = u
        self.y = y
        self.na = na
        self.nb = nb
        self.k0 = max(na,nb)

    def __getitem__(self, k):
        k += self.k0
        hist = np.concatenate((self.u[k-self.nb:k].flat,self.y[k-self.na:k].flat)).astype(np.float32)
        Y = self.y[k].astype(np.float32)
        return [hist, Y]

    def __len__(self):
        return len(self.u) - self.k0

class hist_future_dataset(Dataset):
    """docstring for hist_future_dataset"""
    def __init__(self, u, y, na=10, nb=10, nf=5, force_multi_u=False, force_multi_y=False):
        super(hist_future_dataset, self).__init__()
        self.u = u
        self.y = y
        self.na = na
        self.nb = nb
        self.nf = nf
        self.k0 = max(na,nb)
        self.force_multi_u = force_multi_u
        self.force_multi_y = force_multi_y

    def __getitem__(self, k):
        k += self.k0 + self.nf
        yhist = self.y[k-self.na-self.nf:k-self.nf]
        uhist = self.u[k-self.nb-self.nf:k-self.nf]
        yfuture = self.y[k-self.nf:k]
        ufuture = self.u[k-self.nf:k]

        if self.force_multi_u and uhist.ndim==1: #(N, time_seq, nu)
            uhist = uhist[:,None]
            ufuture = ufuture[:,None]
        if self.force_multi_y and yhist.ndim==1: #(N, time_seq, ny)
            yhist = yhist[:,None]
            yfuture = yfuture[:,None]
        return [a.astype(np.float32) for a in [uhist, yhist, ufuture, yfuture]]

    def __len__(self):
        return len(self.u) - self.k0 - self.nf + 1

class ss_dataset(Dataset):
    """docstring for ss_dataset"""
    def __init__(self, u, y, nf=20, stride=1, force_multi_u=False, force_multi_y=False):
        super(ss_dataset, self).__init__()
        self.u = u
        self.y = y
        self.nf = nf
        self.force_multi_u = force_multi_u
        self.force_multi_y = force_multi_y

    def __getitem__(self, k):
        k += self.nf
        yfuture = self.y[k-self.nf:k]
        ufuture = self.u[k-self.nf:k]

        if self.force_multi_u and ufuture.ndim==1: #(N, time_seq, nu)
            ufuture = ufuture[:,None]
        if self.force_multi_y and yfuture.ndim==1: #(N, time_seq, ny)
            yfuture = yfuture[:,None]
        return [a.astype(np.float32) for a in [ufuture, yfuture]]

    def __len__(self):
        return len(self.u) - self.nf + 1

class encoder_dataset(Dataset):
    """docstring for encoder_dataset"""
    def __init__(self, u, y, na=10,nb=10,nf=5,stride=1,force_multi_u=False,force_multi_y=False):
        super(encoder_dataset, self).__init__()
        self.u = u
        self.y = y
        self.na = na
        self.nb = nb
        self.nf = nf
        self.k0 = max(na,nb)
        self.force_multi_u = force_multi_u
        self.force_multi_y = force_multi_y

    def __getitem__(self, k):
        k += self.k0 + self.nf
        hist = np.concatenate([self.u[k-self.nb-self.nf:k-self.nf].flat, self.y[k-self.na-self.nf:k-self.nf].flat])
        yfuture = self.y[k-self.nf:k]
        ufuture = self.u[k-self.nf:k]
        if self.force_multi_u and ufuture.ndim==1: #(N, time_seq, nu)
            ufuture = ufuture[:,None]
        if self.force_multi_y and yfuture.ndim==1: #(N, time_seq, ny)
            yfuture = yfuture[:,None]
        return [a.astype(np.float32) for a in [hist, ufuture, yfuture]]

    def __len__(self):
        return len(self.u) - self.k0 - self.nf + 1

class System_data(object):
    '''System Data to be used 

    Attributes
    ----------
    u : array
        The u input signal, first dimension is time. 
    y : array or None
        The y output signal, first dimension is time. It can be not present.
    x : array or None
        The x internal state signal, first dimension is time. (often unused)
    nu : None, Number, Tuple
        Number of input dimensions. 
        None -> self.u.shape == (N)
        Number -> self.u.shape == (N,nu)
        Tuple -> self.u.shape == (N,...)
    ny : None, Number, Tuple
        Number of input dimensions. 
        None -> self.u.shape == (N)
        Tuple -> self.y.shape == (N,...)
    cheat_n : int 
        Number of samples copied when simulation was applied. This number of samples will not be used to calculate RMS and such.
    multi_u : bool
        Check for multi dimensional u, used to check SISO. 
    multi_y : bool
        Check for multi dimensional y, used to check SISO. 
    normed : bool
        Check if this data set is normed (i.e. was used as norm.transform(sys_data)) mostly used for debugging. 
    '''
    def __init__(self, u=None, y=None, x=None, cheat_n=0, normed=False, dt=None):
        '''Create System data to be used in fitting, plotted and 

        Parameters
        ----------
        u : array or list or None
            Input signal with an array where the first dimension is time
        y : array or list or None
            output signal with an array where the first dimension is time
        x : array or list or None
            internal state signal with an array where the first dimension is time
        cheat_n : int
            Number of samples copied when simulation was applied. This number of samples will not be used to calculate RMS and such.
        normed : bool
            Check if this data set is normed (i.e. was used as norm.transform(sys_data)) mostly used for debugging. 
        '''
        super(System_data, self).__init__()
        assert (y is not None) or (u is not None), 'either y or u requires to be not None or'
        N_samples = len(u) if u is not None else len(y)
        
        #do not make a copy if they are already an ndarray, saves some memory
        self.u = (u if isinstance(u,np.ndarray) else np.array(u)) if u is not None else np.zeros((N_samples,0)) #if y exists than u will always exists
        self.x = (x if isinstance(x,np.ndarray) else np.array(x)) if x is not None else None
        self.y = (y if isinstance(y,np.ndarray) else np.array(y)) if y is not None else None
        self.cheat_n = cheat_n #when the real simulation starts, used in evaluation
        self.multi_u = self.u.ndim>1
        self.multi_y = self.y.ndim>1 if self.y is not None else True
        self.normed = normed
        self.dt = dt

        #checks
        if self.y is not None:
            assert self.u.shape[0]==self.y.shape[0], f'{self.u.shape[0]}!={self.y.shape[0]}'
        if self.x is not None: 
            assert self.x.shape[0]==self.y.shape[0]

    @property
    def N_samples(self):
        return self.u.shape[0]

    @property
    def ny(self):
        '''Number of output dimensions. None or number or tuple'''
        if self.y is None:
            return 0
        elif self.y.ndim==1:
            return None
        else:
            return self.y.shape[1] if self.y.ndim==2 else self.y.shape[1:]
    @property
    def nu(self):
        '''Number of input dimensions. None or number or tuple'''
        if self.u.ndim==1:
            return None
        else:
            return self.u.shape[1] if self.u.ndim==2 else self.u.shape[1:]

    @property
    def nx(self):
        '''Number of internal states. None or number or tuple'''
        if self.x is None:
            raise ValueError('No state present')
        elif self.x.ndim==1:
            return None
        else:
            return self.x.shape[1] if self.x.ndim==2 else self.x.shape[1:]

    def flatten(self):
        """Flatten both u and y to (N,nu) and (N,ny) returns a new System_data"""
        if self.y is not None and self.y.ndim>2:
            y = self.y.reshape((self.y.shape[0],-1))
        else:
            y = self.y
        if self.u is not None and self.u.ndim>2:
            u = self.u.reshape((self.u.shape[0],-1))
        else:
            u = self.u
        return System_data(u=u,y=y,x=self.x,cheat_n=self.cheat_n,normed=self.normed,dt=self.dt)

    def reshape_as(self, other):
        """Inverse of .flatten and will reshape both u and y to (N,) + other.u.shape[1:] and (N,) + other.y.shape[1:]"""
        #this can fail if either is None
        y = self.y.reshape((self.y.shape[0],)+other.y.shape[1:]) if self.y is not None else None
        u = self.u.reshape((self.u.shape[0],)+other.u.shape[1:]) if self.u is not None else None
        return System_data(u=u,y=y,x=self.x,cheat_n=self.cheat_n,normed=self.normed,dt=self.dt)


    ############################
    ###### Transformations #####
    ############################
    def to_IO_data(self,na=10,nb=10,stride=1,online_construct=False, feedthrough=False):
        '''Transforms the system data to Input-Output structure (hist,Y) with y length of na, and u length of nb

        Parameters
        ----------
        na : int
            y history considered
        nb : int
            u history considered

        Returns
        -------
        hist : ndarray of (Samples, na*ny + nb*nu)
            array of combination of historical inputs and output as [u[k-nb:k].flat,y[k-na:k].flat] for many k
        Y    : ndarray (Samples, features) or (Samples)
            array of single step ahead [y]
        '''
        if online_construct:
            return IO_dataset(self.u, self.y, na=na, nb=nb)

        u, y = np.copy(self.u), np.copy(self.y)
        hist = []
        Y = []
        for k in range(max(na,nb),len(u),stride):
            hist.append(np.concatenate((u[k-nb:k+feedthrough].flat,y[k-na:k].flat))) #size = nb*nu + na*ny
            Y.append(y[k])
        return np.array(hist), np.array(Y)

    def to_hist_future_data(self,na=10,nb=10,nf=5,stride=1,force_multi_u=False,force_multi_y=False,online_construct=False):
        '''Transforms the system data to encoder structure as structure (uhist,yhist,ufuture,yfuture) of 

        Made for simulation error and multi step error methods

        Parameters
        ----------
        na : int
            y history considered
        nb : int
            u history considered
        nf : int
            future inputs considered

        Returns
        -------
        uhist : ndarray (samples, nb, nu) or (sample, nb) if nu=None
            array of [u[k-nb],....,u[k-1]]
        yhist : ndarray (samples, na, ny) or (sample, na) if ny=None
            array of [y[k-nb],....,y[k-1]]
        ufuture : ndarray (samples, nf, nu) or (sample, nf) if nu=None
            array of [u[k],....,u[k+nf-1]]
        yfuture : ndarray (samples, nf, ny) or (sample, nf) if ny=None
            array of [y[k],....,y[k+nf-1]]
        '''
        if online_construct:
            return hist_future_dataset(self.u, self.y, na=na, nb=nb, nf=nf, force_multi_u=force_multi_u, force_multi_y=force_multi_y)

        u, y = np.copy(self.u), np.copy(self.y)
        yhist = []
        uhist = []
        ufuture = []
        yfuture = []
        for k in range(max(nb,na)+nf,len(u)+1,stride):
            yhist.append(y[k-na-nf:k-nf])
            uhist.append(u[k-nb-nf:k-nf])
            yfuture.append(y[k-nf:k])
            ufuture.append(u[k-nf:k])
        uhist, yhist, ufuture, yfuture = np.array(uhist), np.array(yhist), np.array(ufuture), np.array(yfuture)
        if force_multi_u and uhist.ndim==2: #(N, time_seq, nu)
            uhist = uhist[:,:,None]
            ufuture = ufuture[:,:,None]
        if force_multi_y and yhist.ndim==2: #(N, time_seq, ny)
            yhist = yhist[:,:,None]
            yfuture = yfuture[:,:,None]
        return uhist, yhist, ufuture, yfuture


    def to_ss_data(self,nf=20,stride=1,force_multi_u=False,force_multi_y=False,online_construct=False):
        if online_construct:
            return ss_dataset(self.u, self.y, nf=nf, force_multi_u=force_multi_u,force_multi_y=force_multi_y)

        u, y = np.copy(self.u), np.copy(self.y)
        ufuture = []
        yfuture = []
        for k in range(nf,len(u)+1,stride):
            yfuture.append(y[k-nf:k])
            ufuture.append(u[k-nf:k])
        ufuture, yfuture = np.array(ufuture),np.array(yfuture)
        if force_multi_u and ufuture.ndim==2: #(uhist, time_seq, nu)
            ufuture = ufuture[:,:,None]
        if force_multi_y and yfuture.ndim==2: #(yhist, time_seq, ny)
            yfuture = yfuture[:,:,None]
        return ufuture, yfuture


    def to_encoder_data(self,na=10,nb=10,nf=5,stride=1,force_multi_u=False,force_multi_y=False,online_construct=False):
        '''Transforms the system data to encoder structure as structure (hist,ufuture,yfuture) of 

        Parameters
        ----------
        na : int
            y history considered
        nb : int
            u history considered
        nf : int
            future inputs considered
        stride : int
            skipping data for smaller data set.
        force_multi_u : bool
            converts to ufuture to #(samples, time_seq, nu) always
        force_multi_y : bool
            converts to yfuture to #(samples, time_seq, ny) always

        Returns
        -------
        hist : ndarray (samples, ny*na + nu*nb)
            array of concat of u[k-nb-nf:k-nf].flat and y[k-na-nf:k-nf].flat
        ufuture : ndarray (samples, nf, nu) or (sample, nf) if nu=None
            array of [u[k],....,u[k+nf-1]]
        yfuture :  ndarray (samples, nf, ny) or (sample, nf) if ny=None
            array of [y[k],....,y[k+nf-1]]
        '''
        if online_construct:
            return encoder_dataset(self.u, self.y, na=na,nb=nb,nf=nf,stride=stride,force_multi_u=force_multi_u,force_multi_y=force_multi_y)
        u, y = np.copy(self.u), np.copy(self.y)
        hist = []
        ufuture = []
        yfuture = []
        for k in range(max(nb,na)+nf,len(u)+1,stride):
            hist.append(np.concatenate((u[k-nb-nf:k-nf].flat,y[k-na-nf:k-nf].flat)))
            yfuture.append(y[k-nf:k])
            ufuture.append(u[k-nf:k])
        hist, ufuture, yfuture = np.array(hist),np.array(ufuture),np.array(yfuture)
        if force_multi_u and ufuture.ndim==2: #(uhist, time_seq, nu)
            ufuture = ufuture[:,:,None]
        if force_multi_y and yfuture.ndim==2: #(yhist, time_seq, ny)
            yfuture = yfuture[:,:,None]
        return hist, ufuture, yfuture

    def to_video(self, file_name='video.mp4', scale_factor=10, vmin=0, vmax=1, fps=60):
        '''Used cv2 to create a video from y of shape y.shape = (frames, ny1, ny2)'''
        import cv2
        from PIL import Image

        if not file_name.endswith('.mp4'):
            file_name += '.mp4'

        nx,ny = self.y.shape[1], self.y.shape[2] #resolution of simulation
        nx_out,ny_out = round(nx*scale_factor),round(ny*scale_factor) #resolution of video produced

        video = cv2.VideoWriter(file_name, 0, fps, (ny_out,nx_out))

        resize = lambda x: np.array(Image.fromarray(x).resize((ny_out, nx_out)))
        to_img = lambda x: resize((((np.clip(x,vmin,vmax) - vmin)/(vmax - vmin)).copy()[:,:,None]*255*np.ones((1,1,3))).astype(np.uint8))
        try: 
            for yi in self.y:
                video.write(to_img(yi))
        finally:
            cv2.destroyAllWindows()
            video.release()

    def plot_state_space_3d(self,nmax=2000,kernal=None,interpol_n=10):
        from scipy.interpolate import interp1d
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection='3d')
        if self.nx>3:
            if kernal is None:
                from sklearn.decomposition import KernelPCA
                kernal = KernelPCA(n_components=3,fit_inverse_transform=True)
            kernal.fit(self.x)
            F = kernal.transform(self.x[:nmax])
        else:
            F = self.x[:nmax]
        f = interp1d(np.arange(F.shape[0]),F.T,kind='quadratic')
        out = f(np.arange(0,F.shape[0]-1,1/interpol_n))
        ax.plot(out[0],out[1],out[2])
        # plt.show()
        return kernal

    def save(self,file):
        '''Saves data with savez, see also load_system_data'''
        np.savez(file, u=self.u, x=self.x, y=self.y, cheat_n=self.cheat_n, normed=self.normed)

    def __repr__(self):
        return f'System_data of length: {self.N_samples} nu={self.nu} ny={self.ny} normed={self.normed} dt={self.dt}'

    def plot(self,show=False):
        '''Very simple plotting function'''
        plt.ylabel('y' if self.y is not None else 'u')
        plt.xlabel('t')

        tar = np.arange(self.u.shape[0])*(1 if self.dt is None else self.dt)
        plt.plot(tar, self.y.reshape((self.y.shape[0],-1)) if self.y is not None else self.u)
        if show: plt.show()

    def BFR(self,real,multi_average=True):
        '''Best Fit Rate in percent i.e. 100*(1 - np.sum((y-yhat)**2)**0.5/np.std(y)) (100 = best possible fit)'''
        # y, yhat = real.y[self.cheat_n:], self.y[self.cheat_n:]
        # return 100*(1 - np.sum((y-yhat)**2)**0.5/np.sum((y-np.mean(y))**2)**0.5)
        return 100*(1 - self.NRMS(real,multi_average=multi_average))

    def NRMS(self,real,multi_average=True,per_channel=True):
        '''Normalized root mean square i.e. np.sum((y-yhat)**2)**0.5/np.std(y) (0 = best fit possible)'''
        RMS = self.RMS(real,multi_average=False) #RMS list
        if per_channel:
            y_std = np.std(real.y,axis=0) #this breaks when real.y is constant but self is not
        else:
            y_std = np.std(real.y)
            return np.mean(RMS/y_std) if multi_average else RMS/y_std
        
        if y_std.ndim==0: #if there is only one dim than normal equation
            return RMS/y_std

        filt = y_std<1e-14 #filter constant stds
        if np.all(filt):
            raise ValueError('all signals are constant')
        if np.any(filt) and multi_average:
            #todo insert warning on exclusion
            y_std = y_std[np.logical_not(filt)]
            RMS = RMS[np.logical_not(filt)]
        return np.mean(RMS/y_std) if multi_average else RMS/y_std

    def NRMS_per_channel(self,real,multi_average=True):
        '''Normalized root mean square i.e. np.sum((y-yhat)**2)**0.5/np.std(y) (0 = best fit possible)'''
        return self.NRMS(real,multi_average=multi_average,per_channel=True)

    def NRMS_mean_channels(self,real,multi_average=True):
        '''Normalized root mean square i.e. np.sum((y-yhat)**2)**0.5/np.std(y) (0 = best fit possible)'''
        return self.NRMS(real,multi_average=multi_average,per_channel=False)

    def RMS(self,real, multi_average=True):
        '''Root mean square error'''
        #Variance accounted for
        #y output system
        #yhat output model
        y, yhat = real.y[self.cheat_n:], self.y[self.cheat_n:]
        return np.mean((y-yhat)**2)**0.5 if multi_average else np.mean((y-yhat)**2,axis=0)**0.5

    def VAF(self,real,multi_average=True):
        '''Variance accounted also known as R^2 

        calculated as 100*(1 - np.mean((y-yhat)**2)/np.std(y)**2) (100 = best possible fit)'''
        return 100*(1-self.NRMS(real,multi_average=multi_average)**2)

    def __sub__(self,other): 
        '''Calculate the difference between y two System_data a number or array'''
        if isinstance(other, System_data):
            assert len(self.y)==len(other.y), 'both arguments need to be the same length'
            return System_data(u=self.u, x=self.x, y=self.y-other.y, cheat_n=self.cheat_n,dt=self.dt)
        else:
            return System_data(u=self.u, x=self.x, y=self.y-other, cheat_n=self.cheat_n,dt=self.dt)


    def train_test_split(self,split_fraction=0.25):
        '''returns 2 data sets of length n*(1-split_fraction) and n*split_fraction respectively (left, right) split'''
        n_samples = self.u[self.cheat_n:].shape[0]
        split_n = int(n_samples*(1 - split_fraction))
        ul,ur,yl,yr = self.u[self.cheat_n:split_n], self.u[self.cheat_n+split_n:], \
                        self.y[self.cheat_n:split_n], self.y[self.cheat_n+split_n:]
        if self.x is None:
            xl,xr = None,None
        else:
            xl,xr = self.x[:split_n], self.x[split_n:]
        left_data = System_data(u=ul, x=xl, y=yl, normed=self.normed,dt=self.dt)
        right_data = System_data(u=ur, x=xr, y=yr, normed=self.normed,dt=self.dt)
        return left_data, right_data

    def __getitem__(self,arg):
        '''Slice the System_data in time index'''
        assert isinstance(arg,slice),'Please use a slice (e.g. sys_data[20:100]) or use sys_data.u or sys_data.y'
        start, stop, step = arg.indices(self.u.shape[0])
        cheat_n = max(0,self.cheat_n-start)
        unew = self.u[arg]
        ynew = self.y[arg] if self.y is not None else None
        xnew = self.x[arg] if self.x is not None else None
        return System_data(u=unew, y=ynew, x=xnew, cheat_n=cheat_n, normed=self.normed,dt=self.dt)
    
    def __len__(self):
        '''Number of samples len(system_data) = self.N_samples'''
        return self.N_samples

    def down_sample_by_average(self,factor):
        """Down sample method

        Parameters
        ----------
        factor : int
            length will be (original length)/factor        
        """
        assert isinstance(factor, int) 
        L = self.N_samples
        n = (L//factor)*factor
        u,y = self.u, self.y
        if u.ndim==1:
            u = np.mean(self.u[:n].reshape((-1,factor)),axis=1)
        else:
            u = np.stack([np.mean(self.u[:n,i].reshape((-1,factor)),axis=1) for i in range(self.u.shape[1])],axis=1)
        if y.ndim==1:
            y = np.mean(self.y[:n].reshape((-1,factor)),axis=1)
        else:
            y = np.stack([np.mean(self.y[:n,i].reshape((-1,factor)),axis=1) for i in range(self.y.shape[1])],axis=1)
        return System_data(u=u,y=y,x=self.x[::factor] if self.x is not None else None,cheat_n=self.cheat_n//factor,normed=self.normed,dt=self.dt)

    #scipy.signal.decimate lookup
    #other downsample methods
    def down_sample_by_decimate(self,factor):
        """Down sample method
        Parameters
        ----------
        factor : int
            length will be (original length)/factor        
        """
        from scipy.signal import decimate
        u = decimate(self.u.T,factor).T
        y = decimate(self.y.T,factor).T
        return System_data(u=u,y=y,x=None,cheat_n=self.cheat_n,normed=self.normed,dt=self.dt) #todo add x


    def down_sample_by_MIMO(self,factor):
        """Down sample method

        Parameters
        ----------
        factor : int
            length will be (original length)/factor        
        """
        N = self.u.shape[0]
        u = self.u[:N-(N%factor)] # up scaling will be problematic
        y = self.y[:N-(N%factor):factor]
        u = u.reshape(u.shape[0]//factor,-1)
        return System_data(u=u,y=y,x=None,cheat_n=self.cheat_n//factor,normed=self.normed,dt=self.dt) #todo add x




class System_data_list(System_data):
    '''A list of System_data, has most methods of System_data in a list form with only some exceptions listed below

    Attributes
    ----------
    sdl : list of System_data
    y : array
        concatenated y of system_data contained in sdl
    u : array
        concatenated u of system_data contained in sdl

    Methods
    -------
    append(System_data) adds element to sdl
    extend(list) adds elements to sdl
    __getitem__(number) get (ith system data, time slice) 
    '''
    def __init__(self,sys_data_list = None):
        self.sdl = [] if sys_data_list is None else sys_data_list
        self.sanity_check()
    def sanity_check(self):
        for sys_data in self.sdl:
            assert isinstance(sys_data,System_data)
            assert sys_data.ny==self.ny
            assert sys_data.nu==self.nu
            assert sys_data.normed==self.normed
    @property
    def normed(self):
        return self.sdl[0].normed
    @property
    def N_samples(self):
        return sum(sys_data.u.shape[0] for sys_data in self.sdl)
    @property
    def ny(self):
        return self.sdl[0].ny
    @property
    def nu(self):
        return self.sdl[0].nu
    @property
    def y(self): #concatenate or list of lists
        return np.concatenate([sd.y for sd in self.sdl],axis=0)
    @property
    def u(self): #concatenate or list of lists
        return np.concatenate([sd.u for sd in self.sdl],axis=0)    
    @property
    def n_cheat(self):
        return self.sdl[0].n_cheat
    @property
    def dt(self):
        return self.sdl[0].dt
    def flatten(self):
        return System_data_list([sdli.flatten() for sdli in self.sdl])
    def reshape_as(self, other):
        """Inverse of .flatten and will reshape both u and y to (N,) + other.u.shape[1:] and (N,) + other.y.shape[1:]"""
        if isinstance(other,System_data_list):
            return System_data_list([sd.reshape_as(sdo) for sd,sdo in zip(self.sdl,other.sdl)])
        else:
            return System_data_list([sd.reshape_as(other) for sd in self.sdl])

    ## Transformations ##
    def to_IO_data(self,na=10,nb=10,stride=1,online_construct=False,feedthrough=False):
        out = [sys_data.to_IO_data(na=na,nb=nb,stride=stride,online_construct=online_construct,feedthrough=feedthrough) for sys_data in self.sdl]  #((I,ys),(I,ys))
        return [np.concatenate(o,axis=0) for o in  zip(*out)] if not online_construct else ConcatDataset(out) #(I,I,I),(ys,ys,ys)
    def to_hist_future_data(self,na=10,nb=10,nf=5,stride=1,force_multi_u=False,force_multi_y=False,online_construct=False):
        out = [sys_data.to_hist_future_data(na=na,nb=nb,nf=nf,stride=stride,force_multi_u=force_multi_u,\
                force_multi_y=force_multi_y,online_construct=online_construct) for sys_data in self.sdl]  #((I,ys),(I,ys))
        return [np.concatenate(o,axis=0) for o in zip(*out)] if not online_construct else ConcatDataset(out) #(I,I,I),(ys,ys,ys)
    def to_ss_data(self,nf=20,stride=1,force_multi_u=False,force_multi_y=False,online_construct=False):
        out = [sys_data.to_ss_data(nf=nf,stride=stride,force_multi_u=force_multi_u,\
                force_multi_y=force_multi_y,online_construct=online_construct) for sys_data in self.sdl]  #((I,ys),(I,ys))
        return [np.concatenate(o,axis=0) for o in zip(*out)] if not online_construct else ConcatDataset(out) #(I,I,I),(ys,ys,ys)
    def to_encoder_data(self,na=10,nb=10,nf=5,stride=1,force_multi_u=False,force_multi_y=False,online_construct=False):
        out = [sys_data.to_encoder_data(na=na,nb=nb,nf=nf,stride=stride,force_multi_u=force_multi_u,\
                force_multi_y=force_multi_y,online_construct=online_construct) for sys_data in self.sdl]  #((I,ys),(I,ys))
        return [np.concatenate(o,axis=0) for o in zip(*out)] if not online_construct else ConcatDataset(out) #(I,I,I),(ys,ys,ys)

    def save(self,file):
        '''Saves data'''
        out = [dict(u=sd.u, x=sd.x, y=sd.y, cheat_n=sd.cheat_n, normed=sd.normed, dt=sd.dt) for sd in self.sdl]
        np.savez(file, sdl=out)

    def __repr__(self):
        if len(self)==0:
            return f'System_data_list with {len(self.sdl)} series'
        else:
            return f'System_data_list with {len(self.sdl)} series and total length {self.N_samples}, nu={self.nu}, ny={self.ny}, normed={self.normed} lengths={[sd.N_samples for sd in self.sdl]} dt={self.sdl[0].dt}'

    def plot(self,show=False):
        '''Very simple plotting function'''
        for sd in self.sdl:
            sd.plot()
        if show: plt.show()

    def weighted_mean(self,vals):
        return np.average(vals,axis=0,weights=[sd.N_samples for sd in self.sdl])

    def RMS(self,real, multi_average=True):
        return self.weighted_mean([sd.RMS(sdo,multi_average=multi_average) for sd,sdo in zip(self.sdl,real.sdl)])

    def __sub__(self,other):
        if isinstance(other,System_data_list):            
            return System_data_list([System_data(u=sd.u, x=sd.x, y=sd.y-sdo.y, cheat_n=sd.cheat_n, dt=sd.dt) for sd, sdo in zip(self.sdl,other.sdl)])
        elif isinstance(other,(float,int,np.ndarray,System_data)):
            if isinstance(other, System_data):
                other = other.y
            return System_data_list([System_data(u=sd.u, x=sd.x, y=sd.y-other, cheat_n=sd.cheat_n, dt=sd.dt) for sd in self.sdl])

    def train_test_split(self,split_fraction=0.25):
        '''return 2 data sets of length n*(1-split_fraction) and n*split_fraction respectively (left, right) split'''
        out = list(zip(*[sd.train_test_split(split_fraction=split_fraction) for sd in self.sdl]))
        left, right = System_data_list(out[0]), System_data_list(out[1])
        return left, right

    def __getitem__(self,arg): #by data set or by time?
        '''get (ith system data, time slice) '''
        if isinstance(arg,tuple) and len(arg)>1: #
            sdl_sub = self.sdl[arg[0]]
            if isinstance(sdl_sub,System_data):
                return sdl_sub[arg[1]]
            else: #sdl_sub is a list
                return System_data_list([sd[arg[1]] for sd in sdl_sub])
        else:
            if isinstance(arg,int): #sdl[i] -> ith data system set
                return self.sdl[arg]
            else: #slice of something
                return System_data_list(self.sdl[arg])

    def __len__(self): #number of datasets
        return len(self.sdl)

    def down_sample_by_average(self,factor):
        return System_data_list([sd.down_sample_by_average(factor) for sd in self.sdl])

    def append(self,other):
        assert isinstance(other, System_data)
        self.sdl.append(other)
        self.sanity_check()

    def extend(self,other):
        if isinstance(other,(list,tuple)):
            other = System_data_list(other)
        self.sdl.extend(other.sdl)
        self.sanity_check()


    def down_sample_by_average(self,factor):
        """Down sample method

        Parameters
        ----------
        factor : int
            length will be (original length)/factor        
        """
        return System_data_list([s.down_sample_by_average(factor) for s in self.sdl])


    #scipy.signal.decimate lookup
    #other downsample methods
    def down_sample_by_decimate(self,factor):
        """Down sample method
        
        Parameters
        ----------
        factor : int
            length will be (original length)/factor        
        """
        return System_data_list([s.down_sample_by_decimate(factor) for s in self.sdl])


    def down_sample_by_MIMO(self,factor):
        """Down sample method

        Parameters
        ----------
        factor : int
            length will be (original length)/factor        
        """
        return System_data_list([s.down_sample_by_MIMO(factor) for s in self.sdl])

class System_data_norm(object):
    '''A utility to normalize system_data before fitting or usage

    Attributes
    ----------
    u0 : float or array
        average u to be subtracted
    ustd : float or array
        standard divination of u to be divided by
    y0 : float or array
        average y to be subtracted
    ystd : float or array
        standard divination of y to be divided by
    '''
    def __init__(self, u0=0, ustd=1, y0=0, ystd=1):
        self.u0 = u0
        self.ustd = ustd
        self.y0 = y0
        self.ystd = ystd

    @property
    def is_id(self):
        return np.all(self.u0==0) and np.all(self.ustd==1) and np.all(self.y0==0) and np.all(self.ystd==1)

    def make_training_data(self,sys_data):
        if isinstance(sys_data,(list,tuple)):
            out = [self.make_training_data(s) for s in sys_data]
            return [np.concatenate(a,axis=0) for a in zip(*out)] #transpose + concatenate
        return sys_data.u, sys_data.y

    def fit(self,sys_data):
        '''set the values of u0, ustd, y0 and ystd using sys_data (can be a list) given'''
        u, y = self.make_training_data(sys_data)
        self.u0 = np.mean(u,axis=0)
        self.ustd = np.std(u,axis=0) + 1e-15 #does this work with is_id?
        self.y0 = np.mean(y,axis=0)
        self.ystd = np.std(y,axis=0) + 1e-15
        
    def transform(self,sys_data):
        '''Transform the data by 
           u <- (sys_data.u-self.u0)/self.ustd
           y <- (sys_data.y-self.y0)/self.ystd

        Parameters
        ----------
        sys_data : System_data
            sys_data to be transformed

        Returns
        -------
        System_data or System_data_list if a list was given
        '''
        if isinstance(sys_data,(list,tuple)):
            assert sys_data[0].normed==False, 'System_data is already normalized'
            return System_data_list([self.transform(s) for s in sys_data]) #conversion?

        
        if isinstance(sys_data,System_data_list):
            assert sys_data.normed==False, 'System_data is already normalized'
            return System_data_list([self.transform(s) for s in sys_data.sdl])
        
        if self.is_id:
            return System_data(u=sys_data.u,x=sys_data.x,y=sys_data.y, \
                               cheat_n=sys_data.cheat_n,normed=True,dt=sys_data.dt)

        if isinstance(sys_data,System_data):
            assert sys_data.normed==False, 'System_data is already normalized'
            u_transformed = (sys_data.u-self.u0)/self.ustd if sys_data.u is not None else None
            y_transformed = (sys_data.y-self.y0)/self.ystd if sys_data.y is not None else None
            return System_data(u=u_transformed,x=sys_data.x,y=y_transformed, \
                cheat_n=sys_data.cheat_n,normed=True,dt=sys_data.dt)
        else:
            raise NotImplementedError(f'type={type(sys_data)} cannot yet be transformed by norm')

    def inverse_transform(self,sys_data):
        '''Inverse Transform the data by 
           u <- sys_data.u*self.ustd+self.u0
           y <- sys_data.y*self.ystd+self.y0

        Parameters
        ----------
        sys_data : System_data

        Returns
        -------
        System_data or System_data_list if a list was given
        '''

        if isinstance(sys_data,(list,tuple)):
            return System_data_list([self.inverse_transform(s) for s in sys_data])
        elif isinstance(sys_data,System_data_list):
            assert sys_data.normed==True, 'System_data is already un-normalized'
            return System_data_list([self.inverse_transform(s) for s in sys_data.sdl])
        
        if self.is_id:
            return System_data(u=sys_data.u,x=sys_data.x,y=sys_data.y, \
                               cheat_n=sys_data.cheat_n,normed=False,dt=sys_data.dt)

        if isinstance(sys_data,System_data):
            assert sys_data.normed==True, 'System_data is already un-normalized'
            u_inv_transformed = sys_data.u*self.ustd + self.u0 if sys_data.u is not None else None
            y_inv_transformed = sys_data.y*self.ystd + self.y0 if sys_data.y is not None else None
            return System_data(u=u_inv_transformed,x=sys_data.x,y=y_inv_transformed,
                               cheat_n=sys_data.cheat_n,normed=False,dt=sys_data.dt)
        else:
            raise NotImplementedError(f'type={type(sys_data)} cannot yet be inverse_transform by norm')

    def __repr__(self):
        return f'System_data_norm: (u0={self.u0}, ustd={self.ustd}, y0={self.y0}, ystd={self.ystd})'


if __name__=='__main__':
    # tests

    np.random.seed(42)
    sys_data = System_data(u=np.random.normal(scale=2,size=(100,2)),y=np.random.normal(scale=1.5,size=(100,2)))
    sys_data2 = System_data(u=np.random.normal(size=(100,2)),y=np.random.normal(size=(100,2)))
    sys_data3 = System_data(u=np.random.normal(size=(100,2)),y=np.random.normal(size=(100,2)))
    sdl = System_data_list([sys_data,sys_data2])

    dataset = sdl.to_encoder_data(online_construct=True)
    dataset = DataLoader(dataset, batch_size=40, drop_last=True, shuffle=True)
    for hist, ufuture,yfuture in dataset:
        print(hist.shape, ufuture.dtype,yfuture.shape)

    # print(sys_data.NRMS(sys_data2,multi_average=False))
    # print([a.shape for a in sys_data.to_encoder_data(5,7,10)])
    # print(sys_data2[10:20])

    # sdl = System_data_list([sys_data,sys_data2,sys_data3])

    # print(sdl.to_encoder_data(9)[0].shape)
    # print(len(sdl),sdl.N_samples)
    # # sdl.plot(show=True)
    # print(sdl.train_test_split())
    # print(sdl.down_sample_by_average(10))
    # print(sdl.VAF(sdl))
    # norm = System_data_norm()
    # norm.fit(sdl)
    # sdl_transformed = norm.transform(sdl)
    # sys_data_transformed = norm.transform(sys_data)
    # sys_data12_transformed = norm.transform((sys_data,sys_data2))
    # print('transformed:',sdl_transformed,sys_data_transformed,sys_data12_transformed)

    # sdl3 = norm.inverse_transform(sdl_transformed)
    # print(sdl_transformed.NRMS(sdl))
    # print(sdl3.NRMS(sdl))
    # print(np.std(sdl_transformed.y,axis=0))
    # print('yshape=',sdl_transformed.y.shape)

    # print(sdl[1:2,:-10])
    # print(len(sdl))

    # sys_data = System_data(u=np.random.normal(scale=2,size=(100,2,2)),y=np.random.normal(scale=1.5,size=(100,2,5)))
    # sdl = System_data_list([sys_data,sys_data])
    # print('sdl',sdl)
    # print(sdl.NRMS(sdl))
    # sdl_flat = sdl.flatten()
    # print(sdl_flat)
    # print(sdl_flat.reshape_as(sdl))

    sys_data = System_data(u=np.random.normal(scale=2,size=(100,2)),y=np.random.normal(scale=1.5,size=(100,2)),dt=0.1)
    print(sys_data)
    sys_data.plot(show=True)
