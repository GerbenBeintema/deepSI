
import deepSI
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

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
    def __init__(self, u=None, y=None, x=None, cheat_n=0, normed=False):
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
        
        self.u = np.array(u) if u is not None else np.zeros((N_samples,0)) #if y exists than u will always exists
        self.x = np.array(x) if x is not None else None
        self.y = np.array(y) if y is not None else None
        self.cheat_n = cheat_n #when the real simulation starts, used in evaluation
        self.multi_u = self.u.ndim>1
        self.multi_y = self.y.ndim>1 if self.y is not None else True
        self.normed = normed

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
        return System_data(u=u,y=y,x=self.x,cheat_n=self.cheat_n,normed=self.normed)

    def reshape_as(self, other):
        """Inverse of .flatten and will reshape both u and y to (N,) + other.u.shape[1:] and (N,) + other.y.shape[1:]"""
        #this can fail if either is None
        y = self.y.reshape((self.y.shape[0],)+other.y.shape[1:]) if self.y is not None else None
        u = self.u.reshape((self.u.shape[0],)+other.u.shape[1:]) if self.u is not None else None
        return System_data(u=u,y=y,x=self.x,cheat_n=self.cheat_n,normed=self.normed)


    ############################
    ###### Transformations #####
    ############################
    def to_IO_data(self,na=10,nb=10,dilation=1):
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
        u, y = np.copy(self.u), np.copy(self.y)
        hist = []
        Y = []
        for k in range(max(na,nb),len(u),dilation):
            hist.append(np.concatenate((u[k-nb:k].flat,y[k-na:k].flat))) #size = nb*nu + na*ny
            Y.append(y[k])
        return np.array(hist), np.array(Y)

    def to_hist_future_data(self,na=10,nb=10,nf=5,dilation=1,force_multi_u=False,force_multi_y=False):
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
        yhist : ndarray (samples, na, nu) or (sample, na) if nu=None
            array of [y[k-nb],....,y[k-1]]
        ufuture : ndarray (samples, nf, nu) or (sample, nf) if nu=None
            array of [u[k],....,u[k+nf-1]]
        yfuture : ndarray (samples, nf, ny) or (sample, nf) if ny=None
            array of [y[k],....,y[k+nf-1]]
        '''
        u, y = np.copy(self.u), np.copy(self.y)
        yhist = []
        uhist = []
        ufuture = []
        yfuture = []
        for k in range(max(nb,na)+nf,len(u)+1,dilation):
            yhist.append(y[k-na-nf:k-nf])
            uhist.append(u[k-nb-nf:k-nf])
            yfuture.append(y[k-nf:k])
            ufuture.append(u[k-nf:k])
        uhist, yhist, ufuture, yfuture = np.array(uhist), np.array(yhist), np.array(ufuture), np.array(yfuture)
        if force_multi_u and uhist.ndim==2: #(uhist, time_seq, nu)
            uhist = uhist[:,:,None]
            ufuture = ufuture[:,:,None]
        if force_multi_y and yhist.ndim==2: #(yhist, time_seq, ny)
            yhist = yhist[:,:,None]
            yfuture = yfuture[:,:,None]
        return uhist, yhist, ufuture, yfuture


    def to_ss_data(self,nf=20,dilation=1,force_multi_u=False,force_multi_y=False):
        u, y = np.copy(self.u), np.copy(self.y)
        ufuture = []
        yfuture = []
        for k in range(nf,len(u)+1,dilation):
            yfuture.append(y[k-nf:k])
            ufuture.append(u[k-nf:k])
        ufuture, yfuture = np.array(ufuture),np.array(yfuture)
        if force_multi_u and ufuture.ndim==2: #(uhist, time_seq, nu)
            ufuture = ufuture[:,:,None]
        if force_multi_y and yfuture.ndim==2: #(yhist, time_seq, ny)
            yfuture = yfuture[:,:,None]
        return ufuture, yfuture

    def to_encoder_data(self,na=10,nb=10,nf=5,dilation=1,force_multi_u=False,force_multi_y=False):
        '''Transforms the system data to encoder structure as structure (hist,ufuture,yfuture) of 

        Parameters
        ----------
        na : int
            y history considered
        nb : int
            u history considered
        nf : int
            future inputs considered
        dilation : int
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
        u, y = np.copy(self.u), np.copy(self.y)
        hist = []
        ufuture = []
        yfuture = []
        for k in range(max(nb,na)+nf,len(u)+1,dilation):
            hist.append(np.concatenate((u[k-nb-nf:k-nf].flat,y[k-na-nf:k-nf].flat)))
            yfuture.append(y[k-nf:k])
            ufuture.append(u[k-nf:k])
        hist, ufuture, yfuture = np.array(hist),np.array(ufuture),np.array(yfuture)
        if force_multi_u and ufuture.ndim==2: #(uhist, time_seq, nu)
            ufuture = ufuture[:,:,None]
        if force_multi_y and yfuture.ndim==2: #(yhist, time_seq, ny)
            yfuture = yfuture[:,:,None]
        return hist, ufuture, yfuture

    def to_video(self, file_name='video.mp4', scale_factor=10, vmin=0, vmax=1):
        '''Used cv2 to create a video from y of shape y.shape = (frames, ny1, ny2)'''
        import cv2
        from PIL import Image

        if not file_name.endswith('.mp4'):
            file_name += '.mp4'

        nx,ny = self.y.shape[1],self.y.shape[2] #resolution of simulation
        nx_out,ny_out = round(nx*scale_factor),round(ny*scale_factor) #resolution of video produced

        video = cv2.VideoWriter(file_name, 0, 60, (ny_out,nx_out))

        resize = lambda x: np.array(Image.fromarray(x).resize((ny_out, nx_out)))
        to_img = lambda x: resize((np.clip(x,vmin,vmax).copy()[:,:,None]*255*np.ones((1,1,3))).astype(np.uint8))
        try: 
            for yi in self.y:
                video.write(to_img(yi))
        finally:
            cv2.destroyAllWindows()
            video.release()


    def save(self,file):
        '''Saves data with savez, see also load_system_data'''
        np.savez(file, u=self.u, x=self.x, y=self.y, cheat_n=self.cheat_n, normed=self.normed)

    def __repr__(self):
        return f'System_data of length: {self.N_samples} nu={self.nu} ny={self.ny} normed={self.normed}'

    def plot(self,show=False):
        '''Very simple plotting function'''
        plt.ylabel('y' if self.y is not None else 'u')
        plt.xlabel('t')

        plt.plot(self.y.reshape((self.y.shape[0],-1)) if self.y is not None else self.u)
        if show: plt.show()

    def BFR(self,real,multi_average=True):
        '''Best Fit Rate in percent i.e. 100*(1 - np.sum((y-yhat)**2)**0.5/np.std(y)) (100 = best possible fit)'''
        # y, yhat = real.y[self.cheat_n:], self.y[self.cheat_n:]
        # return 100*(1 - np.sum((y-yhat)**2)**0.5/np.sum((y-np.mean(y))**2)**0.5)
        return 100*(1 - self.NRMS(real,multi_average=multi_average))

    def NRMS(self,real,multi_average=True):
        '''Normalized root mean square i.e. np.sum((y-yhat)**2)**0.5/np.std(y) (0 = best fit possible)'''
        RMS = self.RMS(real,multi_average=False) #RMS list
        y_std = np.std(real.y,axis=0) #this breaks when real.y is constant but self is not
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
            return System_data(u=self.u, x=self.x, y=self.y-other.y, cheat_n=self.cheat_n)
        else:
            return System_data(u=self.u, x=self.x, y=self.y-other, cheat_n=self.cheat_n)


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
        left_data = System_data(u=ul, x=xl, y=yl, normed=self.normed)
        right_data = System_data(u=ur, x=xr, y=yr, normed=self.normed)
        return left_data, right_data

    def __getitem__(self,arg):
        '''Slice the System_data in time index'''
        assert isinstance(arg,slice),'Please use a slice (e.g. sys_data[20:100]) or use sys_data.u or sys_data.y'
        start, stop, step = arg.indices(self.u.shape[0])
        cheat_n = max(0,self.cheat_n-start)
        unew = self.u[arg]
        ynew = self.y[arg] if self.y is not None else None
        xnew = self.x[arg] if self.x is not None else None
        return System_data(u=unew, y=ynew, x=xnew, cheat_n=cheat_n, normed=self.normed)
    
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
        return System_data(u=u,y=y,x=self.x[::factor] if self.x is not None else None,cheat_n=self.cheat_n//factor,normed=self.normed)

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
        return System_data(u=u,y=y,x=None,cheat_n=self.cheat_n,normed=self.normed) #todo add x


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
        return System_data(u=u,y=y,x=None,cheat_n=self.cheat_n//factor,normed=self.normed) #todo add x




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
    def __init__(self,sys_data_list):
        assert len(sys_data_list)>0, 'At least one data set should be provided'
        ny = sys_data_list[0].ny
        nu = sys_data_list[0].nu
        normed = sys_data_list[0].normed
        for sys_data in sys_data_list:
            assert isinstance(sys_data,System_data)
            assert sys_data.ny==ny
            assert sys_data.nu==nu
            assert sys_data.normed==normed
        self.sdl = sys_data_list
        self.normed = normed
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
        return self.sdl.n_cheat
    def flatten(self):
        return System_data_list([sdli.flatten() for sdli in self.sdl])
    def reshape_as(self, other):
        """Inverse of .flatten and will reshape both u and y to (N,) + other.u.shape[1:] and (N,) + other.y.shape[1:]"""
        if isinstance(other,System_data_list):
            return System_data_list([sd.reshape_as(sdo) for sd,sdo in zip(self.sdl,other.sdl)])
        else:
            return System_data_list([sd.reshape_as(other) for sd in self.sdl])

    ## Transformations ##
    def to_IO_data(self,na=10,nb=10,dilation=1):
        #normed check?
        out = [sys_data.to_IO_data(na=na,nb=nb,dilation=1) for sys_data in self.sdl]  #((I,ys),(I,ys))
        return [np.concatenate(o,axis=0) for o in  zip(*out)] #(I,I,I),(ys,ys,ys)
    def to_hist_future_data(self,na=10,nb=10,nf=5,dilation=1,force_multi_u=False,force_multi_y=False):
        out = [sys_data.to_hist_future_data(na=na,nb=nb,nf=nf,dilation=dilation,force_multi_u=force_multi_u,force_multi_y=force_multi_y) for sys_data in self.sdl]  #((I,ys),(I,ys))
        return [np.concatenate(o,axis=0) for o in  zip(*out)] #(I,I,I),(ys,ys,ys)
    def to_ss_data(self,nf=20,dilation=1,force_multi_u=False,force_multi_y=False):
        out = [sys_data.to_ss_data(nf=nf,dilation=dilation,force_multi_u=force_multi_u,force_multi_y=force_multi_y) for sys_data in self.sdl]  #((I,ys),(I,ys))
        return [np.concatenate(o,axis=0) for o in  zip(*out)] #(I,I,I),(ys,ys,ys)
    def to_encoder_data(self,na=10,nb=10,nf=5,dilation=1,force_multi_u=False,force_multi_y=False):
        out = [sys_data.to_encoder_data(na=na,nb=nb,nf=nf,dilation=dilation,force_multi_u=force_multi_u,force_multi_y=force_multi_y) for sys_data in self.sdl]  #((I,ys),(I,ys))
        return [np.concatenate(o,axis=0) for o in  zip(*out)] #(I,I,I),(ys,ys,ys)

    def save(self,file):
        '''Saves data'''
        out = [dict(u=sd.u, x=sd.x, y=sd.y, cheat_n=sd.cheat_n, normed=sd.normed) for sd in self.sdl]
        np.savez(file, sdl=out)

    def __repr__(self):
        return f'System_data_list with {len(self.sdl)} series and total length {self.N_samples}, nu={self.nu}, ny={self.ny}, normed={self.normed} lengths={[sd.N_samples for sd in self.sdl]}'

    def plot(self,show=False):
        '''Very simple plotting function'''
        plt.ylabel('y' if self.sdl[0].y is not None else 'u')
        plt.xlabel('t')
        for sd in self.sdl:
            plt.plot(sd.y if sd.y is not None else sd.u)
        if show: plt.show()

    def weighted_mean(self,vals):
        return np.average(vals,axis=0,weights=[sd.N_samples for sd in self.sdl])

    def RMS(self,real, multi_average=True):
        return self.weighted_mean([sd.RMS(sdo,multi_average=multi_average) for sd,sdo in zip(self.sdl,real.sdl)])

    def __sub__(self,other):
        if isinstance(other,System_data_list):            
            return System_data_list([System_data(u=sd.u, x=sd.x, y=sd.y-sdo.y, cheat_n=sd.cheat_n) for sd, sdo in zip(self.sdl,other.sdl)])
        elif isinstance(other,(float,int,np.ndarray,System_data)):
            if isinstance(other, System_data):
                other = other.y
            return System_data_list([System_data(u=sd.u, x=sd.x, y=sd.y-other, cheat_n=sd.cheat_n) for sd in self.sdl])


    def train_test_split(self,split_fraction=0.25):
        '''return 2 data sets of length n*(1-split_fraction) and n*split_fraction respectively (left, right) split'''
        out = list(zip(*[sd.train_test_split(split_fraction=split_fraction) for sd in self.sdl]))
        left,right = System_data_list(out[0]), System_data_list(out[1])
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
        assert isinstance(other,System_data)
        self.sdl.append(other)

    def extend(self,other):
        if isinstance(other,(list,tuple)):
            other = System_data_list(other)
        self.sdl.extend(other.sdl)

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
        return self.u0 is 0 and self.ustd is 1 and self.y0 is 0 and self.ystd is 1

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
                               cheat_n=sys_data.cheat_n,normed=True)

        if isinstance(sys_data,System_data):
            assert sys_data.normed==False, 'System_data is already normalized'
            u_transformed = (sys_data.u-self.u0)/self.ustd if sys_data.u is not None else None
            y_transformed = (sys_data.y-self.y0)/self.ystd if sys_data.y is not None else None
            return System_data(u=u_transformed,x=sys_data.x,y=y_transformed, \
                cheat_n=sys_data.cheat_n,normed=True)
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
                               cheat_n=sys_data.cheat_n,normed=False)

        if isinstance(sys_data,System_data):
            assert sys_data.normed==True, 'System_data is already un-normalized'
            u_inv_transformed = sys_data.u*self.ustd + self.u0 if sys_data.u is not None else None
            y_inv_transformed = sys_data.y*self.ystd + self.y0 if sys_data.y is not None else None
            return System_data(u=u_inv_transformed,x=sys_data.x,y=y_inv_transformed,
                               cheat_n=sys_data.cheat_n,normed=False)
        else:
            raise NotImplementedError(f'type={type(sys_data)} cannot yet be inverse_transform by norm')

    def __repr__(self):
        return f'System_data_norm: (u0={self.u0}, ustd={self.ustd}, y0={self.y0}, ystd={self.ystd}, norm={self.norm})'


if __name__=='__main__':
    # tests

    np.random.seed(42)
    sys_data = System_data(u=np.random.normal(scale=2,size=(100,2)),y=np.random.normal(scale=1.5,size=(100,2)))
    sys_data2 = System_data(u=np.random.normal(size=(100,2)),y=np.random.normal(size=(100,2)))
    sys_data3 = System_data(u=np.random.normal(size=(100,2)),y=np.random.normal(size=(100,2)))

    print(sys_data.NRMS(sys_data2,multi_average=False))
    print([a.shape for a in sys_data.to_encoder_data(5,7,10)])
    print(sys_data2[10:20])

    sdl = System_data_list([sys_data,sys_data2,sys_data3])

    print(sdl.to_encoder_data(9)[0].shape)
    print(len(sdl),sdl.N_samples)
    # sdl.plot(show=True)
    print(sdl.train_test_split())
    print(sdl.down_sample_by_average(10))
    print(sdl.VAF(sdl))
    norm = System_data_norm()
    norm.fit(sdl)
    sdl_transformed = norm.transform(sdl)
    sys_data_transformed = norm.transform(sys_data)
    sys_data12_transformed = norm.transform((sys_data,sys_data2))
    print('transformed:',sdl_transformed,sys_data_transformed,sys_data12_transformed)

    sdl3 = norm.inverse_transform(sdl_transformed)
    print(sdl_transformed.NRMS(sdl))
    print(sdl3.NRMS(sdl))
    print(np.std(sdl_transformed.y,axis=0))
    print('yshape=',sdl_transformed.y.shape)

    print(sdl[1:2,:-10])
    print(len(sdl))

    sys_data = System_data(u=np.random.normal(scale=2,size=(100,2,2)),y=np.random.normal(scale=1.5,size=(100,2,5)))
    sdl = System_data_list([sys_data,sys_data])
    print('sdl',sdl)
    print(sdl.NRMS(sdl))
    sdl_flat = sdl.flatten()
    print(sdl_flat)
    print(sdl_flat.reshape_as(sdl))

