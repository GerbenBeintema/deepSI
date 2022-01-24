from deepSI.system_data import System_data, System_data_list, System_data_norm
import deepSI
import numpy as np
import pickle
from secrets import token_urlsafe
import copy
import gym
from gym.spaces import Box
from matplotlib import pyplot as plt

def load_system(file):
    """This is not a safe function, only use on trusted files"""
    try:
        return pickle.load(open(file,'rb'))
    except (pickle.UnpicklingError, EOFError): #maybe it was saved using torch systems
        import torch
        return torch.load(file)


class System(object):
    '''The base System class

    Attributes
    ----------
    action_space : gym.space or None
        the input shape of input u. (None is a single unbounded float)
    observation_space : gym.space or None
        The input shape of output y. (None is a single unbounded float)
    norm : instance of System_data_norm
        Used in most fittable systems to normalize the input output.
    fitted : Boole
    unique_code : str
        Some random unique 4 digit code (can be used for saving/loading)
    name : str
        concatenation of the the class name and the unique code
    use_norm : bool
    seed : int
        random seed
    random : np.random.RandomState
        unique random generated initialized with seed (only created ones called)
    '''
    def __init__(self, action_space=None, observation_space=None):
        '''Create a System

        Parameters
        ----------
        action_space : gym.space or None
            the input shape of input u. (None is a single unbounded float)
        observation_space : gym.space or None
            The input shape of output y. (None is a single unbounded float)
        '''
        self.action_space, self.observation_space = action_space, observation_space
        self.norm = System_data_norm()
        self.fitted = False
        self.unique_code = token_urlsafe(4).replace('_','0').replace('-','a') #random code
        self.seed = 42
        self.use_norm = True #can be changed later
        self._dt = None

    @property
    def name(self):
        return self.__class__.__name__ + '_' + self.unique_code
    @property
    def random(self): #gets created ones called, this is to make pickle more stable between different version of numpy
        if not hasattr(self,'_random'):
            self._random = np.random.RandomState(seed=self.seed)
        return self._random

    @property
    def act_measure_sys(self):
        if   self.measure_act_multi!=System.measure_act_multi or self.measure_act!=System.measure_act:
            return False
        elif self.act_measure_multi!=System.act_measure_multi or self.act_measure!=System.act_measure:
            return True
        else:
            raise ValueError('this is neither a act_measure or a measure_act system, both methods undefined')

    @property
    def dt(self):
        return self._dt
    
    @dt.setter
    def dt(self,dt):
        self._dt = dt

    def get_state(self):
        '''state of the system (not the parameters)

        Returns
        -------
        state : the user defined state
        '''
        import warnings
        warnings.warn('Calling self.get_state but no state has been set')
        return None
        
    def apply_experiment(self, sys_data, save_state=False):
        if isinstance(sys_data,(tuple,list,System_data_list)):
            # assert dont_set_initial_state is False, 'System_data_list and dont_set_initial_state=True would be errorous'
            return System_data_list([self.apply_experiment(sd, save_state=save_state) for sd in sys_data])
        sys_data_norm = self.norm.transform(sys_data) #do this correctly

        dt_old = self.dt
        if sys_data.dt is not None:
            self.dt = sys_data.dt #calls the dt setter

        U, Y = sys_data_norm.u, []
        if sys_data_norm.y is not None:
            k0 = self.init_state(sys_data_norm)
            Y.extend(sys_data_norm.y[:k0])
        else:
            k0, _ = 0, self.reset_state()
        
        if save_state:
            X = [self.get_state()]*k0

        for u in U[k0:]:
            if save_state:
                X.append(self.get_state())
            Y.append(self.measure_act(u)) #also advances state
        
        if dt_old is not None:
            self.dt = dt_old
        
        return self.norm.inverse_transform(System_data(u=np.array(U),y=np.array(Y),x=np.array(X) if save_state else None,normed=True,cheat_n=k0,dt=sys_data.dt))  

    def measure_act(self, action):
        '''
        1. Measure giving the current state and action
        2. advance state using action
        
        calls measure_act_multi if (measure_act_multi or act_measure_multi is overwritten) was overwritten

        '''
        if self.measure_act_multi!=System.measure_act_multi or self.act_measure_multi!=System.act_measure_multi: #check if it is overwritten
            return self.measure_act_multi([action])[0]
        
        if self.act_measure!=System.act_measure:
            last_output, self.current_output = self.current_output, self.act_measure_multi(action)
            return last_output
        raise NotImplementedError('measure_act or measure_act_multi or act_measure or act_measure_multi should be implemented in subclass')

    def measure_act_multi(self, actions):
        '''
        calls act_measure if it was overwritten otherwise will throw an error
        '''
        if self.act_measure_multi!=System.act_measure_multi:
            last_output, self.current_output = self.current_output, self.act_measure_multi(actions)
            return last_output
        raise NotImplementedError('measure_act_multi or act_measure_multi should be implemented in subclass')
        
    def act_measure(self, action):
        raise NotImplementedError('act_measure should be implemented in subclass')
    
    def act_measure_multi(self, actions):
        raise NotImplementedError('act_measure_multi should be implemented in subclass')

    def reset_state(self):
        '''Should reset the internal state
        
        if the system is act_measure it will call reset_state_and_measure and save the current output.
        '''
        if self.act_measure_sys:
            self.current_output = self.reset_and_measure()
            return #return None
        raise NotImplementedError('reset should be implemented in subclass')
    
    def reset_state_and_measure(self):
        '''Should reset the internal state and return the current measurement'''
        raise NotImplementedError('reset_state_and_measure should be implemented in subclass if act_measure or act_measure_multi is defined')


    #todo make init state chain.

    def init_state(self, sys_data):
        '''initalizes the interal state using the sys_data and returns the number of steps used in the initilization
        
        Returns
        -------
        k0 : int
            number of steps that have been skipped

        Notes
        -----
        Example: x[k0] = encoder(u[t-k0:k0],yhist[t-k0:k0]), and k0
        This function is often overwritten in child. As default it will call self.reset and return 0
        '''

        #self.k0 #need to be given before the function start
        if (self.init_state_multi!=System.init_state_multi and self.measure_act_multi!=System.measure_act_multi) or \
            (self.init_state_and_measure_multi!=System.init_state_and_measure_multi and self.act_measure_multi!=System.act_measure_multi):
            #hist = torch.tensor(sys_data.to_encoder_data(na=self.na,nb=self.nb,nf=len(sys_data)-max(self.na,self.nb))[0][:1],dtype=torch.float32)
            nf = len(sys_data) - self.k0
            k0 = self.init_state_multi(sys_data,nf=nf)
            return k0
             
            

        #warning for if torch fittable?
        if self.act_measure_sys:
            self.current_output, k0 = self.init_state_and_measure(sys_data)
            return k0
        
        #todo; insert warning if it is a system which expects the initial state to be set
        self.reset_state() #this can give errors if multi_step is defined but not 
        return 0
    
    def init_state_multi(self, sys_data, nf=None, dilation=1):
        #todo change dilation wording
        if self.measure_act_multi!=System.measure_act_multi: 
            raise NotImplementedError('init_state_multi should be defined in child if measure_act_multi also exist')
        elif self.init_state_and_measure_multi!=System.init_state_and_measure_multi and self.act_measure_multi!=System.act_measure_multi:
            self.current_output = self.init_state_and_measure_multi(sys_data, nf=nf, dilation=dilation)
            return self.k0
        else:
            raise ValueError

    def init_state_and_measure_multi(self, sys_data, nf=None, dilation=1):
        if self.act_measure_multi==System.act_measure_multi:
            raise NotImplementedError('init_state_and_measure_multi called but act_measure_multi is not defined')
    
    def init_state_and_measure(self, sys_data):
        #todo; insert warting if it is a system which expects the initial state to be set
        return self.reset_state_and_measure(), 0

    def multi_step_ahead(self, sys_data, nf, full=False):
        '''calculates the n-step precition

        Parameters
        ----------
        sys_data : System_data
        n : int
        full : boole
            if full not only the final n step prediction will be return but also the steps inbetween
        '''
        if isinstance(sys_data,(list,tuple)):
            sys_data = System_data_list(sys_data)
        if isinstance(sys_data, System_data_list):
            if not full:
                return System_data_list([self.multi_step_ahead(sd, nf, full=False) for sd in sys_data.sdl])
            else:
                #self.multi_step_ahead(sd, nf, full=False) returns a list [sd step 1, sd step 2, ...]
                return [System_data_list(o) for o in zip(*[self.multi_step_ahead(sd, nf, full=True) for sd in sys_data.sdl])]

        sys_data = self.norm.transform(sys_data)
        obs, k0 = self.init_state_multi(sys_data, nf=nf, dilation=1)
        _, _, ufuture, yfuture = sys_data.to_hist_future_data(na=k0, nb=k0, nf=nf, dilation=1)

        assert full==False
        for i,unow in enumerate(np.swapaxes(ufuture,0,1)[:-1]):
            obs = self.step_multi(unow)
        obs = np.concatenate([sys_data.y[:k0+nf-1], obs],axis=0)

        sys_data_nstep = System_data(u=sys_data.u, y=obs, normed=True, cheat_n=k0+nf)

        if isinstance(sys_data,System_data_list):
            self.init_state(sys_data[0]) #removes large state
        else:
            self.init_state(sys_data) #removes large state

        return self.norm.inverse_transform(sys_data_nstep)



    def one_step_ahead(self, sys_data):
        '''One step ahead prediction'''
        if isinstance(sys_data,(list,tuple,System_data_list)): #requires validation
            return System_data_list([self.apply_experiment(sd) for sd in sys_data])
        sys_data_norm = self.norm.transform(sys_data)
        dt_old = self.dt
        self.dt = sys_data.dt

        obs, k0 = self.init_state_multi(sys_data_norm,nf=1)
        Y = np.concatenate([sys_data_norm.y[:k0],obs],axis=0)
        self.dt = dt_old
        return self.norm.inverse_transform(System_data(u=np.array(sys_data_norm.u),y=np.array(Y),normed=True,cheat_n=k0))   
        # raise NotImplementedError('one_step_ahead is to be implemented')

    def n_step_error(self,sys_data,nf=100,dilation=1,mode='NRMS',mean_channels=True):
        '''Calculate the expected error after taking n=1...nf steps.

        Parameters
        ----------
        sys_data : System_data
        nf : int
            upper bound of n.
        dilation : int
            passed to init_state_multi to reduce memory cost.
        mode : str or System_data_norm
            'NRMS', 'RMS', 'RMS_sys_norm' or a System_data_norm for 
            norm obtained from the sys_data, 
            no norm
            norm obtained from the provided data, 
            given norm
        mean_channels : boole
            return (nf) shape if true and (nf,ny) alike if false.
        '''
        norm = System_data_norm()
        if mode=='NRMS':
            norm.fit(sys_data)
        elif mode=='RMS':
            pass
        elif mode=='RMS_sys_norm':
            norm = self.norm
        elif isinstance(mode,System_data_norm):
            norm = mode
        else:
            raise ValueError("The mode of the n-step error should be one of ('NRMS', 'RMS', 'RMS_sys_norm', instance of System_data_norm)")

        Losses = self.n_step_error_per_channel(sys_data, nf=nf, dilation=dilation)/norm.ystd
        if mean_channels==False:
            return Losses
        else:
            return np.array([np.mean(a) for a in Losses])


    def n_step_error_per_channel(self, sys_data, nf=100, dilation=1):
        '''Calculate the expected error after taking n=1...nf steps, returns the shape (nf, ny) or nf if ny is None

        Parameters
        ----------
        sys_data : System_data
        nf : int
            upper bound of n.
        dilation : int
            passed to init_state_multi to reduce memory costs.
        '''
        if isinstance(sys_data,(list,tuple)):
            sys_data = System_data_list(sys_data)
            # [self.n_step_error(sd,return_weight=True) for sd in sys_data]

        dt_old = self.dt
        self.dt = sys_data.dt

        sys_data = self.norm.transform(sys_data)
        obs, k0 = self.init_state_multi(sys_data, nf=nf, dilation=dilation)
        _, _, ufuture, yfuture = sys_data.to_hist_future_data(na=k0, nb=k0, nf=nf, dilation=dilation)

        Losses = []
        for unow, ynow in zip(np.swapaxes(ufuture,0,1), np.swapaxes(yfuture,0,1)):
            Losses.append(np.mean((ynow-obs)**2,axis=0)**0.5)
            obs = self.step_multi(unow)

        if isinstance(sys_data,System_data_list):
            self.init_state(sys_data[0]) #removes large state
        else:
            self.init_state(sys_data) #removes large state
        return np.array(Losses)*self.norm.ystd #Units of RMS (nf, ny)
        self.dt = dt_old
        
        return np.array(Losses)
    def n_step_error_plot(self, sys_data, nf=100, dilation=1, RMS=False, show=True):
        Losses = self.n_step_error(sys_data,nf=nf,dilation=dilation,RMS=RMS)
        if sys_data.dt is not None:
            dt = sys_data.dt
        else:
            dt = self.dt if self.dt is not None else 1
        tar = np.arange(nf)*dt
        plt.plot(tar,Losses)
        plt.xlabel('time-error' if dt!=1 else 'n-step-error')
        plt.ylabel('NRMS' if RMS==False else 'RMS')
        if show:
            plt.show()


    def save_system(self,file):
        '''Save the system using pickle

        Notes
        -----
        This can be quite unstable for long term storage or switching between versions of this and other modules.
        Consider manually creating a save_system function for a long term solution.
        '''
        pickle.dump(self, open(file,'wb'))

    def __repr__(self):
        simple_action = (self.action_space is None) or (isinstance(self.action_space,gym.spaces.Box) and self.action_space.shape==tuple())
        simple_observation_space = (self.observation_space is None) or (isinstance(self.observation_space,gym.spaces.Box) and self.observation_space.shape==tuple())
        if simple_action and simple_observation_space:
            return f'System: {self.name}'
        else:
            return f'System: {self.name}, action_space={self.action_space}, observation_space={self.observation_space}'

    def sample_system(self,N_sampes=10**4):
        '''Mostly used for testing purposes it, will apply random actions on the system'''
        if self.action_space is None:
            exp = System_data(u=self.random.uniform(-2,2,size=N_sampes))
        else:
            s = copy.deepcopy(self.action_space)
            if isinstance(s,gym.spaces.Box):
                if np.isscalar(s.low) or (isinstance(s.low,np.ndarray) and s.low.ndim==0): #check if it is a 
                    if not np.isfinite(s.low):
                        s.low = np.array(-2.)
                    if not np.isfinite(s.high):
                        s.high = np.array(2.)
                else: #is a vector
                    s.low[1 - np.isfinite(s.low)] = -2
                    s.high[1 - np.isfinite(s.high)] = 2
            exp = System_data(u=[s.sample() for _ in range(N_sampes)])
        return self.apply_experiment(exp)

class Systems_gyms(System):
    """docstring for Systems_gyms"""
    def __init__(self, env, env_kwargs=dict(), n=None):
        if isinstance(env,gym.Env):
            assert n==None, 'if env is already a gym environment than n cannot be given'
            self.env = env

        if n==None:
            self.env = gym.make(env,**env_kwargs)
        else:
            raise NotImplementedError('n requires implementation later')
        super(Systems_gyms, self).__init__(action_space=self.env.action_space, observation_space=self.env.observation_space)

    def reset(self):
        return self.env.reset()
        
    def step(self,action):
        '''Applies the action to the systems and returns the new observation'''
        obs, reward, done, info = self.env.step(action)
        self.done = done
        return obs

class System_ss(System): #simple state space systems
    '''Derived state-space system with continues u, x, y vectors or scalars'''
    def __init__(self, nx, nu=None, ny=None):
        action_shape = tuple() if nu is None else (nu,)
        observation_shape = tuple() if ny is None else (ny,)
        action_space = Box(-float('inf'),float('inf'),shape=action_shape)
        observation_space = Box(-float('inf'),float('inf'),shape=observation_shape)
        super(System_ss,self).__init__(action_space, observation_space)

        assert nx is not None
        self.nx = nx
        self.nu = nu
        self.ny = ny

        self.x = np.zeros((self.nx,) if isinstance(self.nx,int) else self.nx)

    def reset(self):
        self.x = np.zeros((self.nx,) if isinstance(self.nx,int) else self.nx)
        return self.h(self.x)

    def step(self,action):
        self.x = self.f(self.x,action)
        return self.h(self.x)
    # def step_multi(self,actions)

    def f(self,x,u):
        '''x[k+1] = f(x[k],u[k])'''
        raise NotImplementedError('f and h should be implemented in child')
    def h(self,x):
        '''y[k] = h(x[k])'''
        raise NotImplementedError('f and h should be implemented in child')

    def get_state(self):
        return self.x


from scipy.integrate import solve_ivp
class System_deriv(System_ss):
    ''''''

    def __init__(self,dt=None,nx=None,nu=None,ny=None,method='RK4'):
        super(System_deriv, self).__init__(nx, nu, ny)
        self.dt = dt
        self.method = method

    def f(self,x,u):
        assert self.dt is not None, 'please set dt or in the __init__ or in sys_data.dt'
        #https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        #uses self.deriv and self.dt
        #RK4
        if self.method=='RK4': #this is a lot faster and quite accurate if dt is smaller than the shortest characteristic time-scale.
            x = np.array(x)
            k1 = self.dt*np.array(self.deriv(x,u))
            k2 = self.dt*np.array(self.deriv(x+k1/2,u))
            k3 = self.dt*np.array(self.deriv(x+k2/2,u))
            k4 = self.dt*np.array(self.deriv(x+k3,u))
            xnew = x + (k1+2*k2+2*k3+k4)/6
            return x + (k1+2*k2+2*k3+k4)/6
        else:
            f = lambda t,x: self.deriv(x,u)
            sol = solve_ivp(f, [0, self.dt], x, method=self.method) #integration
            return sol.y[:,-1]

    def deriv(self,x,u):
        raise NotImplementedError('self.deriv should be implemented in child')


class System_io(System):
    def __init__(self,na, nb, nu=None, ny=None): #(u,y)
        action_shape = tuple() if nu is None else (nu,) #repeated code
        observation_shape = tuple() if ny is None else (ny,)
        action_space = Box(-float('inf'), float('inf'), shape=action_shape)
        observation_space = Box(-float('inf'), float('inf'), shape=observation_shape)
        super(System_io, self).__init__(action_space, observation_space)

        self.nb = nb #hist length of u
        self.na = na #hist length of y
        self.nu = nu
        self.ny = ny
        #y[k] = step(u[k-nb,k-1],y[k-na,...,k-1])
        #y[k+1] = step(u[k-nb+1,k],y[k-na-1,...,k])
        self.reset()

    def reset(self):
        self.yhist = [0]*self.na if self.ny is None else [[0]*self.ny for i in range(self.na)]
        self.uhist = [0]*(self.nb-1) if self.nu is None else [[0]*self.nu for i in range(self.nb-1)]
        return 0

    @property
    def k0(self):
        return max(self.na,self.nb)

    def init_state(self,sys_data):
        #sys_data already normed
        k0 = max(self.na,self.nb)
        self.yhist = list(sys_data.y[k0-self.na:k0])
        self.uhist = list(sys_data.u[k0-self.nb:k0-1]) #how it is saved, len(yhist) = na, len(uhist) = nb-1
        #when taking an action uhist gets appended to create the current state
        return sys_data.y[k0-1], k0

    def init_state_multi(self,sys_data,nf=100,dilation=1):
        k0 = max(self.na,self.nb)
        self.yhist = np.array([sys_data.y[k0-self.na+i:k0+i] for i in range(0,len(sys_data)-k0-nf+1,dilation)]) #+1? #shape = (N,na)
        self.uhist = np.array([sys_data.u[k0-self.nb+i:k0+i-1] for i in range(0,len(sys_data)-k0-nf+1,dilation)]) #+1? #shape = 
        return self.yhist[:,-1], k0

    def step(self,action):
        self.uhist.append(action)
        uy = np.concatenate((np.array(self.uhist).flat,np.array(self.yhist).flat),axis=0) #might not be the quickest way
        yout = self.io_step(uy)
        self.yhist.append(yout)
        self.yhist.pop(0)
        self.uhist.pop(0)
        return yout

    def step_multi(self,actions):
        self.uhist = np.append(self.uhist,actions[:,None],axis=1)
        uy = np.concatenate([self.uhist.reshape(self.uhist.shape[0],-1),self.yhist.reshape(self.uhist.shape[0],-1)],axis=1) ######todo MIMO
        yout = self.multi_io_step(uy)
        self.yhist = np.append(self.yhist[:,1:],yout[:,None],axis=1)
        self.uhist = self.uhist[:,1:]
        return yout

    def io_step(self,uy):
        raise NotImplementedError('io_step should be implemented in child')

    def multi_io_step(self,uy):
        return self.io_step(uy)

    def get_state(self):
        return [copy.copy(self.uhist), copy.copy(self.yhist)]

class System_bj(System):
    #work in progress, use at own risk

    #yhat_{t} = f(u_{t-nb:t-1},yhat_{t-na:t-1},y_{t-nc:t-1})
    def __init__(self,na,nb,nc):
        #na = length of y hat
        #nb = length of u
        #nc = length of y real
        super(System_bj, self).__init__(None, None) #action_space=None, observation_space=None
        self.na = na
        self.nb = nb
        self.nc = nc

    @property
    def k0(self):
        return max(self.na,self.nb,self.nc)

    def reset(self):
        self.yhisthat = [0]*self.na if self.ny is None else [[0]*self.ny for i in range(self.na)]
        self.uhist = [0]*(self.nb-1) if self.nu is None else [[0]*self.nu for i in range(self.nb-1)]
        self.yhistreal = [0]*self.nb if self.ny is None else [[0]*self.ny for i in range(self.nb)]
        return 0

    def init_state(self,sys_data):
        #sys_data already normed
        k0 = max(self.na,self.nb,self.nc)
        self.yhisthat = list(sys_data.y[k0-self.na:k0])
        self.yhistreal = list(sys_data.y[k0-self.nc:k0-1])
        self.uhist = list(sys_data.u[k0-self.nb:k0-1]) #how it is saved, len(yhist) = na, len(uhist) = nb-1
        #when taking an action uhist gets appended to create the current state
        return self.yhistreal[-1], k0

    def init_state_multi(self,sys_data,nf=100):
        k0 = max(self.na,self.nb,self.nc)
        self.yhisthat = np.array([sys_data.y[k0-self.na+i:k0+i] for i in range(0,len(sys_data)-k0-nf+1)]) #+1? #shape = (N,na)
        self.yhistreal = np.array([sys_data.y[k0-self.nc+i:k0+i-1] for i in range(0,len(sys_data)-k0-nf+1)]) #+1? #shape = (N,nc)
        self.uhist = np.array([sys_data.u[k0-self.nb+i:k0+i-1] for i in range(0,len(sys_data)-k0-nf+1)]) #+1? #shape = 
        return self.yhisthat[:,-1], k0

    def step(self,action): #normal step
        self.uhist.append(action)
        self.yhistreal.append(self.yhisthat[-1])

        uy = np.concatenate((np.array(self.uhist).flat,np.array(self.yhisthat).flat,np.array(self.yhistreal).flat),axis=0) #might not be the quickest way
        yout = self.BJ_step(uy)
        self.yhistreal.pop(0)
        self.yhisthat.pop(0)
        self.uhist.pop(0)

        self.yhisthat.append(yout)
        return yout

    def step_BJ(self,action,y): #normal step
        #y = the last output
        self.uhist.append(action) #append to [u[t-nb],...,u[t-1]]
        self.yhistreal.append(y) #append to [y[t-nc],...,y[t-1]]

        uy = np.concatenate((np.array(self.uhist).flat,np.array(self.yhisthat).flat,np.array(self.yhistreal).flat),axis=0) #might not be the quickest way
        yout = self.BJ_step(uy)
        self.yhisthat.append(yout)
        self.yhisthat.pop(0)
        self.yhistreal.pop(0) #[y[t-nc-1],...,y[t-1]]
        self.uhist.pop(0)
        return yout

    def step_multi(self,actions): #finish this function
        self.uhist = np.append(self.uhist,actions[:,None],axis=1)
        self.yhistreal = np.append(self.yhistreal,self.yhisthat[:,None],axis=1) #(N,nc)
        uy = np.concatenate([self.uhist,self.yhisthat,self.yhistreal],axis=1)
        yout = self.multi_BJ_step(uy)
        self.yhisthat = np.append(self.yhisthat[:,1:],yout[:,None],axis=1)
        self.uhist = self.uhist[:,1:]
        self.yhistreal = self.yhistreal[:,1:]
        return yout

    def step_BJ_multi(self,actions,ys): #normal step
        self.uhist = np.append(self.uhist,actions[:,None],axis=1)
        self.yhistreal = np.append(self.yhistreal,ys[:,None],axis=1) #(N,nc)
        uy = np.concatenate([self.uhist,self.yhisthat,self.yhistreal],axis=1)
        yout = self.multi_BJ_step(uy)
        self.yhisthat = np.append(self.yhisthat[:,1:],yout[:,None],axis=1)
        self.uhist = self.uhist[:,1:]
        self.yhistreal = self.yhistreal[:,1:]
        return yout

    def multi_BJ_step(self,uy):
        return self.BJ_step(uy)

    def apply_BJ_experiment(self,sys_data):
        if isinstance(sys_data,(tuple,list,System_data_list)):
            return System_data_list([self.apply_BJ_experiment(sd) for sd in sys_data])
        if sys_data.y==None:
            return self.apply_experiment(sys_data) #bail if y does not exist

        Yhat = []
        sys_data_norm = self.norm.transform(sys_data)
        U,Yreal = sys_data_norm.u,sys_data_norm.y
        obs, k0 = self.init_state(sys_data_norm) #is reset if init_state is not defined #normed obs
        Yhat.extend(sys_data_norm.y[:k0])

        for k in range(k0,len(U)):
            Yhat.append(obs)
            if k<len(U)-1: #skip last step
                obs = self.step_BJ(U[k],Yreal[k])
        return self.norm.inverse_transform(System_data(u=np.array(U),y=np.array(Yhat),normed=True,cheat_n=k0))
    #continue here
    # make apply_BJ_experiment
    # make make_fit_data or something
    # make loss



if __name__ == '__main__':
    # sys = Systems_gyms('MountainCarContinuous-v0')
    pass
    # sys = Systems_gyms('LunarLander-v2')
    # print(sys.reset())
    # # exp = System_data(u=[[int(np.sin(2*np.pi*i/70)>0)*2-1] for i in range(500)]) #mountain car solve
    # print(sys)
    # exp = System_data(u=[sys.action_space.sample() for i in range(500)]) 
    # print(exp.u.dtype)
    # sys_data =sys.apply_experiment(exp)
    # print(sys_data)
    # sys_data.plot(show=True)

    # sys = deepSI.systems.Nonlin_io_normals()
    # exp = System_data(u=np.random.normal(scale=2,size=100))
    # print(sys.step(1))
    # sys_data = sys.apply_experiment(exp)
    # # sys_data.plot(show=True)
    # sys = deepSI.systems.SS_test()
    # sys_data = sys.apply_experiment(exp)
    # sys_data.plot()

    # sys.save_system('../../testing/test.p')
    # del sys
    # sys = load_system('../../testing/test.p')

    # sys_data = sys.apply_experiment(exp)
    # sys_data.plot(show=True)

    #deriv testing
    # class barrier(System_deriv):
    #     """docstring for barrier"""
    #     def __init__(self, method='RK4'):
    #         super(barrier, self).__init__(nx=2,dt=0.1,method=method)
        
    #     def deriv(self,x,u):
    #         x,vx = x
    #         dxdt = vx
    #         alpha = 0.01
    #         dvxdt = - 1e-3*vx + alpha*( - 1/(x-1)**2 + 1/(x+1)**2) + u
    #         return [dxdt,dvxdt]

    #     def h(self,x):
    #         return x[0]

    # np.random.seed(32)
    # sys = barrier(method='RK45')
    # exp = deepSI.System_data(u=np.random.uniform(-1,1,size=500))
    # d = sys.apply_experiment(exp)
    # d.plot(show=True)
