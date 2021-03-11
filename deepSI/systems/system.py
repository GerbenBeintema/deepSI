from deepSI.system_data import System_data, System_data_list, System_data_norm
import deepSI
import numpy as np
from matplotlib import pyplot as plt
import pickle
from secrets import token_urlsafe
import copy
import gym
from gym.spaces import Box

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

    @property
    def name(self):
        return self.__class__.__name__ + '_' + self.unique_code
    @property
    def random(self): #gets created ones called, this is to make pickle more stable between different version of numpy
        if not hasattr(self,'_random'):
            self._random = np.random.RandomState(seed=self.seed)
        return self._random
    def get_state(self):
        '''state of the system (not the parameters)

        Returns
        -------
        state : the user defined state
        '''
        import warnings
        warnings.warn('Calling sys.state but no state has been set')
        return None

    def apply_experiment(self, sys_data, save_state=False): #can put this in apply controller
        '''Do a experiment with for given system data (fixed u)

        Parameters
        ----------
        sys_data : System_data or System_data_list (or list or tuple)
            The experiment which should be applied

        Notes
        -----
        This will initialize the state using self.init_state if sys_data.y (and u)
        is not None and skip the appropriate number of steps associated with it.
        If either is missing than self.reset() is used to initialize the state. 
        Afterwards this state is advanced using sys_data.u and the output is saved at each step.
        Lastly, the number of skipped/copied steps in init_state is saved as sys_data.cheat_n such 
        that it can be accounted for later.
        '''

        if isinstance(sys_data,(tuple,list,System_data_list)):
            return System_data_list([self.apply_experiment(sd) for sd in sys_data])
        Y = []
        sys_data_norm = self.norm.transform(sys_data)
        
        U = sys_data_norm.u
        if sys_data_norm.y is not None: #if y is not None than init state
            obs, k0 = self.init_state(sys_data_norm) #is reset if init_state is not defined #normed obs
            Y.extend(sys_data_norm.y[:k0]) #h(x_{k0-1})
        else:
            obs, k0 = self.reset(), 0
        if save_state:
            X = [self.get_state()]*(k0+1)

        for k in range(k0,len(U)):
            Y.append(obs) 
            if k<len(U)-1: #skip last step
                action = U[k]
                obs = self.step(action)
                if save_state:
                    X.append(self.get_state())
        return self.norm.inverse_transform(System_data(u=np.array(U),y=np.array(Y),x=np.array(X) if save_state else None,normed=True,cheat_n=k0))   
    
    def apply_controller(self,controller,N_samples):
        '''Same as self.apply_experiment but with a controller

        Parameters
        ----------
        controller : callable
            when called with the current output it return the next action/input that should be taken

        Notes
        -----
        This method is in a very early state and will probably be changed in the near future.
        '''
        Y = []
        U = []
        obs = self.reset() #normed obs
        for i in range(N_samples):
            Y.append(obs)
            action = (controller(obs*self.norm.ystd +self.norm.y0)-self.norm.u0)/self.norm.ustd #transform y and inverse transform resulting action
            U.append(action)
            obs = self.step(action)
        # Y = Y[:-1]
        return self.norm.inverse_transform(System_data(u=np.array(U),y=np.array(Y),normed=True))

    def init_state(self, sys_data):
        '''Initialize the internal state of the model using the start of sys_data

        Returns
        -------
        Output : an observation (e.g. floats)
            The observation/predicted state at time step k0
        k0 : int
            number of steps that should be skipped

        Notes
        -----
        Example: x0 = encoder(u[t-k0:k0],yhist[t-k0:k0]), and return h(x0), k0
        This function is often overwritten in child. As default it will return self.reset(), 0 
        '''
        return self.reset(), 0

    def init_state_multi(self, sys_data, nf=None, dilation=1):
        '''Similar to init_state but to initialize multiple states 
           (used in self.n_step_error and self.one_step_ahead)

            Parameters
            ----------
            sys_data : System_data
                Data used to initialize the state
            nf : int
                skip the nf last states
            dilation: int
                number of states between each state
           '''
        raise NotImplementedError('init_state_multi should be implemented in subclass')

    def step(self, action):
        '''Applies the action to the system and returns the new observation, 
        should always be overwritten in subclass'''
        raise NotImplementedError('one_step_ahead should be implemented in subclass')

    def step_multi(self,actions):
        '''Applies the actions to the system and returns the new observations'''
        return self.step(actions)

    def reset(self):
        '''Should reset the internal state and return the current obs'''
        raise NotImplementedError('one_step_ahead should be implemented in subclass')

    def one_step_ahead(self, sys_data):
        '''One step ahead prediction'''
        if isinstance(sys_data,(list,tuple,System_data_list)): #requires validation
            return System_data_list([self.apply_experiment(sd) for sd in sys_data])
        sys_data_norm = self.norm.transform(sys_data)
        obs, k0 = self.init_state_multi(sys_data_norm,nf=1)
        Y = np.concatenate([sys_data_norm.y[:k0],obs],axis=0)
        return self.norm.inverse_transform(System_data(u=np.array(sys_data_norm.u),y=np.array(Y),normed=True,cheat_n=k0))   
        # raise NotImplementedError('one_step_ahead is to be implemented')

    def n_step_error(self,sys_data,nf=100,dilation=1,RMS=False):
        '''Calculate the expected error after taking n=1...nf steps.

        Parameters
        ----------
        sys_data : System_data
        nf : int
            upper bound of n.
        dilation : int
            passed to init_state_multi to reduce memory cost.
        RMS : boole
            flag to toggle between NRMS and RMS
         '''
        if isinstance(sys_data,(list,tuple)):
            sys_data = System_data_list(sys_data)
            # [self.n_step_error(sd,return_weight=True) for sd in sys_data]
        sys_data = self.norm.transform(sys_data)
        obs, k0 = self.init_state_multi(sys_data, nf=nf, dilation=dilation)
        _, _, ufuture, yfuture = sys_data.to_hist_future_data(na=k0,nb=k0,nf=nf,dilation=dilation)

        Losses = []
        for unow, ynow in zip(np.swapaxes(ufuture,0,1), np.swapaxes(yfuture,0,1)):
            if RMS: #todo check this
                Losses.append(np.mean((ynow-obs)**2*self.norm.ystd**2)**0.5)
            else:
                Losses.append(np.mean((ynow-obs)**2)**0.5)
            obs = self.step_multi(unow)
        self.init_state(sys_data) #remove large state
        return np.array(Losses)

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
                if np.isscalar(s.low):
                    s.low[1 - np.isfinite(s.low)] = -2
                    s.high[1 - np.isfinite(s.high)] = 2
                else:
                    s.low = np.max([s.low, -2])
                    s.high = np.min([s.high, 2])
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

class System_deriv(System_ss):
    ''''''

    def __init__(self,dt=None,nx=None,nu=None,ny=None):
        assert dt is not None
        self.dt = dt
        super(System_deriv,self).__init__(nx,nu,ny)
        

    def f(self,x,u):
        #https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        #uses self.deriv and self.dt
        #RK4, later some adaptive step and other method
        x = np.array(x)
        k1 = self.dt*np.array(self.deriv(x,u))
        k2 = self.dt*np.array(self.deriv(x+k1/2,u))
        k3 = self.dt*np.array(self.deriv(x+k2/2,u))
        k4 = self.dt*np.array(self.deriv(x+k3,u))
        return x + (k1+2*k2+2*k3+k4)/6

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
        super(System_IO_fit_sklearn, self).__init__(None, None) #action_space=None, observation_space=None
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

    def apply_BJ_experiment(sys_data):
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
    sys = Systems_gyms('LunarLander-v2')
    print(sys.reset())
    # exp = System_data(u=[[int(np.sin(2*np.pi*i/70)>0)*2-1] for i in range(500)]) #mountain car solve
    print(sys)
    exp = System_data(u=[sys.action_space.sample() for i in range(500)]) 
    print(exp.u.dtype)
    sys_data =sys.apply_experiment(exp)
    print(sys_data)
    sys_data.plot(show=True)

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