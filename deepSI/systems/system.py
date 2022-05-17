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
        self.init_model_done = False
        self.unique_code = token_urlsafe(4).replace('_','0').replace('-','a') #random code
        self.seed = 42
        self._dt = None
        self.check_valid_system()

    def exists(self, name):
        return getattr(self,name).__func__!=getattr(System, name)

    def check_valid_system(self):
        count = 0
        if self.exists('act_measure'):
            count += 1
            assert self.exists('init_state_and_measure') or self.exists('reset_state_and_measure'), 'act_measure is defined but neither init_state_and_measure or reset_state_and_measure is defined'
        if self.exists('act_measure_multi'):
            count += 1
            assert self.exists('init_state_and_measure_multi'), 'act_measure_multi is defined but init_state_and_measure_multi is not defined'
        if self.exists('measure_act'):
            count += 1
            assert self.exists('init_state') or self.exists('reset_state'), 'measure_act is defined but neither init_state or reset_state is defined'
        if self.exists('measure_act_multi'):
            count += 1
            assert self.exists('init_state_multi'), 'measure_act_multi is defined but init_state_multi is not defined'
        assert count>0, 'no valid method of using the system has been found. One of [act_measure, act_measure_multi, measure_act, measure_act_multi] should be defined'

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
        if   self.measure_act_multi.__func__!=System.measure_act_multi or self.measure_act.__func__!=System.measure_act:
            return False
        elif self.act_measure_multi.__func__!=System.act_measure_multi or self.act_measure.__func__!=System.act_measure:
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
        
    def apply_experiment(self, sys_data, save_state=False, init_state=True):
        if isinstance(sys_data,(tuple,list,System_data_list)):
            #assert dont_set_initial_state is False, 'System_data_list and dont_set_initial_state=True would be errorous'
            return System_data_list([self.apply_experiment(sd, save_state=save_state) for sd in sys_data])
        sys_data_norm = self.norm.transform(sys_data) #do this correctly

        dt_old = self.dt
        if sys_data.dt is not None:
            self.dt = sys_data.dt #calls the dt setter

        U, Y = sys_data_norm.u, []
        if init_state==False:
            k0 = 0
        elif sys_data_norm.y is not None:
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
        if self.exists('measure_act_multi') or self.exists('act_measure_multi'):
            return self.measure_act_multi([action])[0]
        
        if self.exists('act_measure'):
            last_output, self.current_output = self.current_output, self.act_measure(action)
            return last_output
        raise NotImplementedError('measure_act or measure_act_multi or act_measure or act_measure_multi should be implemented in subclass')

    def measure_act_multi(self, actions):
        '''
        calls act_measure if it was overwritten otherwise will throw an error
        '''
        if self.exists('act_measure_multi'):
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
        if self.exists('reset_state_and_measure'):
            self.current_output = self.reset_state_and_measure()
            return
        else:
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
        if self.exists('init_state_multi') or self.exists('init_state_and_measure_multi'):
            nf = len(sys_data) - self.k0
            k0 = self.init_state_multi(sys_data,nf=nf)
            return k0

        #warning for if torch fittable?
        if self.exists('init_state_and_measure'):
            self.current_output, k0 = self.init_state_and_measure(sys_data)
            return k0
        
        #todo; insert warning if it is a system which expects the initial state to be set
        self.reset_state() #this can give errors if multi_step is defined but not 
        return 0
    
    def init_state_multi(self, sys_data, nf=None, stride=1):
        if self.exists('init_state_and_measure_multi'):
            self.current_output, k0 = self.init_state_and_measure_multi(sys_data, nf=nf, stride=stride)
            return k0
        else:
            raise NotImplementedError('init_state_multi should be defined in child if measure_act_multi also exist')

    def init_state_and_measure(self, sys_data):
        #todo; insert warting if it is a system which expects the initial state to be set
        return self.reset_state_and_measure(), 0

    def init_state_and_measure_multi(self, sys_data, nf=None, stride=1):
        raise NotImplementedError('init_state_and_measure_multi called but act_measure_multi is not defined')

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
        k0 = self.init_state_multi(sys_data, nf=nf, stride=1)
        _, _, ufuture, yfuture = sys_data.to_hist_future_data(na=k0, nb=k0, nf=nf, stride=1)

        assert full==False
        for i,unow in enumerate(np.swapaxes(ufuture,0,1)):
            obs = self.measure_act_multi(unow)
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

    def n_step_error(self,sys_data,nf=100,stride=1,mode='NRMS',mean_channels=True):
        '''Calculate the expected error after taking n=1...nf steps.

        Parameters
        ----------
        sys_data : System_data
        nf : int
            upper bound of n.
        stride : int
            passed to init_state_multi to reduce memory cost.
        mode : str or System_data_norm
            'NRMS', 'RMS', 'NRMS_sys_norm' or a System_data_norm for 
            norm obtained from the sys_data, 
            no norm
            norm obtained from the provided data, 
            given norm
        mean_channels : boole
            return (nf) shape if true and (nf,ny) alike if false.
        '''
        norm = System_data_norm()
        if isinstance(mode, tuple):
            norm, error_mode = mode
        elif isinstance(mode,System_data_norm):
            norm, error_mode = mode, 'RMS'
        else:
            #figure out the error mode
            if 'RMS' in mode:
                error_mode = 'RMS'
            elif 'MSE' in mode:
                error_mode = 'MSE'
            elif 'MAE' in mode:
                error_mode = 'MAE'
            else:
                raise NotImplementedError(f'mode {mode} should has one of RMS MSE MAE')
            
            if '_sys_norm' in mode:
                norm = self.norm
            elif mode[0]=='N':
                norm.fit(sys_data)

        if isinstance(sys_data,(list,tuple)):
            sys_data = System_data_list(sys_data)

        dt_old = self.dt
        self.dt = sys_data.dt

        sys_data = self.norm.transform(sys_data)
        k0 = self.init_state_multi(sys_data, nf=nf, stride=stride)
        _, _, ufuture, yfuture = sys_data.to_hist_future_data(na=k0, nb=k0, nf=nf, stride=stride)

        Losses = []
        for unow, ynow in zip(np.swapaxes(ufuture,0,1), np.swapaxes(yfuture,0,1)):
            obs = self.measure_act_multi(unow)
            res = (ynow-obs)*self.norm.ystd/norm.ystd
            if callable(error_mode):
                Losses.append(error_mode(res))
            elif error_mode=='RMS':
                Losses.append(np.mean(res**2,axis=0)**0.5)
            elif error_mode=='MSE':
                Losses.append(np.mean(res**2,axis=0))
            elif error_mode=='MAE':
                Losses.append(np.mean(np.abs(res),axis=0))
            else:
                raise NotImplementedError('error_mode should be one of ["RMS","MSE","MAE"]')

        #Remove large state
        if isinstance(sys_data,System_data_list):
            self.init_state(sys_data[0]) 
        else:
            self.init_state(sys_data)
        
        if dt_old is not None:
            self.dt = dt_old

        return np.array([np.mean(a) for a in Losses]) if mean_channels else np.array(Losses)

    def n_step_error_plot(self, sys_data, nf=100, stride=1, mode='NRMS', mean_channels=True, show=True):
        Losses = self.n_step_error(sys_data, nf=nf, stride=stride, mode=mode, mean_channels=mean_channels)
        if sys_data.dt is not None:
            dt = sys_data.dt
        else:
            dt = self.dt if self.dt is not None else 1
        tar = np.arange(nf)*dt
        plt.plot(tar,Losses)
        plt.xlabel(f'Time (sec) (dt={dt})' if dt!=1 else 'index time')
        plt.ylabel(mode)
        plt.grid()
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

class System_ystd(System):
    # 
    # y and ystd
    # Every time an output is returned a std is also returned
    #
    pass


class System_gym(System):
    """docstring for System_gym"""
    def __init__(self, env, env_kwargs=dict(), n=None):
        if isinstance(env,gym.Env):
            assert n==None, 'if env is already a gym environment than n cannot be given'
            self.env = env

        if n==None:
            self.env = gym.make(env,**env_kwargs)
        else:
            raise NotImplementedError('n requires implementation later')
        super(System_gym, self).__init__(action_space=self.env.action_space, observation_space=self.env.observation_space)

    def reset_state_and_measure(self):
        return self.env.reset()
        
    def act_measure(self,action):
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

    def reset_state(self):
        self.x = np.zeros((self.nx,) if isinstance(self.nx,int) else self.nx)

    def measure_act(self,action):
        y = self.h(self.x, action)
        self.x = self.f(self.x,action)
        return y

    def f(self,x,u):
        '''x[k+1] = f(x[k],u[k])'''
        raise NotImplementedError('f and h should be implemented in child')
    def h(self,x,u): 
        '''y[k] = h(x[k],u[k])'''
        raise NotImplementedError('f and h should be implemented in child')

    def get_state(self):
        return self.x


from scipy.integrate import solve_ivp
class System_deriv(System_ss):
    '''ZOH integration for datageneration'''
    def __init__(self, dt=None, nx=None, nu=None, ny=None, method='RK4'):
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
            return x + (k1+2*k2+2*k3+k4)/6
        else:
            f = lambda t,x: self.deriv(x,u)
            sol = solve_ivp(f, [0, self.dt], x, method=self.method) #integration
            return sol.y[:,-1]

    def deriv(self,x,u):
        raise NotImplementedError('self.deriv should be implemented in child')


class System_io(System):
    def __init__(self,na, nb, nu=None, ny=None, feedthrough=False): #(u,y)
        action_shape = tuple() if nu is None else (nu,) #repeated code
        observation_shape = tuple() if ny is None else (ny,)
        action_space = Box(-float('inf'), float('inf'), shape=action_shape)
        observation_space = Box(-float('inf'), float('inf'), shape=observation_shape)
        super(System_io, self).__init__(action_space, observation_space)

        self.nb = nb #hist length of u
        self.na = na #hist length of y
        self.nu = nu
        self.ny = ny
        self.feedthrough = feedthrough
        #y[k] = step(u[k-nb,k-1],y[k-na,...,k-1])
        #y[k+1] = step(u[k-nb+1,k],y[k-na-1,...,k])
        self.reset_state()

    def reset_state(self):
        self.yhist = [0]*self.na if self.ny is None else [[0]*self.ny for i in range(self.na)]
        self.uhist = [0]*self.nb if self.nu is None else [[0]*self.nu for i in range(self.nb)]
        return 0

    @property
    def k0(self):
        return max(self.na,self.nb)

    def init_state(self, sys_data):
        #sys_data already normed
        k0 = self.k0
        self.yhist = list(sys_data.y[k0-self.na:k0])
        self.uhist = list(sys_data.u[k0-self.nb:k0]) #how it is saved, len(yhist) = na, len(uhist) = nb-1 or nb if feedthrough
        #when taking an action uhist gets appended to create the current state
        return k0

    def init_state_multi(self,sys_data,nf=100,stride=1):
        k0 = self.k0
        self.yhist = np.array([sys_data.y[k0-self.na+i:k0+i] for i in range(0,len(sys_data)-k0-nf+1,stride)]) #+1? #shape = (N,na)
        self.uhist = np.array([sys_data.u[k0-self.nb+i:k0+i] for i in range(0,len(sys_data)-k0-nf+1,stride)]) #+1? #shape = (N,nb-1(+1)) +1 if feedthrough
        return k0

    def measure_act(self, action):
        if self.feedthrough:
            self.uhist.append(action) #add current output only when feedthrough is enabled
        uy = np.concatenate((np.array(self.uhist).flat,np.array(self.yhist).flat),axis=0) #might not be the quickest way
        yout = self.io_step(uy)
        if not self.feedthrough:
            self.uhist.append(action)
        self.yhist.append(yout)
        self.yhist.pop(0)
        self.uhist.pop(0)
        return yout

    def measure_act_multi(self,actions):
        if self.feedthrough:
            self.uhist = np.append(self.uhist,actions[:,None],axis=1)
        uy = np.concatenate([self.uhist.reshape(self.uhist.shape[0],-1),self.yhist.reshape(self.uhist.shape[0],-1)],axis=1) ######todo MIMO
        yout = self.multi_io_step(uy)
        if not self.feedthrough:
            self.uhist = np.append(self.uhist,actions[:,None],axis=1)
        self.yhist = np.append(self.yhist[:,1:],yout[:,None],axis=1)
        self.uhist = self.uhist[:,1:]
        return yout

    def io_step(self,uy):
        raise NotImplementedError('io_step should be implemented in child')

    def multi_io_step(self,uy):
        return self.io_step(uy)

    def get_state(self):
        return [copy.copy(self.uhist), copy.copy(self.yhist)]

if __name__ == '__main__':
    pass
    # sys = Systems_gyms('MountainCarContinuous-v0')
    # pass
    # sys = Systems_gyms('LunarLander-v2')
    # print(sys.reset_state())
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

    #     def h(self,x,u):
    #         return x[0]

    # np.random.seed(32)
    # sys = barrier(method='RK45')
    # exp = deepSI.System_data(u=np.random.uniform(-1,1,size=500))
    # d = sys.apply_experiment(exp)
    # d.plot(show=True)
