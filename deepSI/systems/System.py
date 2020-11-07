from deepSI.system_data import System_data, System_data_list, System_data_norm
import deepSI
import numpy as np
from matplotlib import pyplot as plt
import pickle
from secrets import token_urlsafe

def load_system(file):
    """This is not a safe function, only use on trusted files"""
    return pickle.load(open(file,'rb'))

class System(object):
    # action_space, observation_space = None, None #backwards  
    def __init__(self, action_space=None, observation_space=None):
        #implement action_space observation space later
        self.action_space, self.observation_space = action_space, observation_space
        self.norm = System_data_norm()
        self.fitted = False
        self.unique_code = token_urlsafe(4).replace('_','0').replace('-','a') #random code
        self.name = self.__class__.__name__ + '_' + self.unique_code
        self.seed = 42
        self.use_norm = True #can be changed later 

    @property
    def random(self): #gets created ones called, this is to make pickle more stable between different version of numpy
        if not hasattr(self,'_random'):
            self._random = np.random.RandomState(seed=self.seed)
        return self._random
    
    def apply_controller(self,controller,N_samples):
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


    def apply_experiment(self,sys_data): #can put this in apply controller
        if isinstance(sys_data,(tuple,list,System_data_list)):
            return System_data_list([self.apply_experiment(sd) for sd in sys_data])
        Y = []
        sys_data_norm = self.norm.transform(sys_data)
        
        U = sys_data_norm.u
        if sys_data_norm.y is not None: #if y is not None than init state
            obs, k0 = self.init_state(sys_data_norm) #is reset if init_state is not defined #normed obs
            Y.extend(sys_data_norm.y[:k0])
        else:
            obs, k0 = self.reset(), 0

        for k in range(k0,len(U)):
            Y.append(obs)
            if k<len(U)-1: #skip last step
                action = U[k]
                obs = self.step(action)
        return self.norm.inverse_transform(System_data(u=np.array(U),y=np.array(Y),normed=True,cheat_n=k0))   

    def init_state(self,sys_data):
        '''sys_data is already normed'''
        return self.reset(), 0

    def init_state_multi(self, n):
        '''sys_data is already normed'''
        raise NotImplementedError('init_state_multi should be implemented in child')

    def step(self,action):
        '''Applies the action to the systems and returns the new observation'''
        raise NotImplementedError('one_step_ahead should be implemented in child')

    def step_multi(self,actions):
        return self.step(actions)

    def reset(self):
        '''Should reset the internal state and return the current obs'''
        raise NotImplementedError('one_step_ahead should be implemented in child')

    def reset_multi(self,n):
        '''Should reset the internal state and return the current obs'''
        raise NotImplementedError('reset_multi is to be implemented')


    def one_step_ahead(self,sys_data):
        if isinstance(sys_data,(list,tuple,System_data_list)): #requires validation
            return System_data_list([self.apply_experiment(sd) for sd in sys_data])
        sys_data_norm = self.norm.transform(sys_data)
        obs, k0 = self.init_state_multi(sys_data_norm,nf=1)
        Y = np.concatenate([sys_data_norm.y[:k0],obs],axis=0)
        return self.norm.inverse_transform(System_data(u=np.array(sys_data_norm.u),y=np.array(Y),normed=True,cheat_n=k0))   
        # raise NotImplementedError('one_step_ahead is to be implemented')

    def n_step_error(self,sys_data,nf=100,RMS=False):
        # 1. init a multi state
        # do the normal loop, 
        # how to deal with list sys_data?
        # raise NotImplementedError('n_step_error is to be implemented')

        if isinstance(sys_data,(list,tuple)):
            sys_data = System_data_list(sys_data)
            # [self.n_step_error(sd,return_weight=True) for sd in sys_data]
        sys_data = self.norm.transform(sys_data)
        obs, k0 = self.init_state_multi(sys_data, nf=nf)
        _,_,ufuture,yfuture = sys_data.to_hist_future_data(na=k0,nb=k0,nf=nf)

        Losses = []
        for unow, ynow in zip(np.swapaxes(ufuture,0,1), np.swapaxes(yfuture,0,1)):
            if RMS:
                self.norm.ystd
                Losses.append(np.mean((ynow-obs)**2*self.norm.ystd**2)**0.5)
            else:
                Losses.append(np.mean((ynow-obs)**2)**0.5)
            obs = self.step_multi(unow)
        return np.array(Losses)

    def n_step_error_slow(self,sys_data,nf=100):
        '''Slow variant of n_step_error'''
        # do it in a for loop
        # for k in range(len(sys_data)-nf-k0?):
        #     init state
        raise NotImplementedError('one_step_ahead is to be implemented')

    def save_system(self,file):
        pickle.dump(self, open(file,'wb'))

    def __repr__(self):
        simple_action = (self.action_space is None) or (isinstance(self.action_space,gym.spaces.Box) and self.action_space.shape==tuple())
        simple_observation_space = (self.observation_space is None) or (isinstance(self.observation_space,gym.spaces.Box) and self.observation_space.shape==tuple())
        if simple_action and simple_observation_space:
            return f'System: {self.name}'
        else:
            return f'System: {self.name}, action_space={self.action_space}, observation_space={self.observation_space}'

    def get_train_data(self):
        exp = System_data(u=self.random.uniform(-2,2,size=10**4))
        return self.apply_experiment(exp)

    def get_test_data(self):
        exp = System_data(u=self.random.uniform(-2,2,size=10**3))
        return self.apply_experiment(exp)

import gym
from gym.spaces import Box

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

class System_SS(System): #simple state space systems
    def __init__(self,nx,nu=None,ny=None):
        action_shape = tuple() if nu is None else (nu,)
        observation_shape = tuple() if ny is None else (ny,)
        action_space = Box(-float('inf'),float('inf'),shape=action_shape)
        observation_space = Box(-float('inf'),float('inf'),shape=observation_shape)
        super(System_SS,self).__init__(action_space,observation_space)

        assert nx is not None
        self.nx = nx
        self.nu = nu
        self.ny = ny

        self.x = np.zeros((self.nx,))

    def reset(self):
        self.x = np.zeros((self.nx,))
        return self.h(self.x)

    def reset_multi(self,n):
        self.x = np.zeros((n,self.nx))
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

class System_Deriv(System_SS):
    def __init__(self,dt=None,nx=None,nu=None,ny=None):
        assert dt is not None
        self.dt = dt
        super(System_Deriv,self).__init__(nx,nu,ny)
        

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


class System_IO(System):
    def __init__(self,na,nb,nu=None,ny=None): #(u,y)
        action_shape = tuple() if nu is None else (nu,) #repeated code
        observation_shape = tuple() if ny is None else (ny,)
        action_space = Box(-float('inf'), float('inf'), shape=action_shape)
        observation_space = Box(-float('inf'), float('inf'), shape=observation_shape)
        super(System_IO, self).__init__(action_space, observation_space)

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

    def init_state_multi(self,sys_data,nf=100):
        k0 = max(self.na,self.nb)
        self.yhist = np.array([sys_data.y[k0-self.na+i:k0+i] for i in range(0,len(sys_data)-k0-nf+1)]) #+1? #shape = (N,na)
        self.uhist = np.array([sys_data.u[k0-self.nb+i:k0+i-1] for i in range(0,len(sys_data)-k0-nf+1)]) #+1? #shape = 
        return self.yhist[:,-1], k0

    def step(self,action):
        self.uhist.append(action)
        uy = np.concatenate((np.array(self.uhist).flat,np.array(self.yhist).flat),axis=0) #might not be the quickest way
        yout = self.IO_step(uy)
        self.yhist.append(yout)
        self.yhist.pop(0)
        self.uhist.pop(0)
        return yout

    def step_multi(self,actions):
        self.uhist = np.append(self.uhist,actions[:,None],axis=1)
        uy = np.concatenate([self.uhist,self.yhist],axis=1)
        yout = self.multi_IO_step(uy)
        self.yhist = np.append(self.yhist[:,1:],yout[:,None],axis=1)
        self.uhist = self.uhist[:,1:]
        return yout

    def IO_step(self,uy):
        raise NotImplementedError('IO_step should be implemented in child')

    def multi_IO_step(self,uy):
        return self.IO_step(uy)

class System_BJ(System):
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
    # make CallLoss



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

    # sys = deepSI.systems.nonlin_Ibased_normals_system()
    # exp = System_data(u=np.random.normal(scale=2,size=100))
    # print(sys.step(1))
    # sys_data = sys.apply_experiment(exp)
    # # sys_data.plot(show=True)
    # sys = deepSI.systems.sys_ss_test()
    # sys_data = sys.apply_experiment(exp)
    # sys_data.plot()

    # sys.save_system('../../testing/test.p')
    # del sys
    # sys = load_system('../../testing/test.p')

    # sys_data = sys.apply_experiment(exp)
    # sys_data.plot(show=True)