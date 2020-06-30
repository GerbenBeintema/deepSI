from deepSI.system_data import System_data, System_data_list, System_data_norm
import deepSI
import numpy as np
from matplotlib import pyplot as plt
import pickle

def load_system(file):
    """This is not a safe function, only use on trusted files"""
    return pickle.load( open(file,'rb') )

class System(object):
    def __init__(self,action_space=None,observation_space=None):
        #implement action_space observation space later
        self.norm = System_data_norm()
        self.fitted = False
        self.name = self.__class__.__name__
        self.random = np.random.RandomState(seed=42)


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

        for action in U[k0:]:
            Y.append(obs)
            obs = self.step(action)
        return self.norm.inverse_transform(System_data(u=np.array(U),y=np.array(Y),normed=True,cheat_n=k0))   

    def init_state(self, sys_data):
        '''sys_data is already normed'''
        return self.reset(), 0

    def init_state_multi(self, sys_data, nf=100):
        '''sys_data is already normed'''
        raise NotImplementedError('init_state_multi should be implemented in child')

    def step(self,action):
        '''Applies the action to the systems and returns the new observation'''
        raise NotImplementedError('one_step_ahead should be implemented in child')

    def step_multi(self,actions):
        return self.step(actions)

    def reset():
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

    def n_step_error(self,sys_data,nf=100):
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
        for unow,ynow in zip(np.swapaxes(ufuture,0,1),np.swapaxes(yfuture,0,1)):
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
        return f'System: {self.name}'

    def get_train_data(self):
        exp = System_data(u=self.random.uniform(-2,2,size=10**4))
        return self.apply_experiment(exp)

    def get_test_data(self):
        exp = System_data(u=self.random.uniform(-2,2,size=10**3))
        return self.apply_experiment(exp)

class System_SS(System): #simple state space systems
    def __init__(self,nx,nu=None,ny=None):
        super(System_SS,self).__init__()
        assert nx is not None
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.reset()

    def reset(self):
        self.x = np.zeros((self.nx,))
        return self.h(self.x)

    def reset_multi(self,n):
        self.x = np.zeros((n,self.nx))
        return self.h(self.x)

    def step(self,action):
        self.x = self.f(self.x,action)
        return self.h(self.x)

    def f(self,x,u):
        '''x[k+1] = f(x[k],u[k])'''
        raise NotImplementedError('f and h should be implemented in child')
    def h(self,x):
        '''y[k] = h(x[k])'''
        raise NotImplementedError('f and h should be implemented in child')

class System_Deriv(System_SS):
    def __init__(self,dt=None,nx=None,nu=None,ny=None,):
        assert dt is not None
        assert nx is not None
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
        super(System_IO, self).__init__()
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
        return self.yhist[:,k0-1], k0

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
        yout = self.IO_step(uy) #multi IO?
        self.yhist = np.append(self.yhist[:,1:],yout[:,None],axis=1)
        self.uhist = self.uhist[:,1:]
        return yout

    def IO_step(self,uy):
        raise NotImplementedError('f and h should be implemented in child')



if __name__ == '__main__':
    sys = deepSI.systems.nonlin_Ibased_normals_system()
    exp = System_data(u=np.random.normal(scale=2,size=100))
    print(sys.step(1))
    sys_data = sys.apply_experiment(exp)
    # sys_data.plot(show=True)
    sys = deepSI.systems.sys_ss_test()
    sys_data = sys.apply_experiment(exp)
    sys_data.plot()

    sys.save_system('../../testing/test.p')
    del sys
    sys = load_system('../../testing/test.p')

    sys_data = sys.apply_experiment(exp)
    sys_data.plot(show=True)