
from deepSI.systems.system import System_ss, System_data, System_deriv
from deepSI import fit_systems
import numpy as np

class Double_bath(System_ss):
    def __init__(self,k1=0.5,k2=0.4,k3=0.2,k4=0.3,sigmaw=0.,sigmav=0.,):
        self.sigmav=sigmav
        super(Double_bath, self).__init__(nx=2)
        self.k1=k1
        self.k2=k2
        self.k3=k3
        self.k4=k4
        self.sigmaw=sigmaw
        

    def f(self,x,u):
        x1,x2 = x
        wk = self.random.normal(scale=self.sigmaw)
        x1new = np.clip(x1 - self.k1*np.sqrt(x1) + self.k2*(u + wk),0,np.inf)
        x2new = np.clip(x2 + self.k3*np.sqrt(x1) - self.k4*np.sqrt(x2),0,np.inf)
        # vk = np.random.normal(scale=sigmav,size=(N,))
        return [x1new,x2new]

    def h(self,x,u):
        vk = self.random.normal(scale=self.sigmav)
        return x[0]+vk

    def get_train_data(self,seed=None):
        u = []
        for i in range(1000):
            if i%7==0:
                u.append(self.random.normal(scale=1)+1)
            else:
                u.append(u[-1])
        exp = System_data(u=u)
        return self.apply_experiment(exp)

    def get_test_data(self,seed=None):
        u = []
        for i in range(1000):
            if i%7==0:
                u.append(self.random.normal(scale=1)+1)
            else:
                u.append(u[-1])
        exp = System_data(u=u)
        return self.apply_experiment(exp)


class Cascaded_tanks_continuous(System_deriv):
    def __init__(self,k1=0.5,k2=0.4,k3=0.2,k4=0.3,sigmav=0.,dt=2, x1overflow=8, x2overflow = 25):
        super(Cascaded_tanks_continuous, self).__init__(nx=2, dt=dt)
        self.sigmav=sigmav
        self.k1=k1
        self.k2=k2
        self.k3=k3
        self.k4=k4
        self.x1overflow = x1overflow
        self.x2overflow = x2overflow     

    def f(self,x,u):
        x = super(Cascaded_tanks_continuous, self).f(x,u)
        x1, x2 = x
        x1 = np.clip(x1,0,self.x1overflow)
        x2 = np.clip(x2,0,self.x2overflow)
        return [x1,x2]


    def h(self,x,u):
        vk = self.random.normal(scale=self.sigmav)
        return x[1]+vk

    def deriv(self,x,u):
        x1, x2 = x
        x1 = max(x1,0)
        x2 = max(x2,0)
        k1, k2, k3, k4 = self.k1, self.k2, self.k3, self.k4

        dx1 = -k1*np.sqrt(x1) + k4*max(u,0)
        dx2 = k2*np.sqrt(x1) - k3*np.sqrt(x2)

        if x1==0 and dx1<0:
            dx1 = 0
        elif x1>self.x1overflow and dx1>0:
            dx1 = 0

        if x2==0 and dx2<0:
            dx2 = 0
        elif x2>self.x2overflow and dx2>0:
            dx2 = 0

        return [dx1, dx2]



if __name__ == '__main__':
    # sys = Double_bath()
    # sys_data = sys.get_train_data()

    # SYS = fit_systems.System_IO_fit_linear
    # score, sys_fit, kwargs, _ = fit_systems.grid_search(SYS, sys_data, dict(na=range(0,3),nb=range(1,20)))
    # sys_data_predict = sys_fit.apply_experiment(sys_data)
    # sys_data.plot()
    # sys_data_predict.plot(show=True)
    sys = Cascaded_tanks_continuous(dt=2)
    data = System_data(u=np.ones(shape=(300))*5)
    data = sys.apply_experiment(data,save_state=True)
    data.plot(show=True)
    from matplotlib import pyplot as plt

    plt.plot(data.x)
    plt.show()
    print(np.max(data.x,axis=0))

    #longest timescale = 25
    #max x1 = 9. and x2 = 35
    #overflow at 20 x2 and 
    #u max = 8
    #dt = 2
