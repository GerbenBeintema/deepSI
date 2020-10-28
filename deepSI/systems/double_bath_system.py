
from deepSI.systems.System import System_SS, System_data
from deepSI import fit_systems
import numpy as np

class double_bath_system(System_SS):
    def __init__(self,k1=0.5,k2=0.4,k3=0.2,k4=0.3,sigmaw=0.,sigmav=0.,):
        self.sigmav=sigmav
        super(double_bath_system, self).__init__(nx=2)
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

    def h(self,x):
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

if __name__ == '__main__':
    sys = double_bath_system()
    sys_data = sys.get_train_data()

    SYS = fit_systems.System_IO_fit_linear
    score, sys_fit, kwargs, _ = fit_systems.grid_search(SYS, sys_data, dict(na=range(0,3),nb=range(1,20)))
    sys_data_predict = sys_fit.apply_experiment(sys_data)
    sys_data.plot()
    sys_data_predict.plot(show=True)