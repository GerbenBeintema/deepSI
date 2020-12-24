
from deepSI.systems.system import System_deriv
from deepSI import datasets
from deepSI.system_data import System_data
import numpy as np

class Bouc_wen(System_deriv):
    def __init__(self,seed=None,dt=1/750,integration_factor=20):
        super(Bouc_wen,self).__init__(dt=dt/integration_factor,nx=3)
        self.mL = 2
        self.cL = 10
        self.kL = 5*10**4
        self.alpha = 5*10**4
        self.beta = 1*10**3
        self.gamma = 0.8
        self.delta = -1.1
        self.integration_factor = integration_factor
        self.nu = 1 #dummy

    def deriv(self,x,u):
        y,yd,z = x
        ydd = (u-(self.kL*y+self.cL*yd)-z)/self.mL
        zd = self.alpha*yd-self.beta*(self.gamma*abs(yd)*z+self.delta*yd*abs(z)) #nu=1
        return yd,ydd,zd

    def h(self, x):
        return x[0]

    def get_train_data(self):
        exp = Experiment(u=self.random.uniform(-1,1,size=10**5))
        return self.apply_experiment(exp)

    def get_test_data(self):
        exp = Experiment(u=self.random.uniform(-1,1,size=10**4))
        return self.apply_experiment(exp)

    def apply_experiment(self,exp):
        exp.u,utijd = (exp.u[:,None]*np.ones((1,self.integration_factor))).flatten(), exp.u
        exp.y,ytijd = None,exp.y
        from scipy import signal

        sys_data = super(Bouc_wen, self).apply_experiment(exp)
        sys_data = sys_data.down_sample_by_average(self.integration_factor)

        b,a = signal.butter(4, 350/sys_data.N_samples,'low',analog=False)
        noise = self.random.normal(loc=0.0, scale=8*10**-6, size=sys_data.N_samples)
        noise = signal.lfilter(b,a,noise)
        sys_data.y += noise
        exp.y,exp.u = ytijd,utijd
        return sys_data

if __name__=='__main__':
    from matplotlib import pyplot as plt
    sys = Bouc_wen(seed=None,integration_factor=20)

    val_multi, val_sinesweep = datasets.Bouc_wen(dir_placement=None, force_download=False, split_data=False).sdl
    val_multi_reproduce = sys.apply_experiment(val_multi)
    # factor = 20
    val_multi.plot()
    val_multi_reproduce.plot(show=True)
    print('difference:',(val_multi_reproduce[200:]).NRMS(val_multi[200:]))
    plt.plot(val_multi_reproduce[200:].y-val_multi[200:].y,'.')
    plt.show()
    #integration_factor=5  -> 0.03504608873410655
    #integration_factor=20 -> 0.0134480539274244
    #integration_factor=20 -> 0.01100388543684641 #noise?


