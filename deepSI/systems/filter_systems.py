from deepSI.systems.System import System, System_io, System_data
import numpy as np
import deepSI

class Filter_system(System):
    def __init__(self):
        super(Filter_system,self).__init__()

    def reset(self,seed=None):
        pass

    def apply_experiment(self,experiment):
        '''Return the u,(x),y combination 
        todo: move this function to System'''
        u = experiment.u
        # N_transient = experiment.N_transient
        x = None
        from scipy import signal
        y = signal.lfilter(self.b,self.a,u)

        return System_data(u=u,x=x,y=y,system_dict=self.settings,experiment_dict=experiment.settings)

    def get_train_data(self):
        exp = System_data(u=np.random.normal(size=1000))
        return self.apply_experiment(exp)

    def get_test_data(self):
        exp = System_data(u=np.random.normal(size=1000))
        return self.apply_experiment(exp)

class Cheby1(Filter_system):
    """docstring for Cheby1"""
    def __init__(self, order=4, rp=5, Wn=0.1,btype='low', analog=False):
        super(Cheby1, self).__init__()
        from scipy import signal
        self.b, self.a = signal.cheby1(order,rp,Wn,btype,analog=analog) 
        self.a /= 1.05
        self.order = order
        self.rp = rp
        self.Wn = Wn
        self.btype = btype
        self.analog = analog
        self.name = 'cheby1'
        self.settings = {**self.settings,**dict(name=self.name,order=order,rp=rp,Wn=Wn,btype=btype,analog=analog)} #todo seed?

class Butter(Filter_system):
    """todo: fix butter"""
    def __init__(self, order=4, Wn=0.1, btype='low', analog=False):
        super(Butter, self).__init__()
        from scipy import signal
        self.b, self.a = signal.butter(order,Wn,btype,analog=analog)
        self.a /= 1.05
        self.order = order
        self.Wn = Wn
        self.btype = btype
        self.analog = analog
        self.name = 'butter'
        self.settings = {**self.settings,**dict(name=self.name,order=order,Wn=Wn,btype=btype,analog=analog)} #todo seed?

