
from deepSI.system_data import System_data, System_data_list, System_data_norm
from deepSI.systems.System import System, Systems_gyms
from deepSI.fit_systems.IO_systems import System_IO_pytorch
from deepSI.fit_systems.encoder_systems import System_encoder
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

#  SISO experiment design

class experiment_generators(object):
    """docstring for experiment_generators"""
    def __init__(self, sys):
        super(experiment_generators, self).__init__()
        self.sys = sys
        self.sys_data = []
    
    def random_init(self,n_steps=100):
        sys.reset()
        out = sys.apply_experiment(System_data([sys.action_space.sample() for i in range(n_steps)]))
        self.sys_data = System_data_list([out])


class Exp_design_var_addive(experiment_generators):
    """docstring for Exp_design_var_addive"""
    def __init__(self, sys, model, n_models):
        super(Exp_design_var_addive, self).__init__(sys)
        self.models = [model() for i in range(n_models)] #new class?

    def run(self):
        self.random_init(n_steps=200)
        for model in self.models:
            model.fit(self.sys_data,sim_val=self.sys_data,epochs=30)
        
        for k in tqdm(range(100)):
            self.add_points()
            for model in self.models:
                model.fit(self.sys_data,sim_val=self.sys_data,epochs=30)


    def add_points(self,n=100):
        for i in range(n):
            upotentials = [sys.action_space.sample() for i in range(20)]
            sys_data0 = self.sys_data[0][-self.models[0].k0-3:]
            y0, ystd = np.mean(self.sys_data.y, axis=0), np.std(self.sys_data.y, axis=0)
            sys_data0.y = np.append(sys_data0.y, sys_data0.y[-1:], axis=0)
            sys_data0.u = np.append(sys_data0.u, sys_data0.u[-1:], axis=0)
            variances = []
            for u in upotentials:
                sys_data0.u[-2] = u
                outs = [model.apply_experiment(sys_data0) for model in self.models]
                model_predics = np.array([o.y[-1] for o in outs])
                model_predics_normed = (model_predics - y0)/ystd
                model_variance = np.mean(np.std(model_predics_normed,axis=0))
                variances.append(model_variance)
                # sys_data0.u = sys_data0.u[:-1]
            sys_data0.y = sys_data0.y[:-1]

            u_best = upotentials[np.argmax(model_variance)]
            y_new = self.sys.step(u_best)
            self.sys_data[0].y = np.append(self.sys_data[0].y,[y_new],axis=0)
            self.sys_data[0].u = np.append(self.sys_data[0].u,[u_best],axis=0)
            self.sys_data[0].u[-2] = u_best #last u is a dummy


        # model.step(self.sys_data) #apply experiment extract?
        # model.predict(self.sys_data) #how to initialize the state?


if __name__ == '__main__':
    # x = np.array([1,2,3])
    # print(np.append(x,1))
    # print(x)
    sys = Systems_gyms('MountainCarContinuous-v0')
    exp_gen = Exp_design_var_addive(sys, System_encoder, 3)
    exp_gen.run()

