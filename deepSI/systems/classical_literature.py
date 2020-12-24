
from deepSI.systems.system import System_data, System, System_io, System_ss
import numpy as np
import deepSI

class Nonlin_io_normals(System_io):
    def __init__(self):
        super(Nonlin_io_normals, self).__init__(na=2,nb=2)

    def io_step(self,uy):
        ukm2, ukm1, ykm2, ykm1 = uy
        ystar = (0.8 - 0.5 * np.exp(-ykm1 ** 2)) * ykm1 - (0.3 + 0.9 * np.exp(-ykm1 ** 2)) * ykm2 \
               + ukm1 + 0.2 * ukm2 + 0.1 * ukm1 * ukm2
        return ystar

class Hammerstein_sysid_book(System_io):
    def __init__(self):
        super(Hammerstein_sysid_book, self).__init__(na=2,nb=2)

    def io_step(self,uy):
        ukm2, ukm1, ykm2, ykm1 = uy
        ystar = 0.01867*np.arctan(ukm1)+0.01746*np.arctan(ukm2) \
                +1.7826*ykm1 - 0.8187*ykm2
        return ystar

class Wiener_sysid_book(System_io):
    def __init__(self):
        super(Wiener_sysid_book, self).__init__(na=2,nb=2)

    def io_step(self,uy):
        ukm2, ukm1, ykm2, ykm1 = uy
        ystar = np.arctan(0.01867*ukm1+0.1746*ukm2) \
                + 1.7826*ykm1 - 0.8187*ykm2 
        return ystar

class Wiener2_sysid_book(System_io):
    def __init__(self):
        super(Wiener2_sysid_book, self).__init__(na=2,nb=2)

    def io_step(self,uy):
        ukm2, ukm1, ykm2, ykm1 = uy
        ystar = np.arctan(1.867*ukm1+1.746*ukm2) \
                + 1.7826*ykm1 - 0.8187*ykm2 
        return ystar

class NDE_squared_sysid_book(System_io):
    def __init__(self):
        super(NDE_squared_sysid_book, self).__init__(na=2,nb=2)

    def io_step(self,uy):
        ukm2, ukm1, ykm2, ykm1 = uy
        ystar = -0.07289*(ukm1-0.2*ykm1**2) \
                +0.09394*(ukm2-0.2*ukm2**2) \
                +1.68364*ykm1 - 0.70468*ykm2
        return ystar

class Dynamic_nonlin_sysid_book(System_io):
    def __init__(self):
        super(Dynamic_nonlin_sysid_book, self).__init__(na=2,nb=3)

    def io_step(self,uy):
        ukm3, ukm2, ukm1, ykm2, ykm1 = uy
        ystar = 0.133*ukm2-.0667*ukm3 + 1.5*ykm1 \
                - 0.7*ykm2 + ukm1*(0.1*ykm1 - 0.2*ykm2)
        return ystar

class Nonlin_io_example_2(System_io):
    def __init__(self):
        super(Nonlin_io_example_2, self).__init__(na=2,nb=1)

    def io_step(self,uy):
        ukm1, ykm2, ykm1 = uy
        ystar = ykm1*ykm2*(ykm1+2.5)/(1+ykm1**2*ykm2**2)+ukm1
        return ystar



if __name__=='__main__':
    from matplotlib import pyplot as plt
    id = 1
    def plot_and_fit(sys):
        global id
        exp = System_data(u=np.random.normal(scale=0.1,size=1000))
        sys = sys()
        sys_data = sys.apply_experiment(exp)

        SYS = deepSI.fit_systems.System_IO_fit_linear
        sys_fit, score, kwargs, _ = deepSI.fit_systems.grid_search(SYS, sys_data, dict(na=range(0,10),nb=range(1,10)))
        na,nb = kwargs['na'], kwargs['nb']

        sys_data_predict = sys_fit.apply_experiment(sys_data)
        plt.subplot(2,3,id)
        sys_data.plot()
        sys_data_predict.plot()
        plt.title(f'{sys.name} BFR {sys_data_predict.NRMS(sys_data):.2%}')
        plt.legend(['real',f'predict na={na},nb={nb}'])
        # plt.title(sys.name)
        id += 1

        # print()
    plt.figure(figsize=(16,10))
    plot_and_fit(Nonlin_io_normals)
    plot_and_fit(Hammerstein_sysid_book)
    plot_and_fit(Wiener_sysid_book)
    plot_and_fit(Wiener2_sysid_book)
    plot_and_fit(NDE_squared_sysid_book)
    plot_and_fit(Dynamic_nonlin_sysid_book)
    plt.show()

    # Nonlin_io_normals().get_train_data().plot()
    # Hammerstein_sysid_book().get_train_data().plot()
    # Wiener_sysid_book().get_train_data().plot()
    # NDE_squared_sysid_book().get_train_data().plot()
    # Dynamic_nonlin_sysid_book().get_train_data().plot()
