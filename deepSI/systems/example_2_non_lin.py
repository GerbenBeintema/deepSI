from deepSI.systems.System import System_io, System_data
import numpy as np
import deepSI

class example_2_non_lin(System_io):
    def __init__(self):
        super(example_2_non_lin, self).__init__(na=2,nb=1)

    def IO_step(self,uy):
        ukm1, ykm2, ykm1 = uy
        ystar = ykm1*ykm2*(ykm1+2.5)/(1+ykm1**2*ykm2**2)+ukm1
        return ystar

if __name__=='__main__':
    sys = example_2_non_lin()
    sys.get_train_data().plot()
    sys.get_test_data().plot(show=True)