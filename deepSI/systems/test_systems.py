
import deepSI
from deepSI.systems.system import System_ss, System_data
import numpy as np


class Test_ss_linear1(System_ss): #discrate system single system
    """docstring for Test_ss_linear1"""
    def __init__(self):
        '''Noise, system setting and x0 settings'''
        super(Test_ss_linear1, self).__init__(nx=2)

    def f(self,x,u):
        x1,x2 = x
        x1new = x2*0.5 + u
        x2new = x1*0.5
        return [x1new,x2new]

    def h(self,x):
        x1,x2 = x
        return x1

class  Test_ss_linear2(System_ss): #https://arxiv.org/pdf/2003.14162.pdf
    """xk+1 = [[0.7,0.8],[0,0.1]] xk + [-1,0.1]*uk + nuk
       yk = xk[0] + wk
       nuk = N(0,0.5) x 2
       wk = N(0,1)"""
    def __init__(self):
        '''Noise, system setting and x0 settings'''
        super(Test_ss_linear2, self).__init__(nx=2)

    def f(self,x,u):
        x1,x2 = x
        x1new = 0.7*x1+0.8*x2 - u + self.random.normal(scale=0.5)
        x2new = x2*0.1 + 0.1*u + self.random.normal(scale=0.5)
        return [x1new,x2new]

    def h(self,x):
        x1,x2 = x
        return x1 + self.random.normal(scale=1)


if __name__=='__main__':
    lingaus = Test_ss_linear2()
    lingaus.get_test_data().plot(show=True)

