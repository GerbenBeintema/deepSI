from deepSI.systems.System import System_SS, System_IO
import numpy as np

class sys_ss_test(System_SS): #discrate system single system
    """docstring for test_system"""
    def __init__(self,seed=None):
        '''Noise, system setting and x0 settings'''
        super(sys_ss_test,self).__init__(nx=2)

    def f(self,x,u):
        x1,x2 = x
        x1new = x2*0.7 + u
        x2new = x1*0.7
        return [x1new,x2new]

    def h(self,x):
        x1,x2 = x
        return x1+x2
