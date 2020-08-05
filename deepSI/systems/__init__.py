from deepSI.systems.System import System, Systems_gyms, System_SS, System_IO, load_system, System_Deriv
from deepSI.systems.simple_systems import sys_ss_test

# from deepSI.systems.book_based_systems import nonlin_Ibased_normals_system,\
#                 Hammerstein_sys_ID_book, Wiener_sys_ID_book, WienerV2_sys_ID_book, NDE_squared_sys_ID_book, dynamic_nonlin_sys_ID_book

from deepSI.systems.test_system import test_system, linear_gaussian_system
from deepSI.systems.double_bath_system import double_bath_system
from deepSI.systems.filter_systems import cheby1,butter
from deepSI.systems.pendulum_system import pendulum_system
from deepSI.systems.NarendraLiBenchmark import NarendraLiBenchmark
from deepSI.systems.book_based_systems import nonlin_Ibased_normals_system,\
                Hammerstein_sys_ID_book, Wiener_sys_ID_book, WienerV2_sys_ID_book, NDE_squared_sys_ID_book, dynamic_nonlin_sys_ID_book
from deepSI.systems.example_2_non_lin import example_2_non_lin
from deepSI.systems.nonlinear_RLC import nonlinear_RLC
all_systems = [test_system,double_bath_system,cheby1,butter,pendulum_system,nonlin_Ibased_normals_system,example_2_non_lin,Hammerstein_sys_ID_book,Wiener_sys_ID_book,
                WienerV2_sys_ID_book, NDE_squared_sys_ID_book, dynamic_nonlin_sys_ID_book, nonlinear_RLC]
from deepSI.systems.nonlindrag import nonlindrag_sys, CED_sim
from deepSI.systems.Lorenz_attractor import Lorenz_attractor
from deepSI.systems.Van_der_Pol_oscillator import Van_der_Pol_oscillator