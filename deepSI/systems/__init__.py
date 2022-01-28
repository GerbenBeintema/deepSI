from deepSI.systems.system import System, System_gym, System_ss, System_io, load_system, System_deriv
from deepSI.systems.test_systems import Test_ss_linear1, Test_ss_linear2
from deepSI.systems.double_bath import Double_bath, Cascaded_tanks_continuous
from deepSI.systems.pendulum import Pendulum
from deepSI.systems.narendra_li_benchmark import NarendraLiBenchmark
from deepSI.systems.classical_literature import Nonlin_io_normals,\
                Hammerstein_sysid_book, Wiener_sysid_book, Wiener2_sysid_book,\
                 NDE_squared_sysid_book, Dynamic_nonlin_sysid_book, Nonlin_io_example_2
from deepSI.systems.nonlinear_rlc import Nonlinear_rlc
all_systems = [Test_ss_linear1, Double_bath, Pendulum,Nonlin_io_normals,Nonlin_io_example_2,Hammerstein_sysid_book,Wiener_sysid_book,
                Wiener2_sysid_book, NDE_squared_sysid_book, Dynamic_nonlin_sysid_book, Nonlinear_rlc]
from deepSI.systems.nonlin_drag import Nonlin_drag, Coupled_electric_drive
from deepSI.systems.lorenz_attractor import Lorenz_attractor, Lorenz_attractor_sincos
from deepSI.systems.van_der_pol_oscillator import Van_der_pol_oscillator
from deepSI.systems.double_well import Double_potential_well, Double_potential_well_video
from deepSI.systems.ball_in_box import Ball_in_box, Ball_in_box_video
from deepSI.systems.boucwen import Bouc_wen