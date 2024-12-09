import deepSI.models
import deepSI.fitting
import deepSI.networks
import deepSI.normalization
from nonlinear_benchmarks import Input_output_data

#default imports
from deepSI.models import SUBNET, SUBNET_CT, Custom_SUBNET, Custom_SUBNET_CT
from deepSI.fitting import fit
from deepSI.networks import MLP_res_net
from deepSI.normalization import Norm, get_nu_ny_and_auto_norm
