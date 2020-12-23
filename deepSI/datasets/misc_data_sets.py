

from deepSI.system_data.system_data import System_data, System_data_list
from deepSI.datasets.dataset_utils import *
import numpy as np

import os.path


def sun_spot_data(dir_placement=None,force_download=False,split_data=True):

    url = 'http://www.sidc.be/silso/DATA/SN_y_tot_V2.0.txt'
    download_size=None
    save_dir = cashed_download(url,'sun_spot_data',dir_placement=dir_placement,download_size=download_size,force_download=force_download,zipped=False)
    with open(os.path.join(save_dir,'SN_y_tot_V2.0.txt'),'r') as f:
        data = f.read()[:-2]
    fixed_name = os.path.join(save_dir,'SN_y_tot_V2.0_fix.txt')
    with open(fixed_name,'w') as f:
        f.write(data)
    data = np.loadtxt(fixed_name)

    yEst = data[:,1]
    datasets = System_data(u=None,y=yEst)
    return datasets.train_test_split(split_fraction=0.4) if split_data else datasets #is already splitted


if __name__=='__main__':
    sun_spot_data(split_data=True)[0].plot()
    sun_spot_data(split_data=True)[1].plot(show=True)