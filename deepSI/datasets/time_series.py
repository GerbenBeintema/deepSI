
#'https://arxiv.org/pdf/2012.07436.pdf'

import deepSI
from deepSI.system_data.system_data import System_data, System_data_list
import os
from scipy.io import loadmat

import tempfile
import os.path
from pathlib import Path
import numpy as np

from deepSI.datasets.dataset_utils import *


def add_time(ETT, name, eq):
    ETT[name] = [eq(t) for t in ETT['date']]
    ETT[name + ' sin'] = np.sin(ETT[name]*2*np.pi)
    ETT[name + ' cos'] = np.cos(ETT[name]*2*np.pi)

def load_cor(name):
    import pandas as pd
    ETT = pd.read_csv(name)
    ETT['date'] = pd.to_datetime(ETT['date'])
    add_time(ETT, 'time of day', lambda t: (t.minute*60+t.hour)/24)
    add_time(ETT, 'time of week', lambda t: (t.minute/60/60+t.hour/24 + t.weekday())/7)
    # add_time(ETT, 'time of year', lambda t: (t.minute/60/60+t.hour/24 + t.day_of_year)/357)
    target = ETT['OT']
    loads = ETT[['HUFL','HULL','MUFL','MULL','LUFL','LULL']]
    times = ETT[['time of day sin', 'time of day cos', 'time of week sin', 'time of week cos']]#, 'time of year sin', 'time of year cos']]
    time = ETT['date']
    return ETT, np.array(target), np.array(loads), np.array(times), np.array(time)

def ETT_data_get(name,dir_placement=None,force_download=False,split_data=True,include_time_in_u=False, full_return=False):
    url = name
    file_name = url.split('/')[-1]
    download_size = None
    save_dir = cashed_download(url,'beihang', dir_placement=dir_placement,\
        download_size=download_size,force_download=force_download,zipped=False)
    file_loc = os.path.join(save_dir, file_name)

    ETT, target, loads, times, time = load_cor(file_loc)
    if full_return:
        return ETT, target, loads, times, time

    u = loads
    if include_time_in_u:
        u = np.concatenate([u, times],axis=1)
    y = target

    sys_data = System_data(u=u, y=y, dt=15/60/24)
    return sys_data.train_test_split(split_fraction=4/20) if split_data else sys_data
    # if not split_data:

    # train_full, test = sys_data.train_test_split(split_fraction=4/20)
    # # train, val = train_full.train_test_split(split_fraction=4/16)
    # return train_full, test

def ETTm1(dir_placement=None,force_download=False,split_data=True,include_time_in_u=False, full_return=False):
    url = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv'
    return ETT_data_get(url, dir_placement=dir_placement,force_download=force_download, \
        split_data=split_data,include_time_in_u=include_time_in_u, full_return=full_return)

def ETTm2(dir_placement=None,force_download=False,split_data=True,include_time_in_u=False, full_return=False):
    url = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv'
    return ETT_data_get(url, dir_placement=dir_placement,force_download=force_download, \
        split_data=split_data,include_time_in_u=include_time_in_u, full_return=full_return)

def ETTh1(dir_placement=None,force_download=False,split_data=True,include_time_in_u=False, full_return=False):
    url = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv'
    return ETT_data_get(url, dir_placement=dir_placement,force_download=force_download, \
        split_data=split_data,include_time_in_u=include_time_in_u, full_return=full_return)

def ETTh2(dir_placement=None,force_download=False,split_data=True,include_time_in_u=False, full_return=False):
    url = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv'
    return ETT_data_get(url, dir_placement=dir_placement,force_download=force_download, \
        split_data=split_data,include_time_in_u=include_time_in_u, full_return=full_return)