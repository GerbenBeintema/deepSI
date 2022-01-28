

import urllib.request
from urllib import request
import os
import os.path
from pathlib import Path
from sys import platform
import shutil

def get_work_dirs():
    '''A utility function which gets the utility directories for each OS

    It creates a working directory called deepSI 

        in LOCALAPPDATA for windows

        in ~/.deepSI/ for unix like

        in ~/Library/Application Support/deepSI/ for darwin

    it creates two directories inside of the deepSI directory

        data_sets : cache location of the downloaded data sets

        checkpoints : used during training of torch models

    Returns
    -------
    dict(base=base_dir, data_sets=data_sets_dir, checkpoints=checkpoints_dir)
    '''

    def mkdir(directory):
        if os.path.isdir(directory) is False:
            os.mkdir(directory)

    from sys import platform
    if platform == "darwin": #not tested but here it goes
        base_dir = os.path.expanduser('~/Library/Application Support/deepSI/')
    elif platform == "win32":
        base_dir = os.path.join(os.getenv('LOCALAPPDATA'),'deepSI/')
    else: #unix like, might be problematic for some weird operating systems.
        base_dir = os.path.expanduser('~/.deepSI/')#Path('~/.deepSI/')
    mkdir(base_dir)
    data_sets_dir = os.path.join(base_dir,'data_sets/')
    mkdir(data_sets_dir)
    checkpoints_dir = os.path.join(base_dir,'checkpoints/')
    mkdir(checkpoints_dir)
    return dict(base=base_dir, data_sets=data_sets_dir, checkpoints=checkpoints_dir)

def clear_cache():
    '''Delete all cached downloads'''
    
    temp_dir = get_work_dirs()['data_sets']
    for l in ['EMPS','CED','F16','WienHammer','BoucWen','ParWHF','WienerHammerBenchMark','Silverbox','Cascaded_Tanks']:
        try:
            shutil.rmtree(os.path.join(temp_dir,l))
        except FileNotFoundError:
            pass
    try:
        shutil.rmtree(os.path.join(temp_dir,'DaISy_data'))
    except FileNotFoundError:
        pass

def clear_checkpoints():
    '''Delete all saved torch checkpoints'''
    checkpoints_dir = get_work_dirs()['checkpoints']
    try:
        shutil.rmtree(checkpoints_dir)
    except FileNotFoundError:
        pass

import progressbar
class MyProgressBar():
    def __init__(self,download_size):
        self.pbar = None
        self.download_size = download_size

    def __call__(self, block_num, block_size, total_size):
        total_size = self.download_size
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def cashed_download(url,name_dir,zip_name=None,dir_placement=None,download_size=None,force_download=False,zipped=True):
    '''url is the file to be downloaded
    name_dir is the directory name where the file and the contents of the file will be saved
    dir_placement is an optional argument that gives the location of the downloaded file 
    if it is none it will download to the temp dir
    if dir_name is None it will be saved in the temp directory of the system'''

    #finding/making directories
    if dir_placement is None:
        p = get_work_dirs()['data_sets'] #use temp dir
    else:
        p = Path(dir_placement) #use given dir
    save_dir = os.path.join(p,Path(name_dir))
    if os.path.isdir(save_dir) is False:
        os.mkdir(save_dir)
    file_name = url.split('/')[-1] if zip_name==None else zip_name
    save_loc = os.path.join(save_dir,file_name) 


    if os.path.isfile(save_loc) and not force_download:
        return save_dir

    if force_download:
        print(f'(re-)downloading dataset from {url} \n in {save_loc}')
    else:
        print(f'dataset not found downloading from {url} \n in {save_loc}')

    if 'drive.google' in url:
        download_file_from_google_drive(url, save_loc)
    else:
        from http.client import IncompleteRead
        tries = 0
        while True:
            try:
                if download_size is None:
                    urllib.request.urlretrieve(url, save_loc)# MyProgressBar() is a steam so no length is given
                    break
                else:
                    urllib.request.urlretrieve(url, save_loc, MyProgressBar(download_size=int(download_size)))
                    break
            except IncompleteRead:
                tries += 1
                print('IncompleteRead download failed, re-downloading file')
                download_size = None
                if tries==5:
                    assert False, 'Download Fail 5 times exiting.'


    if not zipped: return save_dir
    print('extracting file...')
    from zipfile import ZipFile
    
    ending = file_name.split('.')[-1]
    if ending=='gz':
        import shutil
        import gzip
        with open(os.path.join(save_dir,file_name[:-3]), 'wb') as f_out:
            with gzip.open(save_loc, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
    elif ending=='zip':
        File = ZipFile
        with File(save_loc) as Obj:
            Obj.extractall(save_dir)
    else:
        raise NotImplementedError(f'file {file_name} type not implemented')
    return save_dir



import requests

def download_file_from_google_drive(url, destination):
    id = url.split('/')[-2]
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == '__main__':
    import deepSI
    sys_data = deepSI.datasets.CED(split_data=False)
    sys_data.plot(show=True)

    # filename = './file.zip'
    # url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/EMPS/EMPS.zip'
    # urllib.request.urlretrieve(url, filename)

    # resp = urllib.request.urlopen(url)
    # respHtml = resp.read()
    # binfile = open(filename, "wb")
    # binfile.write(respHtml)
    # binfile.close()

    