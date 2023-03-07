Installation
============

You can use any python distribution but I recommend using anaconda python (https://www.anaconda.com/products/individual) for it reduces the number of installation complications to almost zero. 

deepSI has been verified to work for 3.7 <= python <= 3.9

After installing anaconda (adding to path is not necessary) you can open the "anaconda promp" or "anaconda cmd" and type the following commands to install deepSI

First go to https://pytorch.org/get-started/locally/ and use the instruction to install your desired PyTorch version.

Install deepSI

.. code:: sh

    conda install -c anaconda git
    pip install git+https://github.com/GerbenBeintema/deepSI@master

(If you encounter any problem I recommend retrying after setting up a new conda enviroment (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))

To open a jupyter notebook you can use the anaconda navigator or type "jupyter notebook" in the address bar of directory you want to start or open a notebook.

Next steps and support
----------------------

Tutorial, examples, read reference API

