Installation
============

You can use any python distribution but I recommend using anaconda python (https://www.anaconda.com/products/individual) for it reduces the number of installation complications to almost zero. 

deepSI has been verified to work for 3.7 <= python <= 3.9

After installing anaconda (adding to path is not necessary) you can open the "anaconda promp" or "anaconda cmd" and type the following commands to install deepSI

.. code:: sh

    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch 
    conda install -c anaconda git
    pip install git+git://github.com/GerbenBeintema/deepSI@master

(optional; setup an anaconda environment)

To open a jupyter notebook you can use the anaconda navigator or type "jupyter notebook" in the address bar of directory you want to start or open a notebook.

Next steps and support
----------------------

Tutorial, examples, read reference API

