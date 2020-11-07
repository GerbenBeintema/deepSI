from setuptools import setup

setup(name='deepSI',
      version='0.2',
      description = 'Dynamical system identification',
      author = 'Gerben Beintema',
      author_email = 'g.i.beintema@tue.nl',
      license = 'BSD 3-Clause License',
      python_requires = '>=3.6',
      install_requires=['numpy','matplotlib', 'tqdm', 'progressbar','torch','scikit-learn', 'gym']
     )

  # extras_require = dict(
  #   docs=['Sphinx>=1.6','scipy>=0.13','matplotlib>=1.3'],
  #   matrix_scipy=['scipy>=0.13'],
  #   matrix_mkl=['mkl'],
  #   export_mpl=['matplotlib>=1.3','pillow>2.6'],
  #   import_gmsh=['meshio'],
  # ),
  # command_options = dict(
  #   test=dict(test_loader=('setup.py', 'unittest:TestLoader')),
  # ),