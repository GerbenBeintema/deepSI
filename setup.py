from setuptools import setup, find_namespace_packages

with open('requirements.txt') as f:
    install_requires = [line for line in f]

packages = [a for a in find_namespace_packages(where='.') if a[:6]=='deepSI']

setup(name = 'deepSI',
      version = '0.2.14',
      description = 'Dynamical system identification',
      author = 'Gerben Beintema',
      author_email = 'g.i.beintema@tue.nl',
      license = 'BSD 3-Clause License',
      python_requires = '>=3.6',
      packages=packages,
      install_requires = install_requires,
      extras_require = dict(
        docs = ['sphinx>=1.6','sphinx-rtd-theme>=0.5']
        )
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