'''
Created on 2022-12-23
@author: Sambit Giri
Setup script
'''

from setuptools import setup, find_packages
#from distutils.core import setup

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='toolslyman',
      version='0.1.0',
      author='Sambit Giri',
      author_email='sambit.giri@gmail.com',
      packages=find_packages("src"),
      package_dir={"": "src"},
      package_data={'toolslyman': ['input_data/*']},
      install_requires=requirements,
      include_package_data=True,
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
)
