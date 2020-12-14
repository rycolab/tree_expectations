from setuptools import setup, find_packages

setup(name='tree_expectations',
      version='1.0',
      description='Tree Expectations',
      author='Ran Zmigrod',
      url='https://github.com/rycolab/tree_expectations',
      install_requires=[
            'numpy',
            'torch'
      ],
      packages=find_packages(),
      )
