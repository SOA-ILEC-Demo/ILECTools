
from setuptools import setup

setup(name='ilectools',
      version='0.1',
      description='Tools for processing ILEC data and reporting on it',
      url='https://github.com/brianholland/ILECTools.git',
      author='Brian Holland',
      author_email='brian.d.holland@gmail.com',
      license='MIT',
      packages=['ilectools'],
      install_requires=['pandas','numpy','matplotlib'  'pandas-datareader'
                ],
      zip_safe=False)
