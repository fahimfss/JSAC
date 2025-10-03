from setuptools import setup, find_packages

setup(
    name='jsac',
    version='1.0.0',
    description='Jax based real-time Reinforcement Learning for Vision-Based Robotics Utilizing Local and Remote Computers',
    author='Fahim Shahriar',
    author_email='fshahri1@ualberta.ca',
    url='https://github.com/fahimfss/JSAC',
    packages=find_packages(include=['jsac', 'jsac.*']),
    install_requires=[ 
        'gymnasium==1.2.1',
        'seaborn==0.13.2',
        'termcolor==3.1.0',
        'tensorboardX==2.6.4',
        'flax==0.10.7',
        'pyopengl==3.1.10',
        'wandb==0.22.1',
        'tensorflow-probability==0.25.0',
        'imageio==2.37.0',
        'mujoco==3.3.6',
        'dm_control==1.0.34',
        'opencv_python==4.12.0.88',
        'numpy==2.2.6',
        'orbax-checkpoint==0.11.25'
    ],
)
