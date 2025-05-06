from setuptools import setup

setup(
    name='pomdp-domains',
    version='0.0.1',
    packages=['pdomains'],
    install_requires=[
        'numpy',
        'gymnasium',
        'matplotlib',
        'mujoco_py',
        'gin-config',
    ],
)