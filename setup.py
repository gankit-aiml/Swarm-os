from setuptools import setup, find_packages

setup(
    name='swarm-os',
    version='1.0.0',
    description='Decentralized Asymmetric Edge Inference Protocol for AMD',
    author='Your Team Name',
    packages=find_packages(), # Automatically finds the swarm_os folder
    install_requires=[
        'torch',
        'transformers',
        'accelerate',
        'pyzmq',
        'zeroconf',
        'numpy'
    ],
    entry_points={
        'console_scripts':[
            # This creates a terminal command called 'swarm-os'
            # that triggers the main() function inside swarm_os/cli.py
            'swarm-os=swarm_os.cli:main', 
        ],
    },
)