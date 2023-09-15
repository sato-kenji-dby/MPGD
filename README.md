# MPGD

Tensorflow >=1.9
Pytorch = 1.13.0

**Env.py**: the environment where mobile users and platform interact with each other.

**DNC.py**: a simple version of DNC, referring to [paper]( https://ieeexplore.ieee.org/document/8967118).

**model.py**: an implementation of PPO, A2C, MPGD witn DNC module for solving partially observable Markov decision process.

**main.py**: the main function, supporting `MPGD`, `DNC_A2C`, `MAPPO`, `MAA2C`, `Greedy`, and `Random`.

python main.py --algo=MPGD 
