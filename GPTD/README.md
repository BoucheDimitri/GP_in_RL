# Sparsified-online Gaussian Process Temporal Difference

Dependencies are classic scientific python libraries (**numpy** and **matplotlib** only).  
This project has been developped under **Python 3.6**

## Getting started
Running **main.py** will create a predetermined maze environement, simulate trajectories 
on this environement, run online-sparsified GPTD and TD0 and show the estimated value maps. 
The policy is a policy that goes south with probability 0.8 and choose a random action with probability 0.2
