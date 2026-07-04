# OptimalNet

DESCRIPTION
-----------

Code for paper "Optimal network structure for collective performance with strategic information sharing".

The code is used for theoretical calculations and numerical simulations of collective performance on networks, with programs written in MATLAB R2024a and Python 3.10.

FILE
-----------

--Sim_fixation_N_5: simulate the fixation probabilities $\rho_{SW}$ and $\rho_{WS}$ for all 21 graphs of size five.

--Sim_full_performance_N_5: simulate the average individual error in the full-sharing and full-withholding states for all 21 graphs of size five.

--Theory_Sim_collective_performance_N_5: calculate the average individual error under strategy evolution for all 21 graphs of size five, using both theoretical analysis and numerical simulations.

--Sim_fixation_regular_graph: simulate the fixation probabilities $\rho_{SW}$ and $\rho_{WS}$ on random regular graphs with increasing degree $k$.

--Sim_collective_performance_regular_graph: simulate the average individual error under strategy evolution on random regular graphs with increasing degree $k$.

--Sim_fixation_BA_allocation: simulate the fixation probabilities $\rho_{SW}$ and $\rho_{WS}$ on a BA network under uniform, proportional, and inverse sample-allocation schemes.

--Sim_collective_performance_BA_allocation: simulate the average individual error under strategy evolution on a BA network under uniform, proportional, and inverse sample-allocation schemes.

The remaining files are subfunctions or data files required to run the above scripts.

NOTE
-----------
To run the code, make sure that all files are in the same folder. The fixation-probability data should be generated before simulating collective performance. For the sample-allocation simulations, other network types can also be used by changing the network type in the corresponding scripts.
