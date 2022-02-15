# Uncovering the non-equilibrium stationary properties in sparse Boolean networks
This repository reproduces the result of the manuscript *Uncovering the non-equilibrium stationary properties in sparse Boolean networks*, [link to arXiv](https://arxiv.org/abs/2202.06705)
## Requirements
Code requires python > 3.6 and the following libraries:
- networkx
- numpy
- scipy
- numba

By default, Monte Carlo simulation  uses  NVIDIA  GPU. It is recommended to install cupy to benefit of the speedup. Otherwise, simulations are run  in parallel over the CPU
## How to run
Folders are independent of one another. Inside each folder run the python script typing ```python name_of_script.py```. Depending on the script, arguments may be required, use the argument ```--help``` to know more.  The jupyter noteboot inside the folder is used to analyse and produe the plots. 


<a href="https://zenodo.org/badge/latestdoi/447633720"><img src="https://zenodo.org/badge/447633720.svg" alt="DOI"></a>
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

