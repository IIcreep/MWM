# Code for paper "Small Data Reliability Analysis: A Mixture Weibull Model and Applications"
This is a Python implementation of the Mixture Weibull model and its related experiments proposed by our paper, titled "Small Data Reliability Analysis: A Mixture Weibull Model and Applications".

## Concrete.py
Concrete.py is used to fit the distribution of concrete three-point bending test. Taking w=40mm and α=0.075 as an example, a 6-parameter mixture Weibull distribution is employed. Data regeneration is performed by calling ```smote``` and ```CircularBlockBootstrap```, while parameter fitting algorithm is called by invoking the ```NT```.
