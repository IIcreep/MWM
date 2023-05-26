# Code for paper "Small Data Reliability Analysis: A Mixture Weibull Model and Applications"
This is a Python implementation of the Mixture Weibull model and its related experiments proposed by our paper, titled "Small Data Reliability Analysis: A Mixture Weibull Model and Applications".

## Concrete
concrete1.py is a function designed to handle the case where the pre-crack length is 0 in the data. concrete2.py is a function designed to handle the case where the pre-crack length is not 0 in the data. In this situation, the denominator of the probability density function becomes 0, which leads to an inability to compute the function.

## Weibull_modulus
The purpose of this code is to calculate the Weber modulus of a material from some data.Two functions are defined:

1. Function E(L, F, B, H, d) takes length (L), force (F), width (B), height (H), and displacement (d) as parameters, calculates and returns the value of the elastic modulus (E).

2. Function Ur(E, v) takes the elastic modulus (E) and Poisson's ratio (v) as parameters, calculates and returns the value of the displacement response (Ur).
