# Code for paper "Small Data Reliability Analysis: A Mixture Weibull Model and Applications"
This is a Python implementation of the Mixture Weibull model and its related experiments proposed by our paper, titled "Small Data Reliability Analysis: A Mixture Weibull Model and Applications".

## Concrete.py
concrete1.py is a function designed to handle the case where the pre-crack length is 0 in the data. concrete2.py is a function designed to handle the case where the pre-crack length is not 0 in the data. In this situation, the denominator of the probability density function becomes 0, which leads to an inability to compute the function.


