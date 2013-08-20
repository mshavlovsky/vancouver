vancouver
=========

Vancouver crowdsourcing algorithm.

This is an implementation of the Vancouver crowdsourcing algorithm.

The algorithm is implemented in the vancouver.py file. 

Requires: numpy.  It would be easy to rewrite the code and 
eliminate this dependency, if desired.

The analysis directory contains some code that allows the analysis of the
accuracy of the algorithm.  To use that code, tweak the constants in 
crowdsource.py, the user and item models, etc, to your heart's content, 
and run the code via: 

./crowdsource.py

Please refer to the unit-test for an example of how to call the algorithm. 
Please refer to the publications of Luca de Alfaro and Michael Shavlovsky
for an explanation of the algorithm.
