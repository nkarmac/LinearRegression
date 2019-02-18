# Linear Regression

This repo contains three algorithms for training a linear regression model from a dataset of points.<br>

normaleq.py uses the normal equation to give the optimal weights.<br>
batchgradient.py processes the whole dataset and uses gradient descent to update weights.<br>
stochasticgradient.py uses separates the data into batches to run stochastic gradient descent.<br>

Normal equations produces the exact ideal weights for the vast majority of smaller datasets, and stochastic gradient descent is best run on
datasets too large for normal equations (since it is iterative). Batch gradient becomes not useful in this regard, since it is suboptimal and too slow.
	
Uncomment the weight vector print blocks if required.