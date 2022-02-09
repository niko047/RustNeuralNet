# RustNeuralNet
Toy example (for rust learning purposes) of a toy neural network built in Rust from absolute scratch, without external libraries.
It aims at mimicking the simple univariate function 
<img src="https://latex.codecogs.com/gif.latex?O_t=\text { Onset event at time bin } t " /> 
$$y = x**4 - 2 * x**2$$ by NN-Tuning the weights of a simpler y = a + b*x**2 function ( in the interval -2,2), minimizing RMSE.
