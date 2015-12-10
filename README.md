## Synopsis


Parallel implementation of neural networks on Janus.

## Code Example


## Motivation


## Installation


## API Reference


## Tests

Describe and show how to run the tests with code examples.


## License

##program structure

#data

struct nn - neural network

struct training_data
Training names
 incremental
Activation Function
 sigmoid
Error Function
  tanh
Stop Criteria
 mean square error
Net Type
  next layer
  all layers

struct neuron
   index to the first connection
   index to the last connection
   sum of inputs
   steepness of activation Function
   activation Function

struct layer
   a pointer to the first neuron
   a pointer to the last neuron


struct neural network
    learning rate
    learning momentum for backpropogation algorithm
    connection rate (0 - 1)
    network Type
    layer first layer
    layer last layer
    total neurons
    input neurons
    output neurons
    weight array
    neuron connection array
    total connections
    type output - stores output
    number of data used to calculate MSE
    MSE value
    number of outputs which would fail
    max difference between actual output and expected outputs
    error Function
    stop Function
    callback function during Training
    user defined data

connection - weight between two neurons
     from neuron
     to neuron


training neural network

struct train data
      num_data
      num_input
      num output
      int input array
      int output array


function train_data
      
