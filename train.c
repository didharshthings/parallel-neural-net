/*
Term Project for CSCI5576
Author - Siddharth Singh
*/

#include <getopt.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "nn.h"

#define MAX_FILENAME_LENGTH 100
#define MAX_SIZE 10000
#define MAX_LAYERS 10

int main (int argc, char** argv)
{
  network_t *net;
  int num_pairs;
  int num_inputs;
  int num_outputs;
  double input[8];
  double target[4];
  double output[4];
  double error;
  int i,j;
  int num_neurons[3];

  num_neurons[0] = 2;
  num_neurons[1] = 3;
  num_neurons[2] = 1;

  net = net_allocate_l(3,num_neurons);
  //printf("initial net \n");
  //net_print(net);

  #define inputs(i) (input + i * no_of_inputs)
  #define targets(i) (target + i* no_of_outputs)

  int no_of_inputs = 2;
  int no_of_outputs = 1;
  int no_of_pairs = no_of_inputs/no_of_outputs;

  input[0] = 1.0 ; input[1] = 1.0;
  target[0] = 0.0;
  input[2] = 1.0 ; input[3] = 0.0;
  target[1] = 1.0;
  input[4] = 0.0 ; input[5] = 1.0;
  target[2] = 1.0;
  input[6] = 0.0 ; input[7] = 0.0;
  target[3] = 0.0;

// training
  int epoch = 0;
  double total_error = 0;
  while((epoch <= 100) && (total_error >= 0.0))
  {
    i = rand () % no_of_pairs ;
    net_compute(net, inputs(i), output);

    error = net_compute_output_error(net, targets(i));
    net_train(net);
    if (epoch == 0)
    {
      total_error = error;
    }
    else
    {
      total_error = 0.9 * total_error + 0.1 * error;
    }
    //net_print(net);
    epoch++;
  }
  //test data
  input[0] = 0.0; input[1] = 0.0;
  //use MPI_TYPE create sub array to split input

 net_print(net);
  net_compute(net,inputs(i),output);

  for(j=0;j<1;j++)
  {
  printf("output - %f\n",output[j]);
  }
  net_free(net);
}

// validation
