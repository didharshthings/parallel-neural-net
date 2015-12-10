/*
 * Term Project for CSCI5576
 * Author - Siddharth Singh
 * Neural Networks helper
 *
 * for each epoch
    for each training data instance
     propagate error through the network
      adjust the weights
       calculate the accuracy over training data
       for each validation data instance
        calculate the accuracy over the validation data
          if the threshold validation accuracy is met
           exit training
           else
     continue training
 */

#include <stdlib.h>
#include <srdio.h>
#include <string.h>
#include <math.h>
#include "nn.h"

network_t* create_network(int num_layers, int* layers)
{
    network_t *net;
    int i,j,k;
    net = (network_t *)malloc(sizeof(network_t));
    net->no_of_layers = num_layers;
    net->layer = (layer_t *) calloc(num_layers,sizeof(layer_t));
    for(i=0;i<num_layers;i++)
    {
        net->layer[i].no_of_neurons = layers[i];
        net->layer[i].neuron = (neuron_t) calloc(layers[i]+1, sizeof(neuron_t));
    }
    for(i=0;i<num_layers;i++)
    {
        layer_t* upper_layer,lower_layer;
        upper_layer =  net->layer[i-1];
        lower_layer = net->layer[i];
        int layer_neurons;
        for(j=0;j<upper_layer->no_of_neurons;j++)
        {
            upper->neuron[j].weight=(float *)calloc(lower->no_of_neurons+1,sizeof(float));
        }
        upper->neuron[n].weight=NULL;

    }
    net->input_layer = &net->layer[0];
    net->output_layer = &net->layer[num_layers-1];

    net->momentum = 0.1;
    net->learning_rate = 0.25;

    //initialize weights,bias and deltas
    for(i=1;i < net->no_of_layers;i++)
    {
        net->layer[i].neuron[net->layer[i].no_of_neurons].output=1.0;
        for(j=0;j < net->layer[i].no_of_neurons;n++)
        {
            for(k=0;k <= net->layer[l-1].no_of_neurons;k++)
            {
                net->layer[i].neuron[j].weight[k]=2.0*((double)random()/RAND_MAX - 0.5);
            }
        }
    }
 return net;
}

void back_propogate(layer_t* lower, layer_t* upper)
{
    int i,j;
    double ouput,error;

    for(i=0;i<= lower->no_of_neurons;i++)
    {
        error=0.0;
        for(j=0;j<upper->no_of_neurons;j++)
        {
            error += upper->neuron[j].weight[i]*upper->neuron[j].error;
        }
        output=lower->neuron[i].output;
        lower->neuron[i].error=output*(1.0-output)*error;
    }

}

void train(network_t *net)
{
    int i,j,k;
    double error;
    
    //backpropogate
    for(i=net->no_of_layers-1;i>1;i--)
    {
        back_propogate(&net->layer[i-1],&net->layer[i]);
    }
    
    //modify weights
    for(i=0;i<net->no_of_layers;i++)
        for(j=0;j<net->layer[i].no_of_neurons;j++)
        {
            error = net->layer[i].neuron[j].error;
            for(k=0;k<= net->layer[i-1].no_of_neurons;k++)
            {
                net->layer[i].neuron[j].weight[k]=net->learning_rate*error*net->layer[i-1].neuron[k].output;
            }
        }
}

void compute(network_t *net, double *input, double *output)
{   
    int i;

    //set input
    for(i=0;i< net->input_layer->no_of_neurons;i++)
    {
        net->input_layer->neuron[i].output = input[i];
    }

    //forward propogate
    for(i=1;i< net->no_of_layers;i++)
    {
        forward_propogate(&net->layer[i-1], &net->layer[i]);
    }
    //get output
    for(i=0; i< net->output->layer->no_of_neurons; i++)
    {
        output[i] = net->output_layer->neuron[i].output;
    }

}

void forward_propogate(layer_t* lower, layer_t* upper)
{
    int i,j;
    double value;

    for(i=0;i< upper->no_of_neurons; i++)
    {
        value = 0.0;
        for(j=0;j <=lower->no_of_neurons;j++)
        {
            value += upper->neuron[i].weight[j] * lower->neuron[j].output;
        }
        upper->neuron[i].activation = value;
        upper->neuron[i].output = sigma(value);
    }
}

double sigma( double x)
{
    return 1.0/(1.0 + exp(-x));
}

void compute_error(network_t *net, double* target)
{
    int i;
    double output, error;

    net->global_error = 0.0;
    for(i=0; i < net-> output_layer->no_of_neurons; i++)
    {
        output = net->output_layer->neuron[i].output;
        error = target[i] - output;
        net->output_layer->neuron[i].error = output*(1.0-output)*error;
        net->global_error += error*error;
    }  
    net->global_error *= 0.5;
    return net->global_error;
}












