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
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "nn.h"

/* Activation function of every non-input neuron. Declared here because it is
 * used by forward_propogate() below and is not part of the public nn.h API. */
double sigma (double x);

/* Helper prototypes. These are defined below but referenced by the training
 * helpers above them, so they are declared up front. */
void forward_propogate(layer_t* lower, layer_t* upper);
void back_propogate(layer_t* lower, layer_t* upper);
void train(network_t *net);
void compute(network_t *net, double *input, double *output);
double compute_error(network_t *net, double* target);
void free_network(network_t *net);

/*!\brief Allocate an empty network with the given layer sizes.
 *
 * This is a thin convenience wrapper around the nn.c allocation routines. It
 * exists so that the training helpers in this file can be exercised on their
 * own; the drivers in the repository build networks with net_allocate_l().
 */
network_t* create_network(int num_layers, int* layers)
{
    network_t *net;
    int i, j, k;

    if (num_layers < 2 || layers == NULL) {
        return NULL;
    }

    net = (network_t *) malloc(sizeof(network_t));
    if (net == NULL) {
        return NULL;
    }

    net->no_of_layers = num_layers;
    net->no_of_patterns = 0;
    net->momentum = 0.1;
    net->learning_rate = 0.25;
    net->global_error = 0.0;

    net->layer = (layer_t *) calloc(num_layers, sizeof(layer_t));
    if (net->layer == NULL) {
        free(net);
        return NULL;
    }

    /* One extra neuron per layer is the bias neuron. */
    for (i = 0; i < num_layers; i++) {
        net->layer[i].no_of_neurons = layers[i];
        net->layer[i].neuron =
            (neuron_t *) calloc(layers[i] + 1, sizeof(neuron_t));
        if (net->layer[i].neuron == NULL) {
            int l;
            for (l = 0; l < i; l++) {
                free(net->layer[l].neuron);
            }
            free(net->layer);
            free(net);
            return NULL;
        }
    }

    net->input_layer = &net->layer[0];
    net->output_layer = &net->layer[num_layers - 1];

    /* Allocate the incoming weight vector of every neuron above the input
     * layer. The bias neuron of a layer has no incoming weights. */
    for (i = 1; i < num_layers; i++) {
        layer_t *lower = &net->layer[i - 1];
        layer_t *upper = &net->layer[i];

        for (j = 0; j < upper->no_of_neurons; j++) {
            upper->neuron[j].weight =
                (double *) calloc(lower->no_of_neurons + 1, sizeof(double));
            upper->neuron[j].delta =
                (double *) calloc(lower->no_of_neurons + 1, sizeof(double));
            if (upper->neuron[j].weight == NULL ||
                upper->neuron[j].delta == NULL) {
                return NULL;
            }
        }
    }

    /* Set the bias output of each non-input layer to 1, then randomize the
     * weights of the network just as nn.c does. */
    for (i = 1; i < num_layers; i++) {
        net->layer[i].neuron[net->layer[i].no_of_neurons].output = 1.0;

        for (j = 0; j < net->layer[i].no_of_neurons; j++) {
            for (k = 0; k <= net->layer[i - 1].no_of_neurons; k++) {
                net->layer[i].neuron[j].weight[k] =
                    2.0 * ((double) random() / RAND_MAX - 0.5);
            }
        }
    }

    return net;
}

/*!\brief Release a network created with create_network().
 *
 * Frees the weight and delta vectors of every neuron above the input layer,
 * then the neurons, the layers and the network itself. Only the vectors that
 * create_network() allocated are freed, which is why the bias neuron of each
 * layer (whose weight pointer is NULL) is skipped.
 */
void free_network(network_t *net)
{
    int i, j;

    if (net == NULL) {
        return;
    }

    for (i = 1; i < net->no_of_layers; i++) {
        for (j = 0; j < net->layer[i].no_of_neurons; j++) {
            free(net->layer[i].neuron[j].weight);
            free(net->layer[i].neuron[j].delta);
        }
    }
    for (i = 0; i < net->no_of_layers; i++) {
        free(net->layer[i].neuron);
    }
    free(net->layer);
    free(net);
}

void back_propogate(layer_t* lower, layer_t* upper)
{
    int i, j;
    double output, error;

    if (lower == NULL || upper == NULL) {
        return;
    }

    for (i = 0; i <= lower->no_of_neurons; i++)
    {
        error = 0.0;
        for (j = 0; j < upper->no_of_neurons; j++)
        {
            error += upper->neuron[j].weight[i] * upper->neuron[j].error;
        }
        output = lower->neuron[i].output;
        lower->neuron[i].error = output * (1.0 - output) * error;
    }
}

void train(network_t *net)
{
    int i, j, k;
    double error;
    double delta;

    if (net == NULL) {
        return;
    }

    //backpropogate
    for (i = net->no_of_layers - 1; i > 1; i--)
    {
        back_propogate(&net->layer[i - 1], &net->layer[i]);
    }

    //modify weights
    for (i = 1; i < net->no_of_layers; i++)
    {
        for (j = 0; j < net->layer[i].no_of_neurons; j++)
        {
            error = net->layer[i].neuron[j].error;
            for (k = 0; k <= net->layer[i - 1].no_of_neurons; k++)
            {
                /* Apply the weight change, accumulated with the previous
                 * delta scaled by the momentum term. */
                delta = net->learning_rate * error *
                        net->layer[i - 1].neuron[k].output;
                net->layer[i].neuron[j].weight[k] += delta +
                    net->momentum * net->layer[i].neuron[j].delta[k];
                net->layer[i].neuron[j].delta[k] = delta;
            }
        }
    }
}

void compute(network_t *net, double *input, double *output)
{
    int i;

    if (net == NULL || input == NULL || output == NULL) {
        return;
    }

    //set input
    for (i = 0; i < net->input_layer->no_of_neurons; i++)
    {
        net->input_layer->neuron[i].output = input[i];
    }

    //forward propogate
    for (i = 1; i < net->no_of_layers; i++)
    {
        forward_propogate(&net->layer[i - 1], &net->layer[i]);
    }

    //get output
    for (i = 0; i < net->output_layer->no_of_neurons; i++)
    {
        output[i] = net->output_layer->neuron[i].output;
    }
}

void forward_propogate(layer_t* lower, layer_t* upper)
{
    int i, j;
    double value;

    if (lower == NULL || upper == NULL) {
        return;
    }

    for (i = 0; i < upper->no_of_neurons; i++)
    {
        value = 0.0;
        for (j = 0; j <= lower->no_of_neurons; j++)
        {
            value += upper->neuron[i].weight[j] * lower->neuron[j].output;
        }
        upper->neuron[i].output = sigma(value);
    }
}

double sigma( double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double compute_error(network_t *net, double* target)
{
    int i;
    double output, error;

    if (net == NULL || target == NULL) {
        return 0.0;
    }

    net->global_error = 0.0;
    for (i = 0; i < net->output_layer->no_of_neurons; i++)
    {
        output = net->output_layer->neuron[i].output;
        error = target[i] - output;
        net->output_layer->neuron[i].error = output * (1.0 - output) * error;
        net->global_error += error * error;
    }
    net->global_error *= 0.5;

    return net->global_error;
}
