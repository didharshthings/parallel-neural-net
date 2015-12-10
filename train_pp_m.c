/*
 * Term Project for CSCI 5576
 * Model Parallelism
 * Splitting each layer at different nodes
 */

#include <getopt.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "nn.h"
#include <mpi.h>

int ReadFile(char *file_name, int valuesPerLine, int numLines, float* arr){
	FILE *ifp;
	int i, j, val;
	char *mode = "r";
	ifp = fopen(file_name, mode);

	if (ifp == NULL) {
		return 1;
	}

	i = 0;
	while((!feof(ifp)) && (i < (valuesPerLine*numLines)))
	{
		fscanf(ifp, "%d ", &val);

		arr[i] = val;

		i++;
	}

	// closing file
	fclose(ifp);

	return 0;
}

int main(int argc, char** argv)
{

    int num_neurons;
    int rank, np;
    float input[8];
    float target[4];
    float output[4];
    float error;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    //MPI Derived data type
    MPI_Datatype global_weights;
    MPI_Type_contiguous(27,MPI_FLOAT,&global_weights);
    MPI_Type_commit(&global_weights);

    // Initialize the pretty printer
    init_pprintf( rank );
    pp_set_banner( "main" );

    int num_inputs = 2;
    int num_outputs = 1;


  // file handling stuff
    float* trainingSamples;
    float* trainingTargets;
    int numTrainingSamples, numTestSamples;

    trainingSamples = (float *) calloc(num_inputs * 2, sizeof(float));
    trainingTargets = (float *) calloc(num_outputs * 1, sizeof(float));
    char* trainingFile, * trainingTargetFile, * testingFile;

    #define inputs(i) (trainingSamples + i * num_inputs)
    #define targets(i) (trainingTargets + i* num_outputs)



    trainingFile = "xor.txt";
    trainingTargetFile = "xor_targets.txt";





   input[0] = 1.0; input[1] = 1.0;
   target[0] = 0.0;
   input[2] = 1.0; input[3] = 0.0;
   target[1]= 1.0;
   input[4] = 0.0; input[5] = 1.0;
   target[2] = 1.0;
   input[6] = 0.0; input[7] = 0.0;
   target[3] = 0.0;

   int layers[3];
   layers[0] = 3;
   layers[1] = 3;
   layers[2] = 3;

   //one layer in each node

   int upper_layer,lower_layer;

   if (rank == 0 )
   {
    upper_layer = NULL;
    lower_layer = 1;
   }
   else if (rank == np)
   {
    upper_layer = np-1;
    lower_layer = 0;
   }
   else
   {
    upper_layer = rank-1;
    lower_layer = rank+1;
   }

   layer_t local_layer;

   local_layer = (layer_t *)calloc(1,sizeof(layer_t));

   //allocate neurons
   layer->no_of_neurons = 3;
   layer->neurons = (neuron_t *) calloc(4,sizeof(neuron_t));

   if(rank ==0)
   {
        //set input
        int i;
        if(rank == 0)
        {
          ReadFile(trainingFile, num_inputs, 4, trainingSamples);
          ReadFile(trainingTargetFile, num_outputs, 4, trainingTargets);
        }
        for(i=0;i<layer->no_of_neurons;l++)
        {
            layer->neuron[i].output = input[i];
        }:

   }

   if(rank == np-1)
   {
        int i;
        for(i=0;i< layer->no_of_neurons;i++)
        {
            output[i] = layer->neuron[i].output
        }

        //MPI_Send

   }

   //start training

   int epochs = 0;
   int error;
   // MPI Derived datatype
   while(epochs<=2)
   {
     //forward pass
      int i,j;

      if(rank = np-1)
      {
        float* recieve output;
        MPI_Recieve(&recieve_output,num_neurons,MPI_FLOAT,rank - 1,0,MPI_COMM_WORLD);//output

        //compute output error
        int global_error = 0.0;
        for (i = 0; i < layer->no_of_neurons; i++) {
          output = layer->neuron[i].output;
          error = target[i] - output;
          layer->neuron[i].error = output * (1.0 - output) * error;
          global_error += error * error;
        }
        global_error *= 0.5;

        //backpropogation

        float *send_weights;
        float *send_error;
        MPI_Send(&send_weights, num_neurons, MPI_FLOAT,rank-1,0,MPI_COMM_WORLD) //weights
        MPI_Send(&send_error, num_neurons, MPI_FLOAT,rank-1,0,MPI_COMM_WORLD) //error

      }
      else if (rank = 0)
      {
           float* send_output;
           MPI_Send(&send_output, num_neurons, MPI_FLOAT, rank +1,0,MPI_COMM_WORLD);

           //back propogation
        MPI_Recieve(&send_weights, num_neurons, MPI_FLOAT,rank+1,0,MPI_COMM_WORLD) //weights
        MPI_Recieve(&send_error, num_neurons, MPI_FLOAT,rank+1,0,MPI_COMM_WORLD) //error

      }
      else
      {
        float* recieve output;
        MPI_Recieve(&recieve_output,num_neurons,MPI_FLOAT,rank - 1,0,MPI_COMM_WORLD);//output array
        //compute
        float* send_output;
        MPI_Send(&send_output, num_neurons, MPI_FLOAT, rank +1,0,MPI_COMM_WORLD);

        //back propogation

        MPI_Recieve(&send_weights, num_neurons, MPI_FLOAT,rank+1,0,MPI_COMM_WORLD) //weights
        MPI_Recieve(&send_error, num_neurons, MPI_FLOAT,rank+1,0,MPI_COMM_WORLD) //error

         for (nl = 0; nl <= lower->no_of_neurons; nl++)
         {
             error = 0.0;
             for (nu = 0; nu < upper->no_of_neurons; nu++)
             {
                error += upper->neuron[nu].weight[nl] * upper->neuron[nu].error;
             }
               output = lower->neuron[nl].output;
               lower->neuron[nl].error = output * (1.0 - output) * error;
          }

        MPI_Send(&send_weights, num_neurons, MPI_FLOAT,rank-1,0,MPI_COMM_WORLD) //weights
        MPI_Send(&send_error, num_neurons, MPI_FLOAT,rank-1,0,MPI_COMM_WORLD) //error




      }
     epochs++;
   }
   net_print(global_net);
}
