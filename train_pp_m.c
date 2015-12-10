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
#include <sys/time.h>
#include <sys/types.h>
#include "pprintf.h"

double calctime(struct timeval start, struct timeval end)
{
  double time = 0.0;

  //struct timeval {
  //   time_t      tv_sec;     /* seconds */
  //   suseconds_t tv_usec;    /* microseconds */
  //};
  time = end.tv_usec - start.tv_usec;
  time = time/1000000;
  time += end.tv_sec - start.tv_sec;

  return time;
}

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
   layers[0] = 2;
   layers[1] = 3;
   layers[2] = 1;

   //one layer in each node

   layer_t* local_layer;

   local_layer = (layer_t *)calloc(1,sizeof(layer_t));

   int upper_layer,lower_layer;

   if (rank == 0 )
   {
    upper_layer = MPI_PROC_NULL;
    lower_layer = 1;
    local_layer->no_of_neurons = 2;

   }
   else if (rank == np - 1)
   {
    upper_layer = np-1;
    lower_layer = MPI_PROC_NULL;
    local_layer->no_of_neurons = 1;

   }
   else
   {
    upper_layer = rank-1;
    lower_layer = rank+1;
    local_layer->no_of_neurons = 3;

   }


   //allocate neurons
   local_layer->neuron = (neuron_t *) calloc(4,sizeof(neuron_t));
   MPI_Datatype layer_output;
   MPI_Datatype layer_input;
   MPI_Datatype layer_error;
   MPI_Datatype layer_weights;

   MPI_Type_contiguous(27,MPI_FLOAT,&layer_output);
   MPI_Type_contiguous(27,MPI_FLOAT,&layer_input);
   MPI_Type_contiguous(27,MPI_FLOAT,&layer_weights);
   MPI_Type_contiguous(27,MPI_FLOAT,&layer_error);

   MPI_Type_commit(&layer_output);
   MPI_Type_commit(&layer_input);
   MPI_Type_commit(&layer_weights);
   MPI_Type_commit(&layer_error);

   if(rank == 0)
   {
        //set input
        int i;
        if(rank == 0)
        {
          ReadFile(trainingFile, num_inputs, 4, trainingSamples);
          ReadFile(trainingTargetFile, num_outputs, 4, trainingTargets);
        }
        for(i=0;i<local_layer->no_of_neurons;i++)
        {
            local_layer->neuron[i].output = input[i];
        }

  }


   //start training

   int epochs = 0;
   while(epochs<=2)
   {
     //forward pass
      int i,j;
      pprintf("epoch %d",epochs);
      if(rank == np-1)
      {
        float* recieve_output;
        recieve_output = (float *) calloc(local_layer->no_of_neurons, sizeof(float));
        MPI_Recv(recieve_output,1,layer_output,rank - 1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);//output
        pprintf("Forward propogation over");
        //compute output error
        int global_error = 0.0;
        /*
        for (i = 0; i < layer->no_of_neurons; i++) {
          output = layer->neuron[i].output;
          error = target[i] - output;
          local_layer->neuron[i].error = output * (1.0 - output) * error;
          global_error += error * error;
        }
        */
        global_error *= 0.5;

        //backpropogation

        float *send_weights;
        float *send_error;
        send_weights = (float *) calloc(local_layer->no_of_neurons, sizeof(float));
        send_error = (float *) calloc(local_layer->no_of_neurons, sizeof(float));
        //MPI_Send(send_weights, 1, layer_weights,rank-1,0,MPI_COMM_WORLD); //weights
        //MPI_Send(send_error, 1,layer_error ,rank-1,0,MPI_COMM_WORLD); //error

      }
      else if (rank == 0)
      {
           float* send_output;
           send_output = (float *) calloc(local_layer->no_of_neurons, sizeof(float));
           pprintf("starting forward propogation");
           MPI_Send(send_output, 1, layer_output, rank + 1 ,0,MPI_COMM_WORLD);
           pprintf("FP: sending output\n");
           float* send_weights;
           float* send_error;
           send_weights = (float *) calloc(local_layer->no_of_neurons, sizeof(float));
           send_error = (float *) calloc(local_layer->no_of_neurons, sizeof(float));
           //back propogation
        //MPI_Recv(send_weights, 1, layer_weights,rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); //weights
        //pprintf("recieving weights \n");
        //MPI_Recv(send_error, 1, layer_error,rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); //error
        //pprintf("recieving weights \n");

      }
      else
      {
        float* recieve_output;
        recieve_output = (float *) calloc(local_layer->no_of_neurons, sizeof(float));
        MPI_Recv(recieve_output,1,layer_output,rank - 1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);//output array
        pprintf("FP :recieving output\n");
        //compute
        float* send_output;
        send_output = (float *) calloc(local_layer->no_of_neurons, sizeof(float));
        MPI_Send(send_output, 1 , layer_output, rank +1,0,MPI_COMM_WORLD);
        pprintf("FP :sending output");
        float* send_weights;
        float* send_error;
        send_weights = (float *) calloc(local_layer->no_of_neurons, sizeof(float));
        send_error = (float *) calloc(local_layer->no_of_neurons, sizeof(float));
        //back propogation
        /*
        MPI_Recv(send_weights, 1, layer_weights,rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); //weights
        pprintf("re")
        MPI_Recv(send_error, 1, layer_error,rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE); //error
        /*
        for (i = 0; i <= lower->no_of_neurons; i++)
         {
             error = 0.0;
             for (j = 0; j < upper->no_of_neurons; j++)
             {
                error += upper->neuron[nu].weight[nl] * upper->neuron[nu].error;
             }
               output = lower->neuron[nl].output;
               lower->neuron[nl].error = output * (1.0 - output) * error;
          }
          */

        //MPI_Send(send_weights,1, layer_weights,rank-1,0,MPI_COMM_WORLD); //weights
        //MPI_Send(send_error, 1, layer_error,rank-1,0,MPI_COMM_WORLD); //error




      }
     epochs++;
   }
   if(rank == np-1)
   {
        int i;
        for(i=0;i< local_layer->no_of_neurons;i++)
        {
            output[i] = local_layer->neuron[i].output;
        }

   }
}
