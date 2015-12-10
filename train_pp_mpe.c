/*
Term Project for CSCI5576
Author - Siddharth Singh
Data Parallelism/
distributing dataset and training networks at each node
*/

#include <getopt.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "nn.h"
#include <mpi.h>
#include "pprintf.h"
#include "mpe.h"
#define MAX_LAYERS 10

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
void SendInputs(float *input, int trainingInputsCnt, int sendCnt, int worldSize, int tag)
{
	int i, numToSend = 0, dest = 0;
	MPI_Request request;

	/* since p 0 has read in all training data skip those samples */
	if(0 < trainingInputsCnt%worldSize)
	{
		i = sendCnt * (trainingInputsCnt/worldSize + 1);
	}
	else
	{
		i = sendCnt * (trainingInputsCnt/worldSize);
	}

	dest = 1;

	/* send each sample to the respective processor */
	while(dest < worldSize)
	{
		if(dest < trainingInputsCnt%worldSize)
		{
			numToSend = trainingInputsCnt/worldSize + 1;
		}
		else
		{
			numToSend = trainingInputsCnt/worldSize;
		}


		/* send training sources */
		MPI_Isend(&input[i],
				sendCnt*numToSend,
				MPI_FLOAT,
				dest,
				tag,
				MPI_COMM_WORLD,
				&request);


		i += sendCnt*numToSend;
		dest++;

	}

	return;
}
int getIndex3d (int i, int j, int k, int dimB, int dimC)
{
        return (i*dimB*dimC + j*dimC + k);
}
int main (int argc, char** argv)
{

  int num_neurons;
  int rank, np;
  float input[8];
  float target[4];
  float output[4];
  float error;


  MPI_Init(&argc, &argv);
  MPE_Init_log();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  int start_a = MPE_Log_get_event_number();
  int end_a = MPE_Log_get_event_number();
  int start_b = MPE_Log_get_event_number();
  int end_b = MPE_Log_get_event_number();
  int start_c = MPE_Log_get_event_number();
  int end_c = MPE_Log_get_event_number();
  int start_d = MPE_Log_get_event_number();
  int end_d = MPE_Log_get_event_number();
  int start_e = MPE_Log_get_event_number();
  int end_e = MPE_Log_get_event_number();


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



  if(rank == 0)
  {
    ReadFile(trainingFile, num_inputs, 4, trainingSamples);
    ReadFile(trainingTargetFile, num_outputs, 4, trainingTargets);

    SendInputs(&trainingSamples[0], 4, num_inputs, np, 11);

    SendInputs(&trainingTargets[0], 4, num_outputs, np, 22);
  }
  else
  {
    numTrainingSamples = 1;
    MPI_Recv(&trainingSamples[0],(numTrainingSamples * num_inputs),MPI_FLOAT,0,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    MPI_Recv(&trainingTargets[0],(numTrainingSamples * num_outputs),MPI_FLOAT,0,22,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    }

//Training
  if(rank == 0)
  {

   network_t *global_net;
   int global_layers[3];
   global_layers[0] = 2;
   global_layers[1] = 3;
   global_layers[2] = 1;
   global_net = net_allocate_l(3,global_layers);


   float* weights = (float *) malloc((global_net->no_of_layers) * global_net->layer[1].no_of_neurons * global_net->layer[1].no_of_neurons * sizeof(float));

   int i,j,k;

   for(i = 1 ;i< global_net->no_of_layers;i++)
       for(j=0;j< global_net->layer[i].no_of_neurons;j++)
           for(k=0;k <= global_net->layer[i-1].no_of_neurons;k++)
              weights[getIndex3d(i,j,k,3,3)] = global_net->layer[i].neuron[j].weight[k];
   //pprintf("global net\n");
   //net_print(global_net);
   int global_epoch = 0;
   int l;
   MPI_Bcast(weights,1,global_weights,0,MPI_COMM_WORLD);
   pprintf("initial broadcast done\n");
   MPI_Request reqs[np-1];
   double start_time, end_time;
   start_time = MPI_Wtime();
   while(global_epoch <= 100)
   {
    for(l=1;l<np;l++)
    {
   float* temp_weights = (float *) malloc((global_net->no_of_layers) * global_net->layer[1].no_of_neurons * global_net->layer[1].no_of_neurons * sizeof(float));
    //pprintf("recieveing from %d \n",l);
    MPI_Recv(temp_weights,1, global_weights,l,0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    for(i = 1 ;i< global_net->no_of_layers;i++)
        for(j=0;j< global_net->layer[i].no_of_neurons;j++)
            for(k=0;k <= global_net->layer[i-1].no_of_neurons;k++)
            {
              if(temp_weights[getIndex3d(i,j,k,3,3)] != 0)
              {
                global_net->layer[i].neuron[j].weight[k] *= temp_weights[getIndex3d(i,j,k,3,3)];
              }
            }
    }
   // MPI_Waitall(np-1,reqs,MPI_STATUS_IGNORE);
    //figure out how to add different weights
    //pprintf("waiting done for epoch %d\n",global_epoch);
    MPI_Bcast(weights,1,global_weights,0,MPI_COMM_WORLD);
    //pprintf("broadcasting new weights\n");
    global_epoch++;
   }
   end_time = MPI_Wtime();
   pprintf("time - taken %f\n",end_time-start_time);
   net_print(global_net);

   //test data
   input[0] = 0.0; input[1] = 0.0;
   //use MPI_TYPE create sub array to split input

   net_compute(global_net,inputs(i),output);

   for(j=0;j<1;j++)
   {
   printf("rank[%d] -  output[%d] - %f\n",rank, j,output[j]);
   }
  }
  else
  {
    input[0] = 1.0; input[1] = 1.0;
    target[0] = 0.0;
    input[2] = 1.0; input[3] = 0.0;
    target[1]= 1.0;
    input[4] = 0.0; input[5] = 1.0;
    target[2] = 1.0;
    input[6] = 0.0; input[7] = 0.0;
    target[3] = 0.0;
    //use MPI_TYPE create sub array to split input

    int layers[3];
    layers[0] = 2;
    layers[1] = 3;
    layers[2] = 1;


    float total_error = 0;
    int epoch = 0;
    network_t *local_net;
    local_net = net_allocate_l(3,layers);
   float* local_weights = (float *) malloc((local_net->no_of_layers) * local_net->layer[1].no_of_neurons * local_net->layer[1].no_of_neurons * sizeof(float));
    int i;
    int j;
    int k;
    int l, nu, nl;

    int error;
    MPI_Bcast(local_weights,1,global_weights,0,MPI_COMM_WORLD);
    //pprintf("initial broadcast received- \n ");
      for(i = 1 ;i< local_net->no_of_layers;i++)
       for(j=0;j< local_net->layer[i].no_of_neurons;j++)
           for(k=0;k <= local_net->layer[i-1].no_of_neurons;k++)
              local_net->layer[i].neuron[j].weight[k] = local_weights[getIndex3d(i,j,k,3,3)];

    i = rank-1; //select input
    while((epoch <= 100) && (total_error >= 0.0))
    {
        //sync
        //pprintf("starting training\n");
        //net_print(local_net);
        net_compute(local_net,inputs(i),output);
        for(j=0;j<4;j++)
        {
        //printf("rank[%d] -  output[%d] - %f\n",rank, j,output[j]);
        }
        error = net_compute_output_error(local_net, targets(i));
        net_train(local_net);
        //pprintf("epoch - %i \n",epoch);

        //pprintf("sending from rank %d\n",rank);
        MPI_Send(local_weights,1, global_weights,0,0, MPI_COMM_WORLD);//write custom mpi reduce
        //pprintf("sending from rank - %d, epoch - %d\n",rank,epoch);
        MPI_Bcast(local_weights,1,global_weights,0,MPI_COMM_WORLD);
        for(i = 1 ;i< local_net->no_of_layers;i++)
            for(j=0;j< local_net->layer[i].no_of_neurons;j++)
                for(k=0;k <= local_net->layer[i-1].no_of_neurons;k++)
                    local_net->layer[i].neuron[j].weight[k] = local_weights[getIndex3d(i,j,k,3,3)];
        epoch ++;

    }

 }
  //net_free(local_net);
  MPI_Finalize();

}