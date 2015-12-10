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
#include <mpe.h>

#define MAX_LAYERS 10

int ReadFile(char *file_name, int valuesPerLine, int numLines, double* arr){
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
void SendInputs(double *input, int trainingInputsCnt, int sendCnt, int worldSize, int tag)
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
				MPI_DOUBLE,
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
  double input[8];
  double target[4];
  double output[4];
  double error;


  MPI_Init(&argc, &argv);
	MPE_Init_log();
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  //MPI Derived data type
  MPI_Datatype global_weights;
  MPI_Type_contiguous(27,MPI_DOUBLE,&global_weights);
  MPI_Type_commit(&global_weights);

  // Initialize the pretty printer
  init_pprintf( rank );
  pp_set_banner( "main" );

  int num_inputs = 2;
  int num_outputs = 1;

	int a_start = MPE_Log_get_event_number();
  int a_end = MPE_Log_get_event_number();
  int b_start = MPE_Log_get_event_number();
  int b_end = MPE_Log_get_event_number();
  int c_start = MPE_Log_get_event_number();
  int c_end = MPE_Log_get_event_number();
  int d_start = MPE_Log_get_event_number();
  int d_end = MPE_Log_get_event_number();
  int e_start = MPE_Log_get_event_number();
  int e_end = MPE_Log_get_event_number();

if (rank == 0)
{
	MPE_Describe_stage(a_start,a_end,"a","red");
	MPE_Describe_stage(b_start,b_end,"b","green");
	MPE_Describe_stage(c_start,c_end,"c", "gray");
	MPE_Describe_stage(d_start,d_end,"d","blue");
	MPE_Describe_stage(e_start,e_end,"e","yellow");
}

// file handling stuff
  double* trainingSamples;
  double* trainingTargets;
  int numTrainingSamples, numTestSamples;

  trainingSamples = (double *) calloc(num_inputs * 2, sizeof(double));
	trainingTargets = (double *) calloc(num_outputs * 1, sizeof(double));
  char* trainingFile, * trainingTargetFile, * testingFile;

  #define inputs(i) (trainingSamples + i * num_inputs)
  #define targets(i) (trainingTargets + i* num_outputs)



  trainingFile = "xor.txt";
  trainingTargetFile = "xor_targets.txt";



  if(rank == 0)
  {

	 MPE_Log_event(a_start,0,"start a");
    ReadFile(trainingFile, num_inputs, 4, trainingSamples);
    ReadFile(trainingTargetFile, num_outputs, 4, trainingTargets);

    SendInputs(&trainingSamples[0], 4, num_inputs, np, 11);

    SendInputs(&trainingTargets[0], 4, num_outputs, np, 22);
		MPE_Log_event(a_end,0,"end a");
  }
  else
  {
    numTrainingSamples = 1;
		MPE_Log_event(a_start,0,"start a");
    MPI_Recv(&trainingSamples[0],(numTrainingSamples * num_inputs),MPI_DOUBLE,0,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

    MPI_Recv(&trainingTargets[0],(numTrainingSamples * num_outputs),MPI_DOUBLE,0,22,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		MPE_Log_event(a_end,0,"end a");
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

	 //net_print(global_net);
   double* weights = (double *) malloc((global_net->no_of_layers) * global_net->layer[1].no_of_neurons * global_net->layer[1].no_of_neurons * sizeof(double));

   int i,j,k;

   for(i = 1 ;i< global_net->no_of_layers;i++)
       for(j=0;j< global_net->layer[i].no_of_neurons;j++)
           for(k=0;k <= global_net->layer[i-1].no_of_neurons;k++)
              {
								weights[getIndex3d(i,j,k,3,3)] = global_net->layer[i].neuron[j].weight[k];
							//	pprintf("%f \n",global_net->layer[i].neuron[j].weight[k]);
							}
	 //pprintf("initial net\n");
	 //net_print(global_net);
   int global_epoch = 0;
   int l;
	 MPE_Log_event(b_start,0,"start b");
	 MPI_Bcast(weights,1,global_weights,0,MPI_COMM_WORLD);
   //pprintf("initial broadcast done\n");
   MPI_Request reqs[np-1];
   double start_time, end_time;
   start_time = MPI_Wtime();
	 int count = 0;
	 while(global_epoch <= 100 )
   {
		 MPE_Log_event(d_start,0,"syncing");
	  for(l=1;l<np;l++)
    {
   double* temp_weights = (double *) malloc((global_net->no_of_layers) * global_net->layer[1].no_of_neurons * global_net->layer[1].no_of_neurons * sizeof(double));

    MPI_Recv(temp_weights,1, global_weights,l,0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		count = 0;

		for(i = 1 ;i< global_net->no_of_layers;i++)
		{
			for(j=0;j< global_net->layer[i].no_of_neurons;j++)
			{
				for(k=0;k <= global_net->layer[i-1].no_of_neurons;k++)
				{
					count++;
						global_net->layer[i].neuron[j].weight[k] += temp_weights[getIndex3d(i,j,k,3,3)];
						//pprintf("\n %f *\t",global_net->layer[i].neuron[j].weight[k]);
				}
			}
		}
	}
	MPE_Log_event(d_end,0,"syncing end");
MPE_Log_event(c_start,0,"computation");
	for(i = 1 ;i< global_net->no_of_layers;i++)
	{
		for(j=0;j< global_net->layer[i].no_of_neurons;j++)
		{
			for(k=0;k <= global_net->layer[i-1].no_of_neurons;k++)
			{
					global_net->layer[i].neuron[j].weight[k] /= count;
			}
		}
	}
	MPE_Log_event(d_start,0,"computation end");
   // MPI_Waitall(np-1,reqs,MPI_STATUS_IGNORE);
    //figure out how to add different weights
    //pprintf("waiting done for epoch %d\n",global_epoch);
		MPE_Log_event(d_start,0,"syncing");
    MPI_Bcast(weights,1,global_weights,0,MPI_COMM_WORLD);
		MPE_Log_event(d_end,0,"syncing end");
    //pprintf("broadcasting new weights\n");
    global_epoch++;
   }
	 end_time = MPI_Wtime();
	 MPE_Log_event(b_end,0,"end b");
	 pprintf("time - taken %f\n",end_time-start_time);
   //net_print(global_net);

   //test data
   input[0] = 0.0; input[1] = 0.0;

   net_compute(global_net,inputs(i),output);

   for(j=0;j<1;j++)
   {
   //printf("rank[%d] -  output[%d] - %f\n",rank, j,output[j]);
   }

  }
  else
  {

    int layers[3];
    layers[0] = 2;
    layers[1] = 3;
    layers[2] = 1;


    double total_error = 0;
    int epoch = 0;
    network_t *local_net;
    local_net = net_allocate_l(3,layers);
   double* local_weights = (double *) malloc((local_net->no_of_layers) * local_net->layer[1].no_of_neurons * local_net->layer[1].no_of_neurons * sizeof(double));
    int i;
    int j;
    int k;
    int l, nu, nl;

    int error;
		MPE_Log_event(b_start,0,"training start");
    MPI_Bcast(local_weights,1,global_weights,0,MPI_COMM_WORLD);
    //pprintf("initial broadcast received- \n ");
      for(i = 1 ;i< local_net->no_of_layers;i++)
       for(j=0;j< local_net->layer[i].no_of_neurons;j++)
           for(k=0;k <= local_net->layer[i-1].no_of_neurons;k++)
              local_net->layer[i].neuron[j].weight[k] = local_weights[getIndex3d(i,j,k,3,3)];

    i = rank-1; //select input
    while((epoch <= 100))
    {
        //sync
        //pprintf("starting training\n");
        //net_print(local_net);
				MPE_Log_event(c_start,0,"computation");
        net_compute(local_net,inputs(i),output);
        for(j=0;j<4;j++)
        {
        //printf("rank[%d] -  output[%d] - %f\n",rank, j,output[j]);
        }
        error = net_compute_output_error(local_net, targets(i));
        net_train(local_net);

				if (epoch == 0)
				{
					total_error = error;
				}
				else
				{
					total_error = 0.9 * total_error + 0.1 * error;
				}
			  //pprintf("epoch - %i \n",epoch);
				MPE_Log_event(c_end,0,"computation end");
        //pprintf("sending from rank %d\n",rank);
				for(i = 1 ;i< local_net->no_of_layers;i++)
				 for(j=0;j< local_net->layer[i].no_of_neurons;j++)
						 for(k=0;k <= local_net->layer[i-1].no_of_neurons;k++)
								{
									local_weights[getIndex3d(i,j,k,3,3)]=local_net->layer[i].neuron[j].weight[k];
									//pprintf("local_weights after training %f \n",local_net->layer[i].neuron[j].weight[k]);
								}
								MPE_Log_event(d_start,0,"syncing");
        MPI_Send(local_weights,1, global_weights,0,0, MPI_COMM_WORLD);//write custom mpi reduce

				MPI_Bcast(local_weights,1,global_weights,0,MPI_COMM_WORLD);
        for(i = 1 ;i< local_net->no_of_layers;i++)
            for(j=0;j< local_net->layer[i].no_of_neurons;j++)
                for(k=0;k <= local_net->layer[i-1].no_of_neurons;k++)
                    {
											local_net->layer[i].neuron[j].weight[k] = local_weights[getIndex3d(i,j,k,3,3)];
											//pprintf("local_weights after sync %f \n",local_net->layer[i].neuron[j].weight[k]);
										}
											MPE_Log_event(d_end,0,"syncing end");

        epoch ++;

    }
		MPE_Log_event(b_end,0,"training end");
 }
  //net_free(local_net);
  MPI_Finalize();
	return 0;
}
