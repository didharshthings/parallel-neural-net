/*
Term Project for CSCI5576
Author - Siddharth Singh
Data Parallelism/
distributing dataset and training networks at each node
* TODO - training/testing
* TODO - profiling
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

void ReadFile(char *file_name, int valuesPerLine, int numLines, double* arr){
	FILE *ifp;
	int i, j;
	double val;
	char *mode = "r";
	ifp = fopen(file_name, mode);

	if (ifp == NULL) {

	}

	i = 0;
	while((!feof(ifp)) && (i < (valuesPerLine*numLines)))
	{
		fscanf(ifp, "%lf ", &val);

		arr[i] = val;

		i++;
	}

	// closing file
	fclose(ifp);
}
void SendInputs(double *input, int trainingInputsCnt, int sendCnt, int worldSize, int tag)
{
	int i, numToSend = 0, dest = 0;
	MPI_Request request;


	if(0 < trainingInputsCnt%(worldSize-1))
	{
		i = sendCnt * (trainingInputsCnt/(worldSize-1) + 1);
	}
	else
	{
		i = sendCnt * (trainingInputsCnt/(worldSize-1));
	}

	dest = 1;

	i = 0;
	/* send each sample to the respective processor */

	while(dest < worldSize)
	{
		if(dest < trainingInputsCnt%(worldSize-1))
		{
			numToSend = (trainingInputsCnt/(worldSize-1) + 1);
		}
		else
		{
			numToSend = (trainingInputsCnt/(worldSize-1));
		}
		//pprintf(" sending to rank-%d sendCnt-%d num to send %d i - %d \n",dest,sendCnt,numToSend,i);
		/* send training sources */
		MPI_Isend(&input[i],
				sendCnt*numToSend,
				MPI_DOUBLE,
				dest,
				tag,
				MPI_COMM_WORLD,
				&request);


		i += (sendCnt*numToSend);
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
	int event1a, event1b, event2a, event2b,
        event3a, event3b, event4a, event4b,
				event5a, event5b, event6a, event6b,
				event7a, event7b;


  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  int derived_type_size;
	derived_type_size = atoi(argv[2]);
	int total_epochs;
	total_epochs = atoi(argv[3]);
  //MPI Derived data type
  MPI_Datatype global_weights;
	derived_type_size += 50 + 1 +1;
  MPI_Type_contiguous(derived_type_size,MPI_DOUBLE,&global_weights);
  MPI_Type_commit(&global_weights);
	MPE_Init_log();

  // Initialize the pretty printer
  init_pprintf( rank );
  pp_set_banner( "main" );

  int num_inputs = 50;
  int num_outputs = 1;

// file handling stuff
  int sample_size;
	sample_size = atoi(argv[1]);
	int hidden_neurons;
	hidden_neurons = atoi(argv[2]);

  double* trainingSamples;
  double* trainingTargets;
  int numTrainingSamples, numTestSamples;

  trainingSamples = (double *) calloc(num_inputs * sample_size, sizeof(double));
	trainingTargets = (double *) calloc(num_outputs * sample_size, sizeof(double));
  char* trainingFile, * trainingTargetFile, * testingFile;

  #define inputs(i) (trainingSamples + i * num_inputs)
  #define targets(i) (trainingTargets + i* num_outputs)


  trainingFile = "xor.txt";
  trainingTargetFile = "xor_targets.txt";

	event1a = MPE_Log_get_event_number();
	event1b = MPE_Log_get_event_number();
	event2a = MPE_Log_get_event_number();
	event2b = MPE_Log_get_event_number();
	event3a = MPE_Log_get_event_number();
	event3b = MPE_Log_get_event_number();
	event4a = MPE_Log_get_event_number();
	event4b = MPE_Log_get_event_number();
	event5a = MPE_Log_get_event_number();
	event5b = MPE_Log_get_event_number();
	event6a = MPE_Log_get_event_number();
	event6b = MPE_Log_get_event_number();
	event7a = MPE_Log_get_event_number();
	event7b = MPE_Log_get_event_number();

	if (rank == 0)
	{
		MPE_Describe_state(event1a, event1b, "reading inputs", "red");
		MPE_Describe_state(event2a, event2b, "training",   "blue");
		MPE_Describe_state(event3a, event3b, "inital broadcast",    "green");
		MPE_Describe_state(event4a, event4b, "recieving weights",      "yellow");
		MPE_Describe_state(event5a, event5b, "computation",      "gray");
		MPE_Describe_state(event6a, event6b, "sending new weights",      "pink");
		MPE_Describe_state(event7a, event7b, "adjusting new weights",      "white");
	}

  if(rank == 0)
  {
	MPE_Log_event(event1a, 0, "reading initial file");

    ReadFile(trainingFile, num_inputs+50, sample_size, trainingSamples);
    ReadFile(trainingTargetFile, num_outputs+1, sample_size, trainingTargets);

    SendInputs(&trainingSamples[0], sample_size, num_inputs, np, 11);

    SendInputs(&trainingTargets[0], sample_size, num_outputs, np, 22);
			MPE_Log_event(event1b, 0, "reading initial file");

  }
  else
  {

		MPE_Log_event(event1a, 0, "reading initial file");

		numTrainingSamples  = sample_size/(np-1);

		MPI_Recv(&trainingSamples[0],(numTrainingSamples * num_inputs+50),MPI_DOUBLE,0,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		MPI_Recv(&trainingTargets[0],(numTrainingSamples * num_outputs+1),MPI_DOUBLE,0,22,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		//pprintf("recieved training data by rank %d \n",rank);
		MPE_Log_event(event1b, 0, "reading initial file");

	}


//Training
  if(rank == 0)
  {

   network_t *global_net;
   int global_layers[3];
   global_layers[0] = 50;
   global_layers[1] = derived_type_size;
   global_layers[2] = 1;
   global_net = net_allocate_l(3,global_layers);

	 //net_print(global_net);
   double* weights = (double *) calloc((global_net->no_of_layers) * global_net->layer[1].no_of_neurons * global_net->layer[1].no_of_neurons , sizeof(double));

   int i,j,k;

   for(i = 1 ;i< global_net->no_of_layers;i++)
       for(j=0;j< global_net->layer[i].no_of_neurons;j++)
           for(k=0;k <= global_net->layer[i-1].no_of_neurons;k++)
              {
								weights[getIndex3d(i,j,k,4,derived_type_size)] = global_net->layer[i].neuron[j].weight[k];
								//pprintf("%f \n",global_net->layer[i].neuron[j].weight[k]);
							}
	 //pprintf("initial net\n");
	 //net_print(global_net);
   int global_epoch = 0;
   int l;
	 MPE_Log_event(event3a, 0, "initial broadcast");
   MPI_Bcast(weights,1,global_weights,0,MPI_COMM_WORLD);
	 MPE_Log_event(event3b, 0, "initial broadcast");

   //pprintf("initial broadcast done\n");
   MPI_Request reqs[np-1];
   double start_time, end_time;
   start_time = MPI_Wtime();
	 int count = 0;
	 MPE_Log_event(event2a, 0, "training");
	 while(global_epoch <= total_epochs )
   {
    for(l=1;l<np;l++)
    {
   double* temp_weights = (double *) calloc((global_net->no_of_layers) * global_net->layer[1].no_of_neurons * global_net->layer[1].no_of_neurons , sizeof(double));
	 MPE_Log_event(event4a, 0, "recieving weights");
    MPI_Recv(temp_weights,1, global_weights,l,0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
		//pprintf("recieved from rank %d\n",l);
		MPE_Log_event(event4b, 0, "recieving weights");
		count = 0;
		MPE_Log_event(event5a, 0, "computation");
		for(i = 1 ;i< global_net->no_of_layers;i++)
		{
			for(j=0;j< global_net->layer[i].no_of_neurons;j++)
			{
				for(k=0;k <= global_net->layer[i-1].no_of_neurons;k++)
				{
					count++;
						global_net->layer[i].neuron[j].weight[k] += temp_weights[getIndex3d(i,j,k,4,derived_type_size)];
						//pprintf("\n %f *\t",global_net->layer[i].neuron[j].weight[k]);
				}
			}
		}
	}

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
	MPE_Log_event(event5b, 0, "computation");

   // MPI_Waitall(np-1,reqs,MPI_STATUS_IGNORE);
    //figure out how to add different weights
    //pprintf("waiting done for epoch %d\n",global_epoch);
		MPE_Log_event(event6a, 0, "sending new weights");

    MPI_Bcast(weights,1,global_weights,0,MPI_COMM_WORLD);
		MPE_Log_event(event6b, 0, "Sending new weights");

    //pprintf("broadcasting new weights\n");
    global_epoch++;
   }
	 MPE_Log_event(event2b, 0, "training");

   end_time = MPI_Wtime();
   pprintf("time - taken %f\n",end_time-start_time);

	 //validation

   //net_free(global_net);
	 //free(weights);

  }
  else
  {

    int layers[3];
    layers[0] = 50;
    layers[1] = derived_type_size;
    layers[2] = 1;

    double total_error = 0;
    int epoch = 0;
    network_t *local_net;
    local_net = net_allocate_l(3,layers);
   double* local_weights = (double *) calloc((local_net->no_of_layers) * local_net->layer[1].no_of_neurons * local_net->layer[1].no_of_neurons ,sizeof(double));
    int i;
    int j;
    int k;
    int l, nu, nl;

    int error;
		MPE_Log_event(event3a, 0, "initial broadcast");

    MPI_Bcast(local_weights,1,global_weights,0,MPI_COMM_WORLD);
		MPE_Log_event(event3b, 0, "initial broadcast");

	  //pprintf("initial broadcast received- \n ");
      for(i = 1 ;i< local_net->no_of_layers;i++)
       for(j=0;j< local_net->layer[i].no_of_neurons;j++)
           for(k=0;k <= local_net->layer[i-1].no_of_neurons;k++)
              local_net->layer[i].neuron[j].weight[k] = local_weights[getIndex3d(i,j,k,4,derived_type_size)];
							int sample;
							sample = 0;
							MPE_Log_event(event2a, 0, "training");

							while((epoch <= total_epochs))
							{
								//sync
								//pprintf("starting training\n");
								//net_print(local_net);

								//pprintf("%f \n",trainingSamples[sample]);
								MPE_Log_event(event5a, 0, "computation");
								net_compute(local_net,inputs(i),output);

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

								//pprintf("sending from rank %d\n",rank);
								for(i = 1 ;i< local_net->no_of_layers;i++)
								for(j=0;j< local_net->layer[i].no_of_neurons;j++)
								for(k=0;k <= local_net->layer[i-1].no_of_neurons;k++)
								{
									local_weights[getIndex3d(i,j,k,4,3)]=local_net->layer[i].neuron[j].weight[k];
									//pprintf("local_weights after training %f \n",local_net->layer[i].neuron[j].weight[k]);
								}
								MPE_Log_event(event5b, 0, "computation");

								MPE_Log_event(event4a, 0, "sending weights");
								MPI_Send(local_weights,1, global_weights,0,0, MPI_COMM_WORLD);//write custom mpi reduce
								MPE_Log_event(event4b, 0, "sending weights");

								MPE_Log_event(event6a, 0, "recieving new weights");
								MPI_Bcast(local_weights,1,global_weights,0,MPI_COMM_WORLD);
								MPE_Log_event(event6b, 0, "recieving new weights");
								MPE_Log_event(event7a, 0, "setting new  weights");
								for(i = 1 ;i< local_net->no_of_layers;i++)
								for(j=0;j< local_net->layer[i].no_of_neurons;j++)
								for(k=0;k <= local_net->layer[i-1].no_of_neurons;k++)
								{
									local_net->layer[i].neuron[j].weight[k] = local_weights[getIndex3d(i,j,k,4,derived_type_size)];
									//pprintf("local_weights after sync %f \n",local_net->layer[i].neuron[j].weight[k]);
								}
								MPE_Log_event(event7b, 0, "setting new weights");

        epoch ++;
        sample += 50;
    }
		MPE_Log_event(event2b, 0, "training");

		//net_free(local_net);
		//free(local_weights);
	 }

	 //free(trainingSamples);
	 //free(trainingTargets);
	   MPE_Finish_log("cpilog");

    MPI_Finalize();
return 0;
}
