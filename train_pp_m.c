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

/* Read valuesPerLine * numLines doubles from file_name into arr.
 * Returns the number of values read, or -1 if the file could not be opened.
 * Reads with %lf: the samples are doubles, and the previous %d wrote ints
 * through a float*. */
int ReadFile(char *file_name, int valuesPerLine, int numLines, double* arr){
	FILE *ifp;
	int i;
	int wanted = valuesPerLine * numLines;
	char *mode = "r";
	ifp = fopen(file_name, mode);

	if (ifp == NULL) {
		return -1;
	}

	i = 0;
	while((i < wanted) && (fscanf(ifp, "%lf", &arr[i]) == 1))
	{
		i++;
	}

	// closing file
	fclose(ifp);

	return i;
}

int main(int argc, char** argv)
{

    int rank, np;
    double input[8];
    double target[4];
    double output[4];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    /* The per-layer transfer datatypes are created below, once the local
     * layer widths are known. */

    // Initialize the pretty printer
    init_pprintf( rank );
    pp_set_banner( "main" );

    int num_inputs = 2;
    int num_outputs = 1;


  // file handling stuff
    double* trainingSamples;
    double* trainingTargets;

    /* The XOR truth table has 4 samples; ReadFile below reads 4 per buffer. */
    trainingSamples = (double *) calloc(num_inputs * 4, sizeof(double));
    trainingTargets = (double *) calloc(num_outputs * 4, sizeof(double));
    char* trainingFile, * trainingTargetFile;

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
   if (local_layer == NULL) {
     fprintf(stderr, "rank %d: could not allocate layer\n", rank);
     MPI_Abort(MPI_COMM_WORLD, 1);
   }

   if (rank == 0 )
   {
    local_layer->no_of_neurons = 2;

   }
   else if (rank == np - 1)
   {
    local_layer->no_of_neurons = 1;

   }
   else
   {
    local_layer->no_of_neurons = 3;

   }


   //allocate neurons (plus one bias neuron, as nn.c does)
   local_layer->neuron =
       (neuron_t *) calloc(local_layer->no_of_neurons + 1, sizeof(neuron_t));
   if (local_layer->neuron == NULL) {
     fprintf(stderr, "rank %d: could not allocate neurons\n", rank);
     MPI_Abort(MPI_COMM_WORLD, 1);
   }
   /* Wire format for the per-layer transfers.
    *
    * Every rank used to build one datatype from its own neuron count and then
    * use that same type for both send and receive. Neighbouring ranks have
    * different layer widths (2, 3, ..., 1), so the receiver's buffer was
    * always the wrong size and MPI aborted with MPI_ERR_TRUNCATE. Use a single
    * fixed width that fits the widest layer taking part in the exchange. */
   int xfer_count = 0;
   {
     int li;
     for (li = 0; li < 3; li++) {
       if (layers[li] > xfer_count) {
         xfer_count = layers[li];
       }
     }
   }
   if (xfer_count < 1) {
     xfer_count = 1;
   }

   MPI_Datatype layer_output;
   MPI_Datatype layer_input;
   MPI_Datatype layer_error;
   MPI_Datatype layer_weights;

   MPI_Type_contiguous(xfer_count, MPI_DOUBLE, &layer_output);
   MPI_Type_contiguous(xfer_count, MPI_DOUBLE, &layer_input);
   MPI_Type_contiguous(xfer_count, MPI_DOUBLE, &layer_weights);
   MPI_Type_contiguous(xfer_count, MPI_DOUBLE, &layer_error);

   MPI_Type_commit(&layer_output);
   MPI_Type_commit(&layer_input);
   MPI_Type_commit(&layer_weights);
   MPI_Type_commit(&layer_error);

   if(rank == 0)
   {
        //set input
        int i;
        ReadFile(trainingFile, num_inputs, 4, trainingSamples);
        ReadFile(trainingTargetFile, num_outputs, 4, trainingTargets);
        for(i=0;i<local_layer->no_of_neurons;i++)
        {
            local_layer->neuron[i].output = input[i];
        }

  }


   //start training

   int epochs = 0;
   while(epochs<=2)
   {
      pprintf("epoch %d", epochs);

      /* Forward pass, pipelined so that send/receive pairs stay balanced.
       *
       * The original code had each rank send and receive in a different order
       * depending on its position, and used a plain blocking MPI_Send. Rank 0
       * therefore ran ahead and queued all of its sends while rank 1 was still
       * completing the first handshake; once the eager buffers filled, the
       * ranks fell out of step and a rank exited without calling MPI_Finalize.
       * Every rank now receives first and then sends, in the same order. */
      double* recv_buf = (double *) calloc(xfer_count, sizeof(double));
      double* send_buf = (double *) calloc(xfer_count, sizeof(double));
      if (recv_buf == NULL || send_buf == NULL) {
        fprintf(stderr, "rank %d: could not allocate transfer buffers\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      if (rank > 0) {
        MPI_Recv(recv_buf, 1, layer_output, rank - 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      if (rank < np - 1) {
        MPI_Send(send_buf, 1, layer_output, rank + 1, 0, MPI_COMM_WORLD);
      }

      free(recv_buf);
      free(send_buf);

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

   free(local_layer->neuron);
   free(local_layer);
   free(trainingSamples);
   free(trainingTargets);

   MPI_Type_free(&layer_output);
   MPI_Type_free(&layer_input);
   MPI_Type_free(&layer_weights);
   MPI_Type_free(&layer_error);

   MPI_Finalize();
   return 0;
}
