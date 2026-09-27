/*
Term Project for CSCI5576
Author - Siddharth Singh
*/

#include <getopt.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "nn.h"

#define MAX_FILENAME_LENGTH 100
#define MAX_SIZE 10000
#define MAX_LAYERS 10

/* Number of floating point values to read for one sample. Must match the
 * first dimension of the network's input layer. */
#define NUM_INPUTS 50
#define NUM_OUTPUTS 1

/* Read valuesPerLine * numLines doubles from file_name into arr.
 * Returns the number of values actually read, or -1 if the file could not
 * be opened. Callers must check the return value: a short read means the
 * file held fewer samples than requested, and training on the untouched
 * tail of the buffer would use zeros. */
int ReadFile(char *file_name, int valuesPerLine, int numLines, double* arr){
	FILE *ifp;
	int i;
	char *mode = "r";
	int wanted = valuesPerLine * numLines;

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

int main (int argc, char** argv)
{
  network_t *net;
  int num_pairs;
  double input[8];
  double target[4];
  double output[4];
  double error;
  int i;
  int num_neurons[3];
  double time;
  struct timeval start;
  struct timeval end;
  int samples_read;
  int targets_read;

  if (argc < 4) {
    fprintf(stderr, "usage: %s <sample_size> <hidden_neurons> <total_epochs>\n", argv[0]);
    return 1;
  }

int total_epochs;
total_epochs = atoi(argv[3]);
int sample_size;
sample_size = atoi(argv[1]);
int derived_type_size;
derived_type_size = atoi(argv[2]);

if (sample_size <= 0 || derived_type_size <= 0 || total_epochs < 0) {
  fprintf(stderr, "sample_size, hidden_neurons and total_epochs must be positive\n");
  return 1;
}

num_neurons[0] = NUM_INPUTS;
num_neurons[1] = derived_type_size;
num_neurons[2] = 1;


net = net_allocate_l(3,num_neurons);
//printf("initial net \n");
//net_print(net);


//reading from file
int num_inputs = NUM_INPUTS;
int num_outputs = NUM_OUTPUTS;

// file handling stuff
double* trainingSamples;
double* trainingTargets;

trainingSamples = (double *) calloc(num_inputs * sample_size, sizeof(double));
trainingTargets = (double *) calloc(num_outputs * sample_size, sizeof(double));
if (trainingSamples == NULL || trainingTargets == NULL) {
  fprintf(stderr, "out of memory allocating training buffers\n");
  net_free(net);
  return 1;
}
char* trainingFile, * trainingTargetFile;

#define inputs(i) (trainingSamples + i * num_inputs)
#define targets(i) (trainingTargets + i* num_outputs)



trainingFile = "xor.txt";
trainingTargetFile = "xor_targets.txt";


samples_read = ReadFile(trainingFile, num_inputs, sample_size, trainingSamples);
targets_read = ReadFile(trainingTargetFile, num_outputs, sample_size, trainingTargets);

if (samples_read < 0 || targets_read < 0) {
  fprintf(stderr, "could not open training data\n");
  net_free(net);
  free(trainingSamples);
  free(trainingTargets);
  return 1;
}

/* The data files may legitimately hold fewer samples than requested; a
 * partially filled sample must not be trained on, so derive the sample
 * count from what was actually read. */
num_pairs = samples_read / num_inputs;
if (num_pairs > targets_read) {
  num_pairs = targets_read;
}
if (num_pairs < 1) {
  fprintf(stderr, "training data too small: no complete samples found\n");
  net_free(net);
  free(trainingSamples);
  free(trainingTargets);
  return 1;
}
if (num_pairs < sample_size) {
  fprintf(stderr, "warning: requested %d samples, training on %d\n", sample_size, num_pairs);
}

// training
  int epoch = 0;
  double total_error = 0;

  gettimeofday(&start, NULL);
  while((epoch <= total_epochs))
  {
    i = rand () % num_pairs ;
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
  gettimeofday(&end, NULL);

 // calc & print results
 time = calctime(start, end);
printf("%lf(s) \n",time);
  //test data
  input[0] = 0.0; input[1] = 0.0;
  //use MPI_TYPE create sub array to split input
	//net_print(net);

// net_print(net);
  //net_compute(net,inputs(i),output);

  //printf("final rolling training error: %lf\n", total_error);
  //printf("output - %f\n", output[0]);
  net_free(net);
  free(trainingSamples);
  free(trainingTargets);
  return 0;
}

// validation
