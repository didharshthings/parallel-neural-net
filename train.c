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
  int i,j;
  int num_neurons[3];
  double time;
  struct timeval start;
  struct timeval end;


  num_neurons[0] = 50;
  num_neurons[1] = 40;
	num_neurons[2] = 20;
	num_neurons[3] = 30;
	num_neurons[4] = 10;
	num_neurons[5] = 1;


  net = net_allocate_l(6,num_neurons);
  //printf("initial net \n");
  //net_print(net);


/*
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
*/
//reading from file
int num_inputs = 50;
int num_outputs = 1;

num_pairs = num_inputs/num_outputs;

// file handling stuff
double* trainingSamples;
double* trainingTargets;
int numTrainingSamples, numTestSamples;

trainingSamples = (double *) calloc(num_inputs * 50, sizeof(double));
trainingTargets = (double *) calloc(num_outputs * 1, sizeof(double));
char* trainingFile, * trainingTargetFile, * testingFile;

#define inputs(i) (trainingSamples + i * num_inputs)
#define targets(i) (trainingTargets + i* num_outputs)



trainingFile = "small_test_set.txt";
trainingTargetFile = "small_test_targets.txt";

ReadFile(trainingFile, num_inputs, 100, trainingSamples);
ReadFile(trainingTargetFile, num_outputs, 100, trainingTargets);

// training
  int epoch = 0;
  double total_error = 0;

  gettimeofday(&start, NULL);
  while((epoch <= 100))
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

 net_print(net);
  net_compute(net,inputs(i),output);

  for(j=0;j<1;j++)
  {
  printf("output - %f\n",output[j]);
  }
  net_free(net);
}

// validation
