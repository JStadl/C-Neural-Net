//
// Created by jws22 on 20/06/23.
//
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ai.h"


float sigmoid(float n) {
  return 1/(1+exp(-n));
}

void intializeTestSet(float **desired, float **batch){

  float (*des1) = malloc(sizeof (float));
  *des1 = 0;
  float (*des2) = malloc(sizeof (float));
  *des2 = 1;
  float (*des3) = malloc(sizeof (float));
  *des3 = 1;
  float (*des4) = malloc(sizeof (float));
  *des4 = 0;
  float (*batch1) = malloc(sizeof (float) * 2);
  batch1[0] = 0;
  batch1[1] = 0;
  float (*batch2) = malloc(sizeof (float) * 2);
  batch2[0] = 0;
  batch2[1] = 1;
  float (*batch3) = malloc(sizeof (float) * 2);
  batch3[0] = 1;
  batch3[1] = 0;
  float (*batch4) = malloc(sizeof (float) * 2);
  batch4[0] = 1;
  batch4[1] = 1;

  desired[0] = des1;
  desired[1] = des2;
  desired[2] = des3;
  desired[3] = des4;

  batch[0] = batch1;
  batch[1] = batch2;
  batch[2] = batch3;
  batch[3] = batch4;
}

void freeTrainingSet(float **desired, float **batch) {
  free(desired[0]);
  free(desired[1]);
  free(desired[2]);
  free(desired[3]);

  free(batch[0]);
  free(batch[1]);
  free(batch[2]);
  free(batch[3]);

}

int main(int argc, char **argv) {

  int64_t neuron_dims[] = {2,1};
  float learningRate = 0.15;
  unsigned int seed = 123456789;
  neural_network_t *xorNetwork = new_neural_network(2,2,neuron_dims,&sigmoid,learningRate, &seed);

  float (**desired) = malloc(sizeof(float *) * 4);
  float (**batch) = malloc(sizeof(float *) * 4);
  intializeTestSet(desired, batch);
  FILE *deltaFile = fopen("delta.txt", "w");
  FILE *errorFile = fopen("error.txt", "w");


  train_network(desired, batch, 4, 5000, xorNetwork, deltaFile, errorFile);

  fclose(deltaFile);
  fclose(errorFile);

  freeTrainingSet(desired, batch);
  free(desired);
  free(batch);
  free_neural_network(xorNetwork);
}

