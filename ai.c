
#include<stdlib.h>
#include<stdio.h>

#include <time.h>
#include <string.h>

#include "ai.h"
#include "linalg.h"

#define FRAND_DIV ((float)RAND_MAX + 0.02)

neural_network_t* new_neural_network(int64_t input_size, int64_t num_layers,
	int64_t *neuron_dims, activation_t activation, float learning_rate, unsigned* seed)
{
	float **weights;

	rand_weights(&weights, num_layers, input_size, neuron_dims, seed);

	neural_network_t *network = (neural_network_t*)malloc(sizeof(neural_network_t));
	if (network == NULL)
	{
		return NULL;
	}

	network->input_size = input_size;
	network->input = NULL;
	network->num_layers = num_layers;
	network->neuron_dims = neuron_dims;
	network->weights = weights;
	network->activation = activation;
	network->learning_rate = learning_rate;
	network->tracking = NULL;

	return network;
}

void free_neural_network(neural_network_t *network)
{
	free(network->input);
	free_tracking(network->tracking);

	for (int64_t i = 0; i < network->num_layers; i++)
	{
		free(network->weights[i]);
	}
	free(network->weights);
	free(network);
}

void set_input(neural_network_t *network, float *input)
{
	if (network->input != NULL)
	{
		free(network->input);
	}
	size_t input_byte_size = sizeof(float) * network->input_size;
	network->input = (float *)malloc(input_byte_size);

	memcpy(network->input, input, input_byte_size);
}

void update_weight(neural_network_t *network, float** deltaweights)
{
	int64_t prev_dimensions = network->input_size + 1;
	int64_t next_dimensions;
	int64_t weight_size;

	for (int64_t i = 0; i < network->num_layers; i++)
	{
		next_dimensions = network->neuron_dims[i];
		weight_size = prev_dimensions * next_dimensions;

		vector_vector_sum(network->weights[i], deltaweights[i],
			network->weights[i], weight_size);

		prev_dimensions = next_dimensions;
	}

}

void rand_weights(float ***weight_ptr, int64_t no_layers, int64_t no_inputs, int64_t *nd, unsigned *seed)
{
  frand(seed);

	*weight_ptr = (float **)malloc(sizeof(float*) * no_layers);
	if (*weight_ptr == NULL)
	{
		exit(-6);
	}
	float **weights = *weight_ptr;

	int64_t prev_dims = no_inputs + 1;
	int64_t weight_size;
	for (int64_t i = 0; i < no_layers; i++)
	{
		weight_size = nd[i] * prev_dims;
		*(weights + i) = (float *)malloc(sizeof(float) * weight_size);
		if (*(weights + i) == NULL)
		{
			exit(-5);
		}

		for (int64_t j = 0; j < weight_size; j++)
		{
			*(*(weights + i) + j) = frand(NULL);
		}
		prev_dims = nd[i];
	}
}

void feedforward(neural_network_t *network)
{
	if (network->tracking != NULL)
	{
		free_tracking(network->tracking);
	}

  // initialize network tracking structure
	network->tracking = (feedforward_tracking_t *)malloc(sizeof(feedforward_tracking_t));

	network->tracking->total_layer_count = network->num_layers + 1;

	network->tracking->layers = (float **)malloc(sizeof(float*) * network->tracking->total_layer_count);

	network->tracking->pre_activation_layers = (float **)malloc(sizeof(float*) * network->tracking->total_layer_count);

  // set up the input layer as prev layer and add the bias term
	int64_t prev_layer_size = (network->input_size + 1);
	float *prev_layer = (float *)malloc(sizeof(float) * prev_layer_size);

	*prev_layer = 1.f; //The bias term
	memcpy(prev_layer + 1, network->input, (sizeof(float) * network->input_size));

	network->tracking->layers[0] = prev_layer;
	network->tracking->pre_activation_layers[0] = prev_layer;

	float *thislayer;
	float *bthislayer;

  // feedforward through layers loop
	for (int64_t i = 0; i < network->num_layers; i++)
	{
		int64_t num_neurons = network->neuron_dims[i];

		thislayer = (float *)malloc(sizeof(float) * num_neurons);

		//Must be zero'd for the matrix multiplication
		bthislayer = (float *)calloc(num_neurons, sizeof(float));

		//Weight Multiplication
		matrix_multiply(prev_layer, network->weights[i], bthislayer, 1, prev_layer_size, num_neurons);

		for (int64_t j = 0; j < num_neurons; j++)
		{
			thislayer[j] = network->activation(bthislayer[j]);
		}
		prev_layer = thislayer;
		*(network->tracking->layers + i + 1) = thislayer;
		*(network->tracking->pre_activation_layers + i + 1) = bthislayer;
		prev_layer_size = num_neurons;
	}
}

void backpropagation(float ***delta_weights, float *desired, neural_network_t *network)
{
	copy_weights(delta_weights, NULL, network->num_layers,
		network->input_size, network->neuron_dims);

	// Output neurons
	float *output = network->tracking->layers[network->num_layers];

	int64_t curr_neuron_count = network->neuron_dims[network->num_layers - 1];

	float *delta = (float *)malloc(sizeof(float) * curr_neuron_count);


	for (int64_t i = 0; i < curr_neuron_count; i++)
	{
//		*(delta + i) = network->delta_activation(output[i]) * (desired[i] - output[i]);
		*(delta + i) = 2 * output[i] * (1 - output[i]) * (desired[i] - output[i]);
	}

	// The neuron count for the next layer, which is actually
	// the previous layer in the network, as we are backtracking.
	int64_t back_neuron_count;
	int64_t weight_size;

	for (int64_t j = network->num_layers - 1; j >= 0; j--)
	{
		curr_neuron_count = network->neuron_dims[j];

		back_neuron_count = j == 0 ? network->input_size + 1 : network->neuron_dims[j - 1];

		weight_size = (back_neuron_count * curr_neuron_count);

		broadcast_vectors(network->tracking->layers[j], delta, *(*delta_weights + j),
			back_neuron_count, curr_neuron_count, mult());

		scalar_vector_mult(network->learning_rate, *(*delta_weights + j),
			*(*delta_weights + j), weight_size);

		if (j != 0)
		{
			float *next_delta = (float*)malloc(sizeof(float) * back_neuron_count);

			//delta=aconst*curlayers[l]*(1-curlayers[l])*(np.dot(delta,weights[l]));

			for (int64_t k = 0; k < back_neuron_count; k++)
			{
				float clayer = *(network->tracking->layers[j] + k);
				//*(next_delta + k) = network->delta_activation(*(tracking->layers[j] + k));
				*(next_delta + k) = 2 * clayer * (1 - clayer);
			}
			float *delta_update = (float *)calloc(back_neuron_count, sizeof(float));

			matrix_multiply(network->weights[j], delta, delta_update,
				back_neuron_count, curr_neuron_count, 1);

			free(delta);
			delta = (float *)malloc(sizeof(float) * back_neuron_count);

			vector_vector_mult(next_delta, delta_update, delta, back_neuron_count);

			free(next_delta);
			free(delta_update);
		}
	}

	free(delta);
}

void train_network(float **desired, float **batch, int64_t batch_size,
	int64_t epochs, neural_network_t *network, FILE *deltaFile, FILE *errorFile)
{
  float **output = malloc(sizeof(float *));
  *output = (float*)malloc(sizeof(float) * network->neuron_dims[network->num_layers - 1]);

  float **delta_weights = NULL;

	for (int i = 0; i < epochs; i++)
	{
		for (int64_t j = 0; j < batch_size; j++)
		{
			set_input(network, batch[j]);

      float *desired_output = desired[j];

		  feedforward(network);

			get_network_output(network, output);

      float error;
      if (desired[j][0] > **output) {
        error = desired[j][0] - **output;
      } else {
        error =  **output - desired[j][0];
      }
      fprintf(errorFile, "%ld\t %f\n", i * batch_size + j, error);

      backpropagation(&delta_weights, desired_output, network);

      fprintf(deltaFile, "%ld\t %f\n", i * batch_size + j, *delta_weights[1]);

			update_weight(network, delta_weights);

			for (int64_t k = 0; k < network->num_layers; k++)
			{
				free(delta_weights[k]);
			}
			free(delta_weights);
			delta_weights = NULL;

		}
	}
  free(*output);
  free(output);


}

void get_network_output(neural_network_t *network, float **output)
{
	if (network->tracking == NULL)
	{
		feedforward(network);
	}

	int64_t output_size = network->neuron_dims[network->num_layers - 1];

	if (*output == NULL)
	{
		return;
	}

	//network->num_layers is always 1 less than tracking->total_layer_count
	memcpy(*output, network->tracking->layers[network->num_layers], output_size * sizeof(float));
}

void free_tracking(feedforward_tracking_t *tracking)
{
	for (int64_t i = 0; i < tracking->total_layer_count; i++)
	{

		free(tracking->layers[i]);
		if (i > 0)
		{
			free(tracking->pre_activation_layers[i]);
		}
	}

	free(tracking->layers);
	free(tracking->pre_activation_layers);
	free(tracking);
}

int copy_weights(float ***dst, float**src, int64_t num_layers, int64_t input_size, int64_t *neuron_dims)
{
	*dst = (float**)malloc(sizeof(float*) * num_layers);

	if (*dst == NULL)
	{
		return -2;
	}

	int64_t prev_dimensions = input_size + 1;
	int64_t next_dimensions;
	int64_t weight_size;

	for (int64_t i = 0; i < num_layers; i++)
	{
		next_dimensions = neuron_dims[i];
		weight_size = prev_dimensions * next_dimensions;

		*(*dst + i) = (float *)malloc(sizeof(float) * weight_size);

		if (*(*dst + i) == NULL)
		{
			return -1;
		}
		if (src != NULL)
		{
			memcpy(*(*dst + i), *(src + i), (sizeof(float) * weight_size));
		}
		prev_dimensions = next_dimensions;
	}
	return 0;
}

float frand(unsigned int *seed)
{
  if(seed != NULL) {
    srand(*seed);
  }

	return rand() / FRAND_DIV;
}

