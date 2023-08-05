
typedef float (*activation_t)(float);

typedef struct neural_network neural_network_t;

typedef struct feedforward_tracking
{
    /** @c layers: A matrix of the layers through the network. */
    float **layers;
    /**
     * @c pre_act_layers: A matrix of the layers, before the activation
     * is applied, through the network.
     */
    float **pre_activation_layers;

    /** @c total_layer_count: The number of layers (including the
     * input layer). Used primarily to help with freeing memory.
     */
    int64_t total_layer_count;

} feedforward_tracking_t;


typedef struct neural_network
{
    // Number of neurons in input layer
    int64_t input_size;

    // Pointer to the inputs of the network
    float *input;

    // The number of layers in the network, excluding the input layer
    int64_t num_layers;

    // an array describing the dimensions of the layers, eg [2, 1] for a hidden layer of 2 neurons and output layer of 1
    int64_t *neuron_dims;

    // a 2d array of weights between layers
    float **weights;

    // pointer to an activation function to be used e.g sigmoid
    activation_t activation;

    // learning rate of the network
    float learning_rate;

    // tracking struct to help with tracking layers throughout feedforward and backtracking
    feedforward_tracking_t *tracking;


} neural_network_t;

neural_network_t *new_neural_network(int64_t input_size, int64_t num_layers,
	int64_t *neuron_dims, activation_t activation, float learning_rate, unsigned* seed);

void set_input(neural_network_t *network, float *input);

void update_weight(neural_network_t *network, float **deltaweights);

void get_network_output(neural_network_t *network, float **output);

void feedforward(neural_network_t *network);

void backpropagation(float ***delta_weights, float *desired, neural_network_t *network);

void train_network(float **desired, float **batch, int64_t batch_size, int64_t epochs, neural_network_t *network, FILE *deltaFile, FILE *errorFile);

void free_neural_network(neural_network_t *network);

void rand_weights(float ***weight_ptr, int64_t no_layers, int64_t no_inputs, int64_t *nd, unsigned *seed);

//TODO: Hide this later
int copy_weights(float ***dst, float **src, int64_t num_layers, int64_t input_dim, int64_t *neuron_dims);

float frand(unsigned int *seed);

void free_tracking(feedforward_tracking_t *tracking);
