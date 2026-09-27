/*
 * Simple Feedforward Neural Network in C
 * ---------------------------------------
 * A minimal, from-scratch multi-layer perceptron (MLP):
 *   - configurable number of layers / neurons per layer
 *   - sigmoid activation everywhere
 *   - backpropagation + gradient descent training
 *
 * Demo: learns XOR. This is chosen on purpose -- a single neuron
 * (like the old project's version) mathematically CANNOT learn XOR,
 * but a network with a hidden layer can. Watching the loss drop to
 * near zero is proof the network is actually working.
 *
 * Compile:  gcc -o nn nn.c -lm
 * Run:      ./nn
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ---------------- Config: change the architecture here ---------------- */
#define NUM_LAYERS   3   // total layers including input: input -> hidden -> output
#define MAX_NEURONS  8   // max neurons allowed in any single layer (array sizing)
#define MAX_SAMPLES  1000 // max rows readable from a dataset file
#define LEARNING_RATE 0.5
#define EPOCHS 5000

/* Number of neurons in each layer, including the input layer. */
int layer_sizes[NUM_LAYERS] = {2, 4, 1}; // 2 inputs -> 4 hidden neurons -> 1 output

/* ---------------- Network structure ---------------- */
typedef struct {
    double weights[MAX_NEURONS][MAX_NEURONS]; // weights[i][j]: from neuron j (prev layer) to neuron i (this layer)
    double biases[MAX_NEURONS];
    double outputs[MAX_NEURONS];              // activation output of each neuron (after sigmoid)
    double deltas[MAX_NEURONS];               // error term used during backpropagation
    int num_neurons;                          // neurons in this layer
    int num_inputs;                           // neurons in the previous layer
} Layer;

/* One Layer struct per layer AFTER the input (input has no weights of its own). */
Layer network[NUM_LAYERS - 1];

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/* y is already sigmoid(x) (the neuron's stored output), so the derivative
   simplifies to y*(1-y) -- avoids recomputing exp(). */
double sigmoid_derivative(double y) {
    return y * (1.0 - y);
}

double random_weight(void) {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0; // random value in [-1, 1]
}

/* Give every weight and bias a small random starting value. */
void init_network(void) {
    for (int l = 0; l < NUM_LAYERS - 1; l++) {
        network[l].num_neurons = layer_sizes[l + 1];
        network[l].num_inputs  = layer_sizes[l];

        for (int i = 0; i < network[l].num_neurons; i++) {
            network[l].biases[i] = random_weight();
            for (int j = 0; j < network[l].num_inputs; j++) {
                network[l].weights[i][j] = random_weight();
            }
        }
    }
}

/* Forward pass: push `input` through every layer and return a pointer
   to the final layer's outputs. */
double *forward(double *input) {
    double *prev_output = input;

    for (int l = 0; l < NUM_LAYERS - 1; l++) {
        Layer *layer = &network[l];
        for (int i = 0; i < layer->num_neurons; i++) {
            double sum = layer->biases[i];
            for (int j = 0; j < layer->num_inputs; j++) {
                sum += layer->weights[i][j] * prev_output[j];
            }
            layer->outputs[i] = sigmoid(sum);
        }
        prev_output = layer->outputs;
    }

    return network[NUM_LAYERS - 2].outputs; // outputs of the last layer
}

/* Backpropagation for a single (input, target) example.
   Must be called right after forward() on the same input, since it
   reuses the outputs forward() just computed. */
void backward(double *input, double *target) {
    int last = NUM_LAYERS - 2;

    /* 1. Error at the output layer */
    Layer *out_layer = &network[last];
    for (int i = 0; i < out_layer->num_neurons; i++) {
        double output = out_layer->outputs[i];
        double error = output - target[i];
        out_layer->deltas[i] = error * sigmoid_derivative(output);
    }

    /* 2. Error at each hidden layer, propagated backwards from the next layer */
    for (int l = last - 1; l >= 0; l--) {
        Layer *layer = &network[l];
        Layer *next_layer = &network[l + 1];
        for (int i = 0; i < layer->num_neurons; i++) {
            double error = 0.0;
            for (int k = 0; k < next_layer->num_neurons; k++) {
                error += next_layer->weights[k][i] * next_layer->deltas[k];
            }
            layer->deltas[i] = error * sigmoid_derivative(layer->outputs[i]);
        }
    }

    /* 3. Update every weight and bias using its delta */
    for (int l = 0; l < NUM_LAYERS - 1; l++) {
        Layer *layer = &network[l];
        double *prev_output = (l == 0) ? input : network[l - 1].outputs;

        for (int i = 0; i < layer->num_neurons; i++) {
            for (int j = 0; j < layer->num_inputs; j++) {
                layer->weights[i][j] -= LEARNING_RATE * layer->deltas[i] * prev_output[j];
            }
            layer->biases[i] -= LEARNING_RATE * layer->deltas[i];
        }
    }
}

/* Mean squared error over one example's output vector. */
double compute_loss(double *predicted, double *target, int n) {
    double loss = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = predicted[i] - target[i];
        loss += diff * diff;
    }
    return loss / n;
}

/* Load a dataset from a plain-text file. Each line must hold
   num_inputs feature values followed by num_outputs target values,
   separated by whitespace, e.g. for 2 inputs + 1 output:
       0 0 0
       0 1 1
       1 0 1
       1 1 0
   Returns the number of samples read, or -1 if the file can't be opened.
   inputs/targets must be arrays sized [MAX_SAMPLES][MAX_NEURONS]. */
int load_dataset(const char *filename,
                  double inputs[][MAX_NEURONS], double targets[][MAX_NEURONS],
                  int num_inputs, int num_outputs) {
    FILE *f = fopen(filename, "r");
    if (f == NULL) {
        perror("Could not open dataset file");
        return -1;
    }

    int count = 0;
    while (count < MAX_SAMPLES) {
        int row_ok = 1;

        for (int i = 0; i < num_inputs && row_ok; i++) {
            if (fscanf(f, "%lf", &inputs[count][i]) != 1) row_ok = 0;
        }
        for (int i = 0; i < num_outputs && row_ok; i++) {
            if (fscanf(f, "%lf", &targets[count][i]) != 1) row_ok = 0;
        }

        if (!row_ok) break; // stops cleanly at a short/blank final line or EOF
        count++;
    }

    fclose(f);
    return count;
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    init_network();

    int input_size  = layer_sizes[0];
    int output_size = layer_sizes[NUM_LAYERS - 1];

    static double inputs[MAX_SAMPLES][MAX_NEURONS];
    static double targets[MAX_SAMPLES][MAX_NEURONS];
    int num_samples;

    if (argc >= 2) {
        /* ./nn dataset.txt  -- load real data */
        num_samples = load_dataset(argv[1], inputs, targets, input_size, output_size);
        if (num_samples <= 0) {
            printf("No usable rows found in %s\n", argv[1]);
            return 1;
        }
        printf("Loaded %d samples from %s\n\n", num_samples, argv[1]);
    } else {
        /* No file given -- fall back to the hardcoded XOR demo. */
        double xor_inputs[4][2]  = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double xor_targets[4][1] = {{0}, {1}, {1}, {0}};
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < input_size; j++)  inputs[i][j]  = xor_inputs[i][j];
            for (int j = 0; j < output_size; j++) targets[i][j] = xor_targets[i][j];
        }
        num_samples = 4;
        printf("No dataset file given -- using hardcoded XOR data.\n");
        printf("(Run as \"./nn yourfile.txt\" to train on your own data instead.)\n\n");
    }

    printf("Training...\n\n");
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0.0;
        for (int s = 0; s < num_samples; s++) {
            double *predicted = forward(inputs[s]);
            total_loss += compute_loss(predicted, targets[s], output_size);
            backward(inputs[s], targets[s]);
        }
        if (epoch % 500 == 0) {
            printf("Epoch %5d | Loss: %.6f\n", epoch, total_loss / num_samples);
        }
    }

    printf("\nFinal predictions:\n");
    for (int s = 0; s < num_samples; s++) {
        double *predicted = forward(inputs[s]);

        printf("Input: (");
        for (int j = 0; j < input_size; j++) printf("%s%.2f", j ? ", " : "", inputs[s][j]);
        printf(") -> Predicted: (");
        for (int j = 0; j < output_size; j++) printf("%s%.4f", j ? ", " : "", predicted[j]);
        printf(") | Target: (");
        for (int j = 0; j < output_size; j++) printf("%s%.2f", j ? ", " : "", targets[s][j]);
        printf(")\n");
    }

    return 0;
}
