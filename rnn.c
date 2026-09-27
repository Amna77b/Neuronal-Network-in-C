/*
 * Simple Recurrent Neural Network (RNN) in C
 * -------------------------------------------
 * This is a REAL RNN: it keeps a hidden state that carries information
 * from one time step to the next (recurrence), unlike the feedforward
 * network in nn.c which has no memory of past inputs.
 *
 * Demo: predict the next character in the word "hello".
 *   input sequence:  h e l l
 *   target sequence: e l l o
 *
 * This example is chosen on purpose: after the FIRST 'l', the next
 * letter is 'l', but after the SECOND 'l', the next letter is 'o'.
 * A model with no memory only ever sees "current letter = l" and can't
 * tell those two situations apart. Only a model that remembers what
 * came before (a hidden state) can get this right -- which is exactly
 * what a recurrent network provides.
 *
 * How it works:
 *   h_t = sigmoid(Wxh * x_t + Whh * h_(t-1) + bh)   <- hidden state, remembers the past
 *   y_t = softmax(Why * h_t + by)                    <- prediction at time t
 *
 * Training uses backpropagation through time (BPTT): the network is run
 * forward across the whole sequence first (storing every hidden state),
 * then errors are propagated backwards through time, accumulating
 * gradients that include the effect of the recurrent connection (Whh).
 *
 * Compile:  gcc -o rnn rnn.c -lm
 * Run:      ./rnn
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---------------- Config ---------------- */
#define HIDDEN_SIZE   8
#define LEARNING_RATE 0.1
#define EPOCHS        2000

/* The training sequence. Change this word to experiment -- the code
   below automatically builds the vocabulary from whatever text you give it. */
const char *TRAINING_TEXT = "hello";

#define MAX_VOCAB 32
#define MAX_SEQ   64

/* ---------------- Vocabulary ---------------- */
char vocab[MAX_VOCAB];   // unique characters seen, in first-seen order
int vocab_size = 0;

/* Return the vocabulary index of character c, adding it if new. */
int char_to_index(char c) {
    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i] == c) return i;
    }
    vocab[vocab_size] = c;
    return vocab_size++;
}

/* ---------------- Network parameters ---------------- */
double Wxh[HIDDEN_SIZE][MAX_VOCAB]; // input  -> hidden
double Whh[HIDDEN_SIZE][HIDDEN_SIZE]; // hidden(t-1) -> hidden(t)   <-- the recurrence
double Why[MAX_VOCAB][HIDDEN_SIZE]; // hidden -> output
double bh[HIDDEN_SIZE];
double by[MAX_VOCAB];

/* Stored across one full forward pass, needed again during BPTT. */
double h_states[MAX_SEQ + 1][HIDDEN_SIZE]; // h_states[0] is the initial (zero) hidden state
double y_outputs[MAX_SEQ][MAX_VOCAB];      // softmax output at each time step
int x_indices[MAX_SEQ];                    // vocabulary index of the input char at each time step
int seq_len = 0;

double random_weight(void) {
    return ((double)rand() / RAND_MAX) * 0.2 - 0.1; // small random values in [-0.1, 0.1]
}

void init_network(void) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < vocab_size; j++) Wxh[i][j] = random_weight();
        for (int j = 0; j < HIDDEN_SIZE; j++) Whh[i][j] = random_weight();
        bh[i] = 0.0;
    }
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) Why[i][j] = random_weight();
        by[i] = 0.0;
    }
}

/* Turn raw scores into a probability distribution (sums to 1). */
void softmax(double *scores, double *probs, int n) {
    double max_val = scores[0];
    for (int i = 1; i < n; i++) if (scores[i] > max_val) max_val = scores[i];

    double sum_exp = 0.0;
    for (int i = 0; i < n; i++) {
        probs[i] = exp(scores[i] - max_val); // subtract max for numerical stability
        sum_exp += probs[i];
    }
    for (int i = 0; i < n; i++) probs[i] /= sum_exp;
}

/* Run the whole sequence forward, filling h_states[] and y_outputs[].
   h_states[0] is always the zero vector: the network starts with "no memory". */
void forward(void) {
    for (int i = 0; i < HIDDEN_SIZE; i++) h_states[0][i] = 0.0;

    for (int t = 0; t < seq_len; t++) {
        int x = x_indices[t];

        /* hidden state: combines this step's input with the PREVIOUS hidden state */
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double sum = bh[i];
            sum += Wxh[i][x];                 // contribution from the current input
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                sum += Whh[i][j] * h_states[t][j]; // contribution carried over from the past
            }
            h_states[t + 1][i] = tanh(sum);
        }

        /* output: prediction for what comes next, based on the hidden state */
        double scores[MAX_VOCAB];
        for (int i = 0; i < vocab_size; i++) {
            double sum = by[i];
            for (int j = 0; j < HIDDEN_SIZE; j++) sum += Why[i][j] * h_states[t + 1][j];
            scores[i] = sum;
        }
        softmax(scores, y_outputs[t], vocab_size);
    }
}

/* Backpropagation through time: walk the sequence backwards, accumulating
   gradients for every weight, including Whh (the recurrent connection). */
void backward(int *target_indices) {
    double dWxh[HIDDEN_SIZE][MAX_VOCAB] = {0};
    double dWhh[HIDDEN_SIZE][HIDDEN_SIZE] = {0};
    double dWhy[MAX_VOCAB][HIDDEN_SIZE] = {0};
    double dbh[HIDDEN_SIZE] = {0};
    double dby[MAX_VOCAB] = {0};

    /* Gradient of the loss with respect to the hidden state, carried
       backwards from one time step to the previous one. Starts at zero
       because there is no "future" beyond the last time step. */
    double dh_next[HIDDEN_SIZE] = {0};

    for (int t = seq_len - 1; t >= 0; t--) {
        /* --- Output layer gradient ---
           For softmax + cross-entropy, dScore = predicted - target(one-hot). */
        double dscore[MAX_VOCAB];
        for (int i = 0; i < vocab_size; i++) {
            dscore[i] = y_outputs[t][i] - (i == target_indices[t] ? 1.0 : 0.0);
        }

        for (int i = 0; i < vocab_size; i++) {
            dby[i] += dscore[i];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                dWhy[i][j] += dscore[i] * h_states[t + 1][j];
            }
        }

        /* --- Hidden layer gradient ---
           Combines: error coming back from THIS step's output, plus
           error coming back from the NEXT step's hidden state (dh_next) --
           this second part is what makes it "through time". */
        double dh[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double from_output = 0.0;
            for (int k = 0; k < vocab_size; k++) from_output += Why[k][i] * dscore[k];
            dh[i] = from_output + dh_next[i];
        }

        /* tanh derivative: 1 - tanh(x)^2, and h_states already stores tanh(x) */
        double dh_raw[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double h_val = h_states[t + 1][i];
            dh_raw[i] = dh[i] * (1.0 - h_val * h_val);
        }

        int x = x_indices[t];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            dbh[i] += dh_raw[i];
            dWxh[i][x] += dh_raw[i];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                dWhh[i][j] += dh_raw[i] * h_states[t][j];
            }
        }

        /* Pass gradient back to the previous time step through Whh. */
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double sum = 0.0;
            for (int i = 0; i < HIDDEN_SIZE; i++) sum += Whh[i][j] * dh_raw[i];
            dh_next[j] = sum;
        }
    }

    /* Apply the accumulated gradients (plain gradient descent). */
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        bh[i] -= LEARNING_RATE * dbh[i];
        for (int j = 0; j < vocab_size; j++)  Wxh[i][j] -= LEARNING_RATE * dWxh[i][j];
        for (int j = 0; j < HIDDEN_SIZE; j++) Whh[i][j] -= LEARNING_RATE * dWhh[i][j];
    }
    for (int i = 0; i < vocab_size; i++) {
        by[i] -= LEARNING_RATE * dby[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) Why[i][j] -= LEARNING_RATE * dWhy[i][j];
    }
}

double sequence_loss(int *target_indices) {
    double loss = 0.0;
    for (int t = 0; t < seq_len; t++) {
        double p = y_outputs[t][target_indices[t]];
        loss += -log(p > 1e-12 ? p : 1e-12); // cross-entropy
    }
    return loss / seq_len;
}

/* After training: start from the first character and let the network
   generate the rest of the sequence one character at a time, feeding
   each prediction back in as the next input -- this only works if the
   hidden state is actually carrying useful memory forward. */
void generate(char seed_char, int length) {
    double h[HIDDEN_SIZE] = {0};
    int x = char_to_index(seed_char);

    printf("%c", seed_char);
    for (int t = 0; t < length; t++) {
        double h_new[HIDDEN_SIZE];
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            double sum = bh[i] + Wxh[i][x];
            for (int j = 0; j < HIDDEN_SIZE; j++) sum += Whh[i][j] * h[j];
            h_new[i] = tanh(sum);
        }
        memcpy(h, h_new, sizeof(h));

        double scores[MAX_VOCAB], probs[MAX_VOCAB];
        for (int i = 0; i < vocab_size; i++) {
            double sum = by[i];
            for (int j = 0; j < HIDDEN_SIZE; j++) sum += Why[i][j] * h[j];
            scores[i] = sum;
        }
        softmax(scores, probs, vocab_size);

        /* pick the most likely next character */
        int best = 0;
        for (int i = 1; i < vocab_size; i++) if (probs[i] > probs[best]) best = i;

        printf("%c", vocab[best]);
        x = best;
    }
    printf("\n");
}

int main(void) {
    srand((unsigned int)time(NULL));

    int text_len = (int)strlen(TRAINING_TEXT);
    seq_len = text_len - 1; // input is text[0..n-2], target is text[1..n-1]

    int target_indices[MAX_SEQ];
    for (int t = 0; t < seq_len; t++) {
        x_indices[t]       = char_to_index(TRAINING_TEXT[t]);
        target_indices[t]  = char_to_index(TRAINING_TEXT[t + 1]);
    }

    printf("Training text: \"%s\"\n", TRAINING_TEXT);
    printf("Vocabulary (%d chars): ", vocab_size);
    for (int i = 0; i < vocab_size; i++) printf("'%c' ", vocab[i]);
    printf("\n\n");

    init_network();

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        forward();
        double loss = sequence_loss(target_indices);
        backward(target_indices);

        if (epoch % 200 == 0) {
            printf("Epoch %5d | Loss: %.6f\n", epoch, loss);
        }
    }

    printf("\nGenerating from seed '%c': ", TRAINING_TEXT[0]);
    generate(TRAINING_TEXT[0], seq_len);

    return 0;
}
