# Neural Networks from Scratch in C

Two small, self-contained neural networks implemented in plain C, with no
external ML libraries — just `stdio`, `stdlib`, and `math`. Each is a single
file so the entire data flow (forward pass, loss, backpropagation, weight
updates) can be read top to bottom without jumping between headers.

This started as an attempt to fix an old, broken "RNN" project. In the
process it became clear the original code wasn't actually recurrent at all
(no memory between time steps), so it's been rebuilt here as two separate,
correct, and honestly-labeled pieces:

- **`nn.c`** — a real multi-layer feedforward neural network (MLP)
- **`rnn.c`** — a real recurrent neural network (RNN), with actual memory
  across time steps

## `nn.c` — Feedforward Neural Network

A configurable multi-layer perceptron trained with backpropagation and
gradient descent.

- Sigmoid activation on every layer
- Mean squared error loss
- Architecture set by a single array (`layer_sizes`), so adding layers or
  neurons doesn't require touching the rest of the code
- Demo task: **XOR** — chosen because a single neuron can never learn it,
  so successfully solving it proves the hidden layer and backpropagation
  are actually working

### Usage

```bash
gcc -o nn nn.c -lm

# Train on the built-in XOR example:
./nn

# Train on your own data instead:
./nn data.txt
```

### Data format

Each line of the dataset file holds the input features followed by the
target output(s), separated by spaces:

```
0 0 0
0 1 1
1 0 1
1 1 0
```

The number of input columns must match `layer_sizes[0]` and the number of
output columns must match the last entry in `layer_sizes`, both set near
the top of `nn.c`.

## `rnn.c` — Recurrent Neural Network

An Elman-style RNN with a hidden state that carries information from one
time step to the next, trained with **backpropagation through time (BPTT)**.

- `tanh` activation for the hidden state, `softmax` for the output
- Cross-entropy loss
- Demo task: predict the next character in **"hello"**, then generate the
  word from scratch starting with just `'h'`

This example is chosen deliberately: after the *first* `l` the next letter
is `l`, but after the *second* `l` the next letter is `o`. A model with no
memory only ever sees "current letter = l" and can't tell these two
situations apart — only a model that remembers what came before can get
it right. Successfully regenerating `hello` from a single seed letter is
proof the recurrence is actually carrying useful memory forward.

### Usage

```bash
gcc -o rnn rnn.c -lm
./rnn
```



