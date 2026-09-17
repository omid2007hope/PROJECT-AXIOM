#ifndef include_perceptron
#define include_perceptron

/*
1. The Context Window (How the AI "reads")
Right now, you have a massive, continuous sequence of 8.2 million tokens (which
translates to over 33 million floats in your embedding space). An AI cannot read
that entire sequence at once—it needs a field of view. This is what we call the
"Context Window."

Imagine reading a book through a magnifying glass that only lets you see 128
words at a time.

You look at words 1 through 128 (your input, $x$).
You try to guess word 129 (your target, $y$).
Then, you slide the magnifying glass over by one word. You look at words 2
through 129, and try to guess word 130. Because each of your tokens is
represented by 4 floats (your embedding_dim), a context window of 128 tokens
means the Perceptron is looking at $128 \times 4 = 512$ floats at any given
time.

So, our data loader's job is simply to march down your massive array, slicing it
into chunks of 512 floats, and feeding them to the Perceptron one by one.

2. The Forward Pass (How the AI "thinks")
Once the Perceptron receives those 512 floats, it has to make a prediction. This
is where the linear algebra comes in. The "Forward Pass" is the core equation of
all deep learning:

$$y = \text{ReLU}(W \cdot x + b)$$

Let's demystify what these letters mean:

$x$ (The Input): The 512 floats from your context window.
$W$ (The Weights): These are the "synapses" of the brain. For every single one
of the 512 inputs, the Perceptron has a specific multiplier (a weight) that
determines how important that input is. If the network needs to output 4 floats
(predicting the next token's embedding), the weight matrix $W$ will have $4
\times 512 = 2,048$ individual weights. The network multiplies the inputs by the
weights and adds them all up (the dot product). $b$ (The Bias): Sometimes, the
math needs a baseline adjustment—an offset. The bias is just a number added to
the result of the weight multiplication to shift the output up or down. ReLU
(The Activation Function): If a neural network only did multiplication and
addition ($W \cdot x + b$), it would just be drawing straight lines. It could
never learn complex, curvy patterns like language or images. We fix this by
wrapping the result in an activation function. ReLU (Rectified Linear Unit) is
the most popular one, and it is laughably simple: If the number is negative,
make it $0$. If the number is positive, leave it alone. By simply throwing away
negative numbers, ReLU injects "non-linearity" into the math, which
mathematically allows the neural network to approximate literally any pattern in
the universe.
*/

#include <cstddef>
#include <string>
#include <vector>

struct Key_Concepts {
  double weight = 0.0;        // Weight
  double biases = 0.0;        // Bias
  double forward_pass = 0.0;  // Guess
  double loss_function = 0.0; // Calculate how wrong the guess was
};

class Perceptron {
public:
  double weight;
  double biases;
  double forward_pass;
  double loss_function;

  Perceptron(Key_Concepts &keys);
};

#endif // include_perceptron
