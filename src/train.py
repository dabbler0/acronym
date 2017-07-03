# Math
import theano
import theano.tensor as T
import numpy

# Model
from model import *
from alphabet import Alphabet
from corpus import Corpus

# Plumbing
import pickle
import argparse
import os
import sys

# Increase the recursion limit, which is required for
# gradient compilation
sys.setrecursionlimit(9999)

# Constant random seed
numpy.random.seed(0)

# Command line arguments
parser=argparse.ArgumentParser(description='Run language model training.')

parser.add_argument('--data', dest='data', help='Data file to run on', required=True)
parser.add_argument('--alphabet', dest='alphabet', help='Alphabet file to run with', required=True)
parser.add_argument('--out', dest='out', help='Output directory', required=True)

# Optional model arguments
parser.add_argument('--embedding_size', type=int, dest='embedding_size', help='Word embedding size', default=300)
parser.add_argument('--rnn_size', type=int, dest='rnn_size', help='Hidden state size', default=500)
parser.add_argument('--batch_size', type=int, dest='batch_size', help='Minibatch size', default=50)
parser.add_argument('--seq_length', type=int, dest='seq_length', help='Sequence length', default=50)

# Optional training arguments
parser.add_argument('--epochs', type=int, dest='epochs', help='Number of epochs', default=5000 * 13)
parser.add_argument('--learning_rate', type=float, dest='learning_rate', help='Learning rate for SGD', default=0.05)
parser.add_argument('--checkpoint_freq', type=int, dest='checkpoint_frequency', help='Save a checkpoint every X epochs', default=5000)
parser.add_argument('--sample_length', type=int, dest='sample_length', help='Length of sample to take', default=50)
parser.add_argument('--sample_freq', type=int, dest='sample_frequency', help='Take a sample every X epochs', default=100)
parser.add_argument('--softmax_temp', type=float, dest='softmax_temperature', help='Softmax temperature for sampling', default=1)

arg = parser.parse_args()

# Load data and alphabet
with open(arg.data, 'rb') as f:
    data = pickle.load(f)

with open(arg.alphabet, 'rb') as f:
    alphabet = pickle.load(f)

if not os.path.exists(arg.out):
    os.makedirs(arg.out)

# Unpack arguments
epochs = arg.epochs
embedding_size = arg.embedding_size
rnn_size = arg.rnn_size
batch_size = arg.batch_size
seq_length = arg.seq_length
learning_rate = arg.learning_rate

# Create corpus
corpus = Corpus(data, seq_length, batch_size)

# Create the model
embedding_layer = Embedding(alphabet.size, embedding_size)
gru_layer_1 = GRU(embedding_size, rnn_size, rnn_size)
gru_layer_2 = GRU(rnn_size, rnn_size, rnn_size)
output_layer = Output(rnn_size, alphabet.size)

forward_network = Composition([embedding_layer, gru_layer_1, gru_layer_2, output_layer])

# Create training-mode model
true_output = T.imatrix('y') # Will ultimately be seq_length x batch_size
train_layer = output_layer.create_training_node(batch_size, true_output)
training_network = Composition([embedding_layer, gru_layer_1, gru_layer_2, train_layer])

# Compute gradients
initial_hidden = tuple(T.matrix('h') for _ in range(training_network.n_hiddens))  # batch_size x rrn_size
inputs = T.imatrix('x') # seq_length x batch_size
new_hidden, costs = training_network.unroll(seq_length, initial_hidden, inputs)
cost = T.mean(costs)

# Training function, which also updates parameters
# according to SGD (TODO adam/nesterov momentum)
print('Compiling training function...')
training_function = theano.function(
    (inputs, true_output) + initial_hidden,
    (cost,) + new_hidden,
    updates = [
        (
            param,
            param - learning_rate * T.grad(cost, param)
        ) for param in training_network.params
    ],
    mode='FAST_RUN'
)
print('Done.')

# Training function
def train(inputs, outputs, hiddens):
    args = (inputs, outputs) + hiddens
    result = training_function(*args)

    # Loss, new hidden state
    return result[0], result[1:]

# Sampling function
print('Compiling sampling function...')
singleton = forward_network.create_singleton_function()
print('Done.')

# Initialize the hidden state at the beginning of all the samples.
current_hiddens = (
    numpy.zeros((batch_size, rnn_size)), # Layer 1
    numpy.zeros((batch_size, rnn_size)) # Layer 2
)
smooth_cost = None

# Training loop
print('Beginning training loop.')
for epoch in range(epochs):
    # Get the next batch from the corpus
    inputs, outputs, resets = corpus.next_batch()

    # Reset any of the samples that wrapped to the beginning
    # of a document
    for i in range(batch_size):
        if resets[i]:
            for j, layer in enumerate(current_hiddens):
                # Zero out this particular batch
                current_hiddens[j][i] = 0

    # Feed inputs and outputs into the training function
    cost, current_hiddens = train(inputs, outputs, current_hiddens)

    # Update smooth cost for logging
    if smooth_cost is None:
        smooth_cost = cost
    else:
        smooth_cost = smooth_cost * 0.01 + 0.99 * cost

    # Log cost
    print('Epoch %d\tSmooth Loss %f\tLoss %f' % (epoch, smooth_cost, cost))

    # Periodically save checkpoints
    if epoch % arg.checkpoint_frequency == 0:
        path = os.path.join(arg.out, 'epoch-%d-%f.pk' % (epoch, smooth_cost))
        with open(path, 'wb') as f:
            print('Saving checkpoint to %s' % path)
            pickle.dump(forward_network, f)

    # Periodically sample on batch 0
    if epoch % arg.sample_frequency == 0:
        tokens = []
        next_token = outputs[0][-1] # Last next token of batch 0
        hiddens = current_hiddens[0]
        for t in range(arg.sample_length):
            predictions, hiddens = singleton(next_token, hiddens)

            # Softmax to get the token given the prediction
            if arg.softmax_temperature == 0:
                next_token = numpy.argmax(predictions)
            else:
                # Apply softmax temperature
                predictions = predictions ** (1 / arg.softmax_temperature)
                predictions /= predictions.sum()

                # Choose probabilistically
                next_token = numpy.random.choice(len(predictions), p = predictions)

            tokens.append(next_tokens)

        print('Sample:')
        print(' '.join(alphabet.to_token(token) for token in tokens))
