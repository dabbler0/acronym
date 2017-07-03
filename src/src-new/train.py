import theano
import theano.tensor as T

# RNN things
from model import *
from alphabet import Alphabet
from corpus import Corpus

import numpy
import argparse
import os

# Constant random seed
numpy.random.seed(0)

# Command line arguments
parser = argparse.ArgumentParser(description = 'Run language model training.')
parser.add_argument('--data', dest = 'data', help = 'Data file to run on')
parser.add_argument('--alphabet', dest = 'alphabet', help = 'Alphabet file to run with')
parser.add_argument('--embedding_size', type = int, dest = 'embedding_size', help = 'Word embedding size')
parser.add_argument('--batch_size', type = int, dest = 'batch_size', help = 'Minibatch size')
parser.add_argument('--seq_length', type = int, dest = 'batch_size', help = 'Sequence length')
parser.add_argument('--epochs', type = int, dest = 'epochs', help = 'Number of epochs')
parser.add_argument('--learning_rate', type = float, dest = 'learning_rate', help = 'Learning rate')
parser.add_argument('--out', dest = 'out', help = 'Output directory')

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
true_output = T.matrix('y') # Will ultimately be seq_length x batch_size
train_layer = output_layer.create_training_node(batch_size, true_output)
training_network = Composition([embedding_layer, gru_layer_1, gru_layer_2, train_layer])

# Compute gradients
initial_hidden = [T.matrix('h') for _ in range(training_network.n_hiddens)]  # batch_size x rrn_size
inputs = T.imatrix('x') # seq_length x batch_size
new_hidden, costs = training_network.unroll(seq_length, initial_hidden, inputs)
cost = T.mean(costs)

training_function = theano.function(
    [inputs, true_output] + initial_hidden,
    [cost] + new_hidden
    updates = [
        (
            param,
            param - learning_rate * T.grad(cost, param)
        ) for param in training_network.params
    ]
)

# Training function
def train(inputs, outputs, hiddens)
    args = [inputs, outputs] + hiddens
    result = training_function(*args)
    return result[0], result[1:]

for i in range(epochs):
    
