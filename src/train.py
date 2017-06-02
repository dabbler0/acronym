import theano
import theano.tensor as T
from model import GRU
from alphabet import Alphabet
from corpus import Corpus
import pickle
import argparse
import os

parser = argparse.ArgumentParser(description = 'Run language model training.')
parser.add_argument('--data', dest='data', help='data file to run on')
parser.add_argument('--alphabet', dest='alphabet', help='alphabet file to run with')
parser.add_argument('--embedding_size', dest='embedding_size', help='size of word embedding to use')
parser.add_argument('--rnn_size', dest='rnn_size', help='rnn size to use')
parser.add_argument('--batch_size', dest='batch_size', help='minibatch size to use')
parser.add_argument('--seq_length', dest='seq_length', help='sequence length to use')
parser.add_argument('--epochs', dest='epochs', help='number of epochs to run for')
parser.add_argument('--out', dest='out', help='output directory (created if not existent)')
parser.add_argument('--resume', dest='resume', help='model file to resume from')
arg = parser.parse_args()

# Load data
data = pickle.load(arg.data)
alphabet = pickle.load(arg.alphabet)

corpus = Corpus(data)

# Create checkpoint directy if it doesn't
# already exist
if not os.path.exists(arg.out):
    os.makedirs(arg.out)

embedding_size = arg.embedding_size
hidden_size = arg.rnn_size
batch_size = rnn.batch_size
seq_length = arg.seq_length

# Create the model
model = GRU(alphabet.size)

# Create word embeddings
embeddings = [
    numpy.random.uniform(
        size = (arg.embedding_size,)
    ) for _ in range(alphabet_size)
]

input_matrix, initial_hidden, result_vars, hidden_vars = model.unroll(seq_length)


# Cost
output_labels = T.matrix('y')
cost = T.nnet.categorical_crossentropy(result_vars[0], output_labels[0])
for i in range(1, seq_length)
    cost += T.nnet.categorical_crossentropy(result_vars[i], output_labels[i])
cost /= seq_length

# Training function
f = theano.function(
    [input_matrix, initial_hidden],
    [
        # Input grad, for accumulation to adjust embeddings
        T.grad(cost, input_matrix),

        # Four gates:
        T.grad(cost, model.reset_gate),
        T.grad(cost, model.update_gate),
        T.grad(cost, model.create_gate),
        T.grad(cost, model.output_gate),

        # New hidden states:
        hidden_vars[seq_length],

        # Loss:
        cost
    ]
)

def train(inputs, outputs, resets, current_hiddens):
    # Reset current hiddens as necessary
    for i, r in enumerate(resets):
        if r:
            current_hiddens[i] = numpy.zeros(hidden_size)

    # Create embedded input
    input_embeddings = numpy.zeros((seq_length, batch_size, embedding_size))
    for i in range(seq_length):
        for j in range(batch_size):
            input_embeddings[i][j] = embeddings[inputs[i][j]]

    # Compute:
    (input_grad, reset_grad,
        update_grad, create_grad,
        output_grad, new_hiddens,
        loss) = f(input_embeddings, current_hiddens)

    # Update parameters:
    model.reset_gate.set_value(
        model.reset_gate.get_value() - lr * reset_grad
    )
    model.update_gate.set_value(
        model.update_gate.get_value() - lr * update_grad
    )
    model.create_gate.set_value(
        model.create_gate.get_value() - lr * create_grad
    )
    model.output_gate.set_value(
        model.output_gate.get_value() - lr * output_grad
    )

    # Update embeddings
    for i, n in enumerate(inputs):
        embeddings[n] -= lr * input_grad[i]

    # Return loss and new hiddens
    return loss, new_hiddens

# Training loop
smooth_loss = None
current_hiddens = numpy.zeros(batch_size, hidden_size)
for epoch in range(epochs):
    inputs, outputs, resets = corpus.next_batch()

    loss, current_hiddens = train(inputs, outputs, resets, current_hiddens
    smooth_loss = smooth_loss * 0.99 + loss * 0.01

    if epoch % 100 == 0:
        print('Epoch %d\tLoss %f\t' % (epoch, smooth_loss))
    if epoch % 1000 == 0:
        with open(os.path.join(arg.out, 'epoch-%d.pkl' % epoch), 'wb') as f
            pickle.dump(f, (model, embeddings))

with open(os.path.join(arg.out, 'final.pkl' % epoch), 'wb') as f
    pickle.dump(f, (model, embeddings))
