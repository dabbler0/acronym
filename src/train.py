import theano
import theano.tensor as T
from model import GRU
from alphabet import Alphabet
from corpus import Corpus
import pickle
import argparse
import os
import numpy
import sys

numpy.random.seed(0)

parser = argparse.ArgumentParser(description = 'Run language model training.')
parser.add_argument('--data', dest='data', help='data file to run on')
parser.add_argument('--alphabet', dest='alphabet', help='alphabet file to run with')
parser.add_argument('--embedding_size', type=int, dest='embedding_size', help='size of word embedding to use')
parser.add_argument('--rnn_size', type=int, dest='rnn_size', help='rnn size to use')
parser.add_argument('--batch_size', type=int, dest='batch_size', help='minibatch size to use')
parser.add_argument('--seq_length', type=int, dest='seq_length', help='sequence length to use')
parser.add_argument('--epochs', type=int, dest='epochs', help='number of epochs to run for')
parser.add_argument('--learning_rate', type=float, dest='learning_rate', help='learning rate')
parser.add_argument('--resume', dest='resume', help='checkpoint file to resume from')
parser.add_argument('--out', dest='out', help='output directory (created if not existent)')
arg = parser.parse_args()

# Load data
data = None
alphabet = None
with open(arg.data, 'rb') as f:
    data = pickle.load(f)
with open(arg.alphabet, 'rb') as f:
    alphabet = pickle.load(f)

# Create checkpoint directy if it doesn't
# already exist
if not os.path.exists(arg.out):
    os.makedirs(arg.out)

epochs = arg.epochs
embedding_size = arg.embedding_size
hidden_size = arg.rnn_size
batch_size = arg.batch_size
seq_length = arg.seq_length
lr = arg.learning_rate

corpus = Corpus(data, seq_length, batch_size)

# Create the model
if arg.resume is not None:
    print('Resuming!')
    with open(arg.resume, 'rb') as f:
        model, embeddings = pickle.load(f)

else:
    model = GRU(embedding_size, hidden_size, alphabet.size)

    # Create word embeddings
    embeddings = [
        numpy.random.uniform(
            size = (arg.embedding_size,)
        ).astype(theano.config.floatX) for _ in range(alphabet.size)
    )]

def run_training(model=None, embeddings=None, corpus=None, alphabet=None, lr=0.05):

    batch_size = corpus.batch_size
    seq_legnth = corpus.seq_length

    input_matrix, initial_hidden, result_vars, new_hidden = model.unroll(seq_length)

    # Cost
    output_labels = T.imatrix('y')
    cost = T.nnet.categorical_crossentropy(T.nnet.softmax(result_vars[0]), output_labels[0])
    for i in range(1, seq_length):
        cost += T.nnet.categorical_crossentropy(T.nnet.softmax(result_vars[i]), output_labels[i])
    cost /= seq_length
    cost = T.mean(cost)

    # Training function
    train_fun = theano.function(
        [output_labels, input_matrix, initial_hidden],
        [
            # Input grad, for accumulation to adjust embeddings
            T.grad(cost, input_matrix),

            # Four gates:
            T.grad(cost, model.reset_gate),
            T.grad(cost, model.update_gate),
            T.grad(cost, model.create_gate),
            T.grad(cost, model.output_gate),

            # New hidden states:
            new_hidden,

            # Loss:
            cost
        ]
    )

    def train(inputs, outputs, resets, current_hiddens):
        # Reset current hiddens as necessary
        for i, r in enumerate(resets):
            if r:
                current_hiddens[i] = numpy.zeros(hidden_size, dtype=theano.config.floatX)

        # Create embedded input
        input_embeddings = numpy.zeros((seq_length, batch_size, embedding_size), dtype=theano.config.floatX)
        for i in range(seq_length):
            for j in range(batch_size):
                input_embeddings[i][j] = embeddings[inputs[i][j]]

        # Compute:
        (input_grad, reset_grad,
            update_grad, create_grad,
            output_grad, new_hiddens,
            loss) = train_fun(outputs, input_embeddings, current_hiddens)

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
        for i, ns in enumerate(inputs):
            for j, n in enumerate(ns):
                embeddings[n] -= lr * input_grad[i][j]

        # Return loss and new hiddens
        return loss, new_hiddens

    singleton = model.singleton()
    def sample(length, hidden, index):
        results = []
        for i in range(length):
            embedding = embeddings[index]
            hidden, indices = singleton(hidden, embedding)
            indices = numpy.ndarray.flatten(indices)
            # Apply softmax by hand
            indices = numpy.exp(indices - numpy.max(indices))
            indices = indices / indices.sum()
            # Choice
            index = numpy.random.choice(len(indices), p = indices)

            results.append(index)

        return list(map(lambda x: alphabet.to_token(x), results))

    # Training loop
    smooth_loss = None
    current_hiddens = numpy.zeros((batch_size, hidden_size), dtype=theano.config.floatX)
    for epoch in range(epochs):
        inputs, outputs, resets = corpus.next_batch()

        #print(outputs)

        # Sample
        if smooth_loss is not None:
            print('Epoch %d\tLoss %f\t' % (epoch, smooth_loss))

        if epoch % 5000 == 0:
            with open(os.path.join(arg.out, 'epoch-%d.pkl' % epoch), 'wb') as f:
                print('Saving checkpoint to', os.path.join(arg.out, 'epoch-%d.pkl' % epoch))
                pickle.dump((model, embeddings), f)

        if epoch % 100 == 0:
            with open(os.path.join(arg.out, 'training-current.pkl'), 'wb') as f:
                print('Saving restore point to', os.path.join(arg.out, 'training-current.pkl'))
                pickle.dump((model, embeddings), f)
                print('Saved. Sample:')
                print(' '.join(sample(seq_length, current_hiddens[0], inputs[0][0])))

        sys.stdout.flush()

        loss, current_hiddens = train(inputs, outputs, resets, current_hiddens)
        if smooth_loss is None:
            smooth_loss = loss
        else:
            smooth_loss = smooth_loss * 0.99 + loss * 0.01

    with open(os.path.join(arg.out, 'final.pkl'), 'wb') as f:
        pickle.dump((model, embeddings), f)
