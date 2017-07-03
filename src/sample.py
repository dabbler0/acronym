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
parser.add_argument('--model', dest='model', help='model file to sample')
parser.add_argument('--alphabet', dest='alphabet', help='alphabet file to run with')
parser.add_argument('--temperature', type=float, dest='temperature', help='temperature; inf = uniform, 0 = true max')
#parser.add_argument('--length', type=int, dest='length', help='alphabet file to run with')
parser.add_argument('--acronym', dest='acronym', help='acronym to expand')
arg = parser.parse_args()

# Load model
with open(arg.model, 'rb') as f:
    model, embeddings = pickle.load(f)

with open(arg.alphabet, 'rb') as f:
    alphabet = pickle.load(f)

singleton = model.singleton()
def sample(acronym, hidden, index):
    results = []
    for char in acronym:
        embedding = embeddings[index]
        hidden, indices = singleton(hidden, embedding)
        indices = numpy.ndarray.flatten(indices)

        # Softmax
        indices = numpy.exp(indices - numpy.max(indices))
        indices = indices / indices.sum()

        # Zero out the probabilites of the things we don't want
        # for this part of the acronym
        if char != '?':
            for i in range(alphabet.size):
                if alphabet.to_token(i)[0] != char:
                    indices[i] = 0

        if arg.temperature == 0:
            index = numpy.random.argmax(indices)
        else:
            indices = indices ** (1 / arg.temperature) # Apply softmax temperature
            indices = indices / sum(indices) # Normalize to be a distribution
            index = numpy.random.choice(len(indices), p = indices)

        results.append(index)

    return list(map(lambda x: alphabet.to_token(x), results))

initial_hidden = numpy.zeros(model.create_gate_values.shape[1], dtype=theano.config.floatX)
index = 0

print(' '.join(sample(arg.acronym, initial_hidden, index)))
