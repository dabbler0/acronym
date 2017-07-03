import math
import theano
import theano.tensor as T
import numpy

# An RNNUnit
class RNN:
    def __init__(self):
        self.params = []
        self.n_hiddens = 0

    # hidden_state: tuple of T.tensors
    # input: a T.tensor
    # batched: True if this is being run in minibatch mode
    def step(self, hidden_state, input, batched):
        raise Exception('Not implemented.')
        # return hidden_state, output
        # hidden_state: list of T.tensors
        # output: a T.tensor

    # If batched is True, inputs here should be a tensor3
    # of size seq_length x batch_size x input_size
    # If batched is False, inputs here should be a matrix
    # of size seq_length x input_size
    def unroll(self, seq_length, hidden, inputs, batched=True):
        outputs = []

        for i in range(seq_length):
            hidden, output = self.step(hidden, inputs[i])
            outputs.append(output)

        return hidden, numpy.stack(outputs)

# Simple embedding layer that selects a n-dimensional vector
# for each integer input
def Embedding(RNN):
    def __init__(self, alphabet_size, embedding_size):
        self.embedding_values = numpy.random.uniform(
              size = (alphabet_size, embedding_size)
          ).astype(theano.config.floatX)

        self.embedding = theano.shared(
            value = self.embedding_values,
            name = 'E',
            borrow = True
        )

        self.params = [
            self.embedding
        ]

        self.n_hiddens = 0

    # Embedding does not have any hidden state.
    def step(self, hidden_state, input, batched):
        return hidden_state, self.embedding[input]

# A GRU
def GRU(RNN):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define to-insert matrix
        self.update_gate_values = numpy.random.normal(
            scale = 1 / (hidden_size + input_size),
            size = (hidden_size + input_size, hidden_size)
        ).astype(theano.config.floatX)
        self.update_gate = theano.shared(
            value = self.update_gate_values,
            name = 'U',
            borrow = True
        )

        # Define insert bias
        self.update_bias_values = numpy.zeros(hidden_size, dtype=theano.config.floatX)
        self.update_bias = theano.shared(
            value = self.update_bias_values,
            name = 'u',
            borrow = True
        )

        # Define to-hidden matrix
        self.create_gate_values = numpy.random.normal(
            scale = 1 / (hidden_size + input_size),
            size = (hidden_size + input_size, hidden_size)
        ).astype(theano.config.floatX)
        self.create_gate = theano.shared(
            value = self.create_gate_values,
            name = 'C',
            borrow = True
        )

        # Define hidden bias
        self.create_bias_values = numpy.zeros(hidden_size, dtype=theano.config.floatX)
        self.create_bias = theano.shared(
            value = self.create_bias_values,
            name = 'c',
            borrow = True
        )

        # Define to-reset matrix
        self.reset_gate_values = numpy.random.normal(
            scale = 1 / (hidden_size + input_size),
            size = (hidden_size + input_size, hidden_size)
        ).astype(theano.config.floatX)
        self.reset_gate = theano.shared(
            value = self.reset_gate_values,
            name = 'R',
            borrow = True
        )

        # Define to-reset bias
        self.reset_bias_values = numpy.zeros(hidden_size, dtype=theano.config.floatX)
        self.reset_bias = theano.shared(
            value = self.reset_bias_values,
            name = 'r',
            borrow = True
        )

        # Define to-output matrix
        self.output_gate_values = numpy.random.normal(
            scale = 1 / hidden_size,
            size = (hidden_size, output_size)
        ).astype(theano.config.floatX)
        self.output_gate = theano.shared(
            value = self.output_gate_values,
            name = 'O',
            borrow = True
        )

        # Define output bias
        self.output_bias_values = numpy.zeros(output_size, dtype=theano.config.floatX)
        self.output_bias = theano.shared(
            value = self.output_bias_values,
            name = 'o',
            borrow = True
        )

        # Pack params together
        self.params = [
            self.output_gate,
            self.create_gate,
            self.reset_gate,
            self.update_gate,

            self.output_bias,
            self.create_bias,
            self.reset_bias,
            self.update_bias
        ]

        self.n_hiddens = 1

    # hidden_state should be an array that contains a single
    # tensor.
    def step(self, hidden_state, input, batched):
        # The primary axis is the axis of state dimensions
        primary_axis = 1 if batched else 0

        # Unpack hidden state from single-element array
        (hidden,) = hidden_state

        # hidden has size batch_size x hidden_size
        # input has size batch_size x input_size
        all_info = T.concatenate([hidden, input], axis=primary_axis)

        reset = T.nnet.sigmoid(T.dot(all_info, self.reset_gate) + self.reset_bias)
        update = T.nnet.sigmoid(T.dot(all_info, self.update_gate) + self.update_bias)

        # Reset some hidden things
        reset_info = T.concatenate([reset * hidden, input], axis=primary_axis)

        # Update
        created = T.tanh(T.dot(reset_info, self.create_gate) + self.create_bias)
        new_hidden = update * created + (1 - update) * hidden
        output = T.dot(new_hidden, self.output_gate) + self.output_bias

        return (new_hidden,), output

def Output(RNN):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        #self.batch_size = batch_size
        self.truth_variable = truth_variable

        # Approximately sqrt the output size
        intermediate_size = int(output_size ** 0.5)
        self.intermediate_size = intermediate_size
        self.outputs_per_class = math.ceil(output_size / intermediate_size)

        # SoftmaxH values
        self.first_level_values = numpy.random.normal(
            scale = 1 / (input_size),
            size = (input_size, intermediate_size)
        ).astype(theano.config.floatX)
        self.first_level = theano.shared(
            value = self.first_level_values,
            name = 'W1',
            borrow = True
        )

        self.first_level_bias_values = numpy.zeros(intermediate_size, dtype=theano.config.floatX)
        self.first_level_bias = theano.shared(
            value = first_level_bias_values,
            name = 'b1',
            borrow = True
        )

        self.second_level_values = numpy.random.normal(
            scale = 1 / (input_size),
            size = (intermediate_size, input_size, outputs_per_class)
        ).astype(theano.config.floatX)
        self.second_level = theano.shared(
            value = self.second_level_values,
            name = 'W2',
            borrow = True
        )

        self.second_level_bias_values = numpy.zeros(
            (intermediate_size, outputs_per_class),
            dtype=theano.config.floatX
        )
        self.second_level_bias = theano.shared(
            value = second_level_bias_values,
            name = 'b1',
            borrow = True
        )

        self.params = [
            self.first_level,
            self.first_level_bias,
            self.second_level,
            self.second_level_bias,
        ]

        self.n_hiddens = 0

    def create_training_node(self, batch_size, truth_variable):
        return TrainingNode(self, batch_size, truth_variable)

    def step(self, hidden_state, input, batched):
        # Predict class probabilities
        class_probs = T.nnet.softmax(T.dot(input, self.first_level) + self.first_level_bias)

        # For each class, predict output probabilities
        output_prob_list = []
        for i in range(self.intermediate_size):
            output_probs.append(
                class_probs[i] * T.nnet.softmax(
                    T.dot(input, self.second_level[i]) + self.second_level_bias[i]
                )
            )

        output_probs = T.stack(output_prob_list)

        # Get rid of excess probabilities and renormalize
        if batched:
            output_probs = output_probs[:, :self.output_size]
            output_probs /= output_probs.sum(1)
        else
            output_probs = output_probs[:self.output_size]
            output_probs /= output_probs.sum()

        # Return
        return output_probs

class TrainingNode(RNN):
    def __init__(self, output_node, batch_size, truth_variable):
        self.outputs_per_class = output_node.outputs_per_class
        self.first_level = output_node.first_level
        self.first_level_bias = output_node.first_level_bias
        self.second_level = output_node.second_level
        self.second_level_bias = output_node.second_level_bias

        self.truth_variable = truth_variable
        self.batch_size = batch_size

        self.params = self.output_node.params
        self.n_hiddens = 0

    def step(self, hidden_state, input, batched):
        if not batched:
            raise Exception('TrainingNode should only be used for training, and in batch mode')

        class_probs = T.nnet.softmax(T.dot(input, self.first_level) + first_level_bias)
        which_class = self.truth_variable // self.outputs_per_class
        which_postclass_index = self.truth_variable % self.outputs_per_class

        costs = []

        for i in range(self.batch_size):
            probs = class_probs[which_class[i]] * T.nnet.softmax(
                T.dot(input, self.second_level[which_class[i]]) + self.second_level_bias[which_class[i]]
            )

            target_prob = probs[which_postclass_index[i]]

            # Cross entropy
            costs.append(-T.log(target_prob))

        # Take mean over minibatches
        return sum(costs) / len(costs)

# A stacked RNN of several other RNNs
class Composition(RNN):
    def __init__(self, layers):
        self.layers = layers
        self.params = sum(x.params for x in layers)
        self.n_hiddens = sum(x.n_hiddens for x in layers)

    def step(self, hidden_state, input, batched):
        new_hidden_state = []
        for i, layer in enumerate(self.layers):
            new_hidden, input = layer.step(hidden_state[i], input)
            new_hidden_state.append(new_hidden)
        return tuple(new_hidden_state), input
