import theano
import theano.tensor as T
import numpy

class GRU:
    def __init__(self, input_size, hidden_size, output_size):
        # Define to-insert matrix
        self.update_gate_values = numpy.random.normal(
            scale = 1 / (hidden_size + input_size),
            size = (hidden_size + input_size, hidden_size),
        ).astype(theano.config.floatX)
        self.update_gate = theano.shared(
            value = self.update_gate_values,
            name = 'U',
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

        # Pack params together
        self.params = [
            self.output_gate,
            self.create_gate,
            self.reset_gate,
            self.update_gate
        ]

    # Create a theano function that can be called
    # on a single hidden state + input index
    # to produce a distribution and another hidden state.
    #
    # To be used for forward-only passes.
    def singleton(self):
        hidden_state = T.vector('h')
        input_vector = T.vector('x')

        output, new_hidden = self.unroll_one_step_nonbatched(
            hidden_state, input_vector
        )

        return theano.function(
            [hidden_state, input_vector],
            [new_hidden, output]
        )

    def unroll_one_step_nonbatched(self, hidden, input):
        # hidden has size batch_size x hidden_size
        # input has size batch_size x input_size
        all_info = T.concatenate([hidden, input], axis=0)

        reset = T.nnet.sigmoid(T.dot(all_info, self.reset_gate))
        update = T.nnet.sigmoid(T.dot(all_info, self.update_gate))

        # Reset some hidden things
        reset_info = T.concatenate([reset * hidden, input], axis=0)

        # Update
        created = T.tanh(T.dot(reset_info, self.create_gate))
        new_hidden = update * created + (1 - update) * hidden
        output = T.dot(new_hidden, self.output_gate)

        return output, new_hidden

    def unroll_one_step_batched(self, hidden, input):
        # hidden has size batch_size x hidden_size
        # input has size batch_size x input_size
        all_info = T.concatenate([hidden, input], axis=1)

        reset = T.nnet.sigmoid(T.dot(all_info, self.reset_gate))
        update = T.nnet.sigmoid(T.dot(all_info, self.update_gate))

        # Reset some hidden things
        reset_info = T.concatenate([reset * hidden, input], axis=1)

        # Update
        created = T.tanh(T.dot(reset_info, self.create_gate))
        new_hidden = update * created + (1 - update) * hidden
        output = T.dot(new_hidden, self.output_gate)

        return output, new_hidden

    # Create a theano function that accepts an entire minibatch
    # matrix and does a training pass on it.
    #
    # To be used for training passes.
    def unroll(self, seq_length):
        # Input matrix is an 3d array of word embeddings
        # of size seq_length x batch_size x input_size
        input_matrix = T.tensor3('x')
        initial_hidden_matrix = theano.tensor.matrix('h')

        # Resulting output distributions
        result_variables = []
        hidden_variables = [initial_hidden_matrix]

        # Unroll (i) times
        for i in range(seq_length):
            output, hidden = self.unroll_one_step_batched(
                hidden_variables[-1],
                input_matrix[i]
            )

            result_variables.append(output)
            hidden_variables.append(hidden)

        return input_matrix, initial_hidden_matrix, result_variables, hidden_variables

class StackedRNN:
    def __init__(self, layers):
        self.layers = layers

        self.params = sum(layer.params for layer in layers)

