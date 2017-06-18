import theano
import theano.tensor as T
import numpy

class Embedding(self, alphabet_size, embedding_size):
    def __init__(self):
        self.embedding_values = numpy.random.uniform(
              size = (alphabet.size, arg.embedding_size)
          ).astype(theano.config.floatX)

        self.update_gate = theano.shared(
            value = self.embedding_values,
            name = 'E',
            borrow = True
        )

    def singleton(self):
        hidden, input = T

    def unroll_one_step_nonbatched(self, hidden, input):
        return hidden, self.embeddings[input]

    # An Embedding layer doesn't actually have a hidden state,
    # so we just pass the hidden state directly through.
    def unroll_one_step_batched(self, hidden, input):
        return hidden, self.embeddings[input]

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
        initial_hidden_matrix = T.matrix('h')

        # Resulting output distributions
        result_variables = []
        current_hidden = initial_hidden_matrix

        # Unroll (i) times
        for i in range(seq_length):
            output, current_hidden = self.unroll_one_step_batched(
                current_hidden,
                input_matrix[i]
            )

            result_variables.append(output)

        return input_matrix, initial_hidden_matrix, result_variables, current_hidden

class StackedRNN:
    def __init__(self, layers, nonlinearity = T.nnet.relu):
        self.layers = layers
        self.nonlinearity = nonlinearity

        _x = T.vector('x')
        self.apply_nonlinearity = theano.function(x, nonlinearity(x))

        self.params = sum(layer.params for layer in layers)

    # Singleton that manually evaluates a single
    # forward run.
    def singleton(self):
        layer_functions = []
        for layer in self.layers:
            layer_functions.append(layer.singleton())

        def run_forward(hiddens, current_input):
            current_output = None
            new_hiddens = []
            for i, fn in enumerate(layer_functions):
                new_hidden, current_output = fn(hiddens[i], current_input)
                new_hiddens.append(new_hidden)
                current_input = self.apply_nonlinearity(current_output)

        pass
            return numpy.stack(new_hiddens), current_input

        return run_forward

    # Unroll to a given sequence length.
    def unroll(self, seq_length):
        # Input matrix is an 3d array of word embeddings
        # of size seq_length x batch_size x input_size
        input_matrix = T.tensor3('x')

        # Hidden matrices are a 2d array of hidden states
        # of size layers x hidden_size
        initial_hidden_matrices = T.matrix('h')

        current_inputs = input_matrix
        current_outputs = None
        new_hiddens = []

        # Advance layers one at a time.
        for i, layer in enumerate(layers):
            current_hidden = initial_hidden_matrices[i]
            current_outputs = []

            # Unroll (i) times
            for i in range(seq_length):
                output, current_hidden = self.unroll_one_step_batched(
                    current_hidden,
                    current_inputs[i]
                )

                current_outputs.append(output)

            new_hiddens.append(current_hidden)
            current_inputs = [self.nonlinearity(value) for value in current_outputs]

        return input_matrix, initial_hidden_matrices, current_outputs, new_hiddens
