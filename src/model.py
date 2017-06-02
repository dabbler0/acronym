import theano
import numpy

class GRU:
    def __init__(self, alphabet_size, embedding_size, hidden_size):
        # Define to-insert matrix
        update_gate_values = numpy.random.normal(
            scale = 1 / (hidden_size + embedding_size),
            size = (hidden_size + embedding_size, hidden_size)
        )
        self.update_gate = theano.shared(
            value = self.update_gate_values,
            name = 'U',
            borrow = True
        )

        # Define to-hidden matrix
        create_gate_values = numpy.random.normal(
            scale = 1 / (hidden_size + embedding_size),
            size = (hidden_size + embedding_size, hidden_size)
        )
        self.create_gate = theano.shared(
            value = self.create_gate_values,
            name = 'C',
            borrow = True
        )

        # Define to-reset matrix
        self.reset_gate_values = numpy.random.normal(
            scale = 1 / (hidden_size + embedding_size)
            size = (hidden_size + embedding_size, hidden_size)
        )
        self.reset_gate = theano.shared(
            value = self.reset_gate_values,
            name = 'R',
            borrow = True
        )

        # Define to-output matrix
        self.output_gate_values = numpy.random.normal(
            scale = 1 / hidden_size,
            size = (hidden_size, alphabet_size)
        )
        self.output_gate = theano.shared(
            value = self.output_gate_values,
            name = 'O',
            borrow = True
        )

    # Create a theano function that can be called
    # on a single hidden state + input index
    # to produce a distribution and another hidden state.
    #
    # To be used for forward-only passes.
    def singleton(self):


    def unroll_one_step_batched(self, hidden, input):
        # hidden has size batch_size x hidden_size
        # input has size batch_size x embedding_size
        all_info = T.concatenate([hidden, input], axis=1)

        reset = T.nnet.sigmoid(self.reset_gate * all_info)
        update = T.nnet.sigmoid(self.update_gate * all_info)

        # Reset some hidden things
        reset_info = T.concatenate([reset * hidden, input], axis=1)

        # Update
        created = T.nnet.tanh(self.create_gate * reset_info)
        new_hidden = update * created + (1 - update) * hidden
        output = T.nnet.softmax(self.output_gate * new_hidden)

        return output, new_hidden

    # Create a theano function that accepts an entire minibatch
    # matrix and does a training pass on it.
    #
    # To be used for training passes.
    def unroll(self, seq_length):
        # Input matrix is an 2d array of word embeddings
        # of size seq_length x batch_size
        input_matrix = T.imatrix('x')
        initial_hidden_matrix = theano.tensor.matrix('h')

        # Resulting output distributions
        result_variables = []
        hidden_variables = [initial_hidden_matrix]

        # Unroll (i) times
        for i in range(seq_length)
            output, hidden = self.unroll_one_step_batched(
                hidden_variables[-1],
                input_matrix[i]
            )

            result_variables.append(output)
            hidden_variables.append(hidden)

        return result_variables, hidden_variables
