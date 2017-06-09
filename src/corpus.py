'''
Corpus utility for creating minibatches for
training
'''
import pickle
import numpy

class Corpus:
    def __init__(self, docs, seq_length, batch_size):
        self.docs = list(filter(lambda x: len(x) > seq_length + 1, docs))
        self.doc_lengths = list(map(lambda x: len(x), self.docs))

        t = sum(self.doc_lengths)
        self.doc_p = list(map(lambda x: x / t, self.doc_lengths))

        self.seq_length = seq_length
        self.batch_size = batch_size

        self.indices = []

        # Initialize batch pointers
        for batch in range(self.batch_size):
            document_index = numpy.random.choice(len(self.docs), p=self.doc_p)
            index = numpy.random.choice(len(self.docs[document_index]) - seq_length - 1)

            self.indices.append((document_index, index))

    # Returns batch_input (tensor), batch_labels (tensor), reset (array),
    # where reset[i] is true if you need to reset the hidden inputs on
    # the i(th) minibatch.
    def next_batch(self):
        seq_length, batch_size = self.seq_length, self.batch_size

        batch_input = numpy.zeros(
            (seq_length, batch_size),
            dtype='int32'
        )
        batch_labels = numpy.zeros(
            (seq_length, batch_size),
            dtype='int32'
        )
        reset = [False for _ in range(batch_size)]

        # Advance each pointer by (seq_length)
        for i, (d, p) in enumerate(self.indices):
            # If we are about to run off the end of the document,
            # don't, and jump to another random document instead
            if p + seq_length + 1 > len(self.docs[d]):
                d = numpy.random.choice(len(self.docs), p=self.doc_p)
                p = numpy.random.choice(len(self.docs[d]) - seq_length - 1)
                reset[i] = True

            # Populate the batch input with the given sequence
            doc = self.docs[d]
            for k in range(seq_length):
                batch_input[k][i] = doc[p + k]

            # Populate the batch labels with the given sequence
            # shifted forward by one
            for k in range(seq_length):
                batch_labels[k][i] = doc[p + k + 1]

            p += seq_length
            self.indices[i] = (d, p)

        return batch_input, batch_labels, reset
