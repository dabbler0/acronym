'''
Alphabet class for mapping tokens to indices and
vice versa.
'''
class Alphabet:
    def __init__(self, tokens, n_unknowns):
        # The (n) most common unknown tokens
        # in a given document, sorted
        for n in range(n_unknowns):
            tokens.append('__UNK_%d__' % n)

        # Any other unknown token
        tokens.append('__UNK__')

        self.size = len(tokens)
        self.n_unknowns = n_unknowns

        self.to_index_dict = {}
        self.to_token_dict = {}
        for i, t in enumerate(tokens):
            self.to_index_dict[t] = i
            self.to_token_dict[i] = t

    def to_index(self, token):
        return self.to_index_dict[token]

    def to_token(self, index):
        return self.to_token_dict[index]

    def __contains__(self, key):
        return key in self.to_index_dict
