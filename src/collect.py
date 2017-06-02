'''
Collect together some tokenized files and turn them into
a single file for training.
'''

import pickle

all_docs = []
for n in range(12014):
    print('loading', n)
    with open(('tokenized_data/wiki_%d.pkl' % n), 'rb') as f:
        all_docs.extend(pickle.load(f))

pickle.dump(all_docs, open('data.pkl', 'wb'))
