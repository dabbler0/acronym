import os
import nltk
import pickle
import json
import argparse
import alphabet as alphabet_lib

def tokenize(text, alphabet):
    # Split into words and lowercase
    tokens = map(lambda x: x.lower(), nltk.word_tokenize(text))

    # Count unknowns
    counts = {}
    for token in tokens:
        if token not in alphabet:
            if token not in counts:
                counts[token] = 1
            else:
                counts[token] += 1

    all_unknowns = sorted(counts.keys(), key = lambda x: -counts[x])
    known_unknowns = all_unknowns[:alphabet.n_unknowns]

    unknowns = {}
    for i, unk in enumerate(known_unknowns):
        unknowns[unk] = ('__UNK_%d__' % i)

    result_indices = []

    # Convert to indices
    for token in tokens:
        if token in alphabet:
            result_indices.append(
                alphabet.to_index(token)
            )
        elif token in unknowns:
            result_indices.append(
                alphabet.to_index(unknowns[token])
            )
        else:
            result_indices.append(
                alphabet.to_index('__UNK__')
            )

    return result_indices

def process(alphabet, filename, dest)
    # Load the wiki entries
    documents = []
    with open(filename) as f:
        for line in f:
            document = json.loads(line)['wiki']
            documents.append(tokenize(document, alphabet))

    with open(dest) as f:
        pickle.dump(f, tokens)

parser = argparse.ArgumentParser(description='Preprocess Wikipedia data generated by wikiparse')
parser.add_argument('--alphabet', dest='alphabet', help='Alphabet file')
parser.add_argument('--filelist', dest='files', help='List of enwiki files in json format')
parser.add_argument('--outdir', dest='outdir', help='Directory in which to put tokenized files')

enwiki_filenames = json.load(open(parser.files))
alphabet = pickle.load(open(parser.alphabet))

n = 0
for filename in enwiki_filenames:
    process(alphabet, filename, os.path.join(parser.outdir, 'wiki_%d.pkl'))
