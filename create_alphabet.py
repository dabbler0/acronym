import json
import pickle
import argparse
from alphabet import Alphabet

parser = argparse.ArgumentParser(description='Preprocess Wikipedia data generated by wikiparse')
parser.add_argument('--size', dest='size', help='Size of the resulting alphabet')
parser.add_argument('--unk', dest='unk', help='Number of unknown tokens to allot indices for')
parser.add_argument('--counts',  dest='counts_file', help='Counts file')
parser.add_argument('--out', dest='out_file', help='Output file')

with open(parser.counts_file) as f:
  all_tokens = json.load(f)
  alphabet = Alphabet(list(map(lambda x: x[0], all_tokens[:parser.size]), parser.unk)
  with open(parser.out_file) as o:
    pickle.dump(o, alphabet)