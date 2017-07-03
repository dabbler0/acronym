import pickle

with open('/home/anthony/Data/gutenberg/input.txt', 'rb') as document:
    doc = []
    string = document.read()
    for char in string:
        doc.append(char)

    with open('../data/gutenberg.pkl', 'wb') as dump:
        pickle.dump([doc], dump)
