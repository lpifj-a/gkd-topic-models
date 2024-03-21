# python 2.7
import os
import pickle
import json

import numpy as np

def onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

if __name__ == "__main__":
    # load the data
    train = np.load("original/train.txt.npy",allow_pickle=True,encoding='bytes')

    test = np.load("original/test.txt.npy", allow_pickle=True,encoding='bytes')

    with open("original/vocab.pkl", "rb") as infile:
        vocab = pickle.load(infile)
    vocab_size = len(vocab)

    # save the data
    train = np.array(
        [onehot(doc.astype('int'), vocab_size) for doc in train if np.sum(doc) != 0]
    )
    test = np.array(
        [onehot(doc.astype('int'), vocab_size) for doc in test if np.sum(doc) != 0]
    )

    os.mkdir("intermediate")

    np.savetxt("intermediate/train.txt", train)
    np.savetxt("intermediate/test.txt", test)

    with open("intermediate/vocab_dict.json", "w", encoding="utf-8") as outfile:
        json.dump(vocab, outfile)


