import os
import torch


class Dictionary(object):

    def __init__(self):
        self.word_to_ix = {}
        self.ix_to_word = []

    def add_word(self, word):
        if word not in self.word_to_ix:
            self.ix_to_word.append(word)
            self.word_to_ix[word] = len(self.ix_to_word) - 1
        return self.word_to_ix[word]

    def __len__(self):
        return len(self.ix_to_word)


class Corpus(object):

    def __init__(self, path, cuda):
        self.dictionary = Dictionary()
        self.cuda = cuda
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):

        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            if self.cuda:
                ids = torch.cuda.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word_to_ix[word]
                    token += 1

        return ids
