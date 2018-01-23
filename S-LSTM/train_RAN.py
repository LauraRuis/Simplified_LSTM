import torch.optim as optim
from RAN import *
import json
from helpers import *
from data import Corpus
import time
import torch
import math
import random
import argparse


parser = argparse.ArgumentParser(description='PyTorch simplified LSTM')
parser.add_argument('--seed', type=int, default=1111, help='set seed')
parser.add_argument('--cuda', action='store_true', help='use GPU to compute')
parser.add_argument('--log', type=int, default=200, metavar='N', help='number of iterations after which to log')
parser.add_argument('--save', type=str,  default='model.pt', help='path to save the final model')
parser.add_argument('--load', action='store_true', help='path to load model to continue training from')
parser.add_argument('--data', type=str, default='../data/penn/', help='data directory')
parser.add_argument('--embed_size', type=int, default=200, help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200, help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--lr', type=float, default=20, help='learning rate')
parser.add_argument('--bsz', type=int, default=40, metavar='N', help='batch size')
parser.add_argument('--bptt', type=int, default=35, help='sequence length')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=150, help='maximum number of epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='percentage of dropout applied between layers')
parser.add_argument('--tying', action='store_true', help='use weight tying if true')
args = parser.parse_args()

# set seed
torch.manual_seed(args.seed)

########################################################################################################################
# adjustable parameters
########################################################################################################################
BPTT = args.bptt
BSZ = args.bsz
EVAL_BSZ = 10
LR = args.lr
CLIP = args.clip
########################################################################################################################
PRINT_EVERY = args.log
CUDA = args.cuda
########################################################################################################################

# save decoders
DECODER = open("decoder.json", "w")
ENCODER = open("encoder.json", "w")

# read data
corpus = Corpus(args.data, CUDA)
vocab_size = len(corpus.dictionary)
print("|V|", vocab_size)

# turn into batches
training_data = batchify(corpus.train, BSZ, CUDA)
validation_data = batchify(corpus.valid, EVAL_BSZ, CUDA)

# set loss function
loss_function = nn.CrossEntropyLoss()

# Load the best saved model or initialize new one
if args.load:
    print('loading')
    with open(args.save, 'rb') as f:
        model = torch.load(f)
else:
    # initialize model
    model = RAN(args.embed_size, vocab_size, args.nhid, args.tying, args.nlayers, args.dropout, CUDA)

if CUDA:
    model.cuda()
    torch.cuda.manual_seed(1111)

# save encoder en decoder to json file
json.dump(corpus.dictionary.ix_to_word, DECODER, indent=4)
json.dump(corpus.dictionary.word_to_ix, ENCODER, indent=4)


def train(current_epoch):

    # enable dropout
    model.train()

    # initialize loss per epoch
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)

    # initialize hidden, latent and content layers with zero filled tensors
    hidden = model.init_states(CUDA, BSZ)
    latent = model.init_states(CUDA, BSZ)

    # shuffle indices to loop through data in random order
    random_indices = [i for batch, i in enumerate(range(0, training_data.size(0) - 1, BPTT))]
    random.seed(current_epoch)
    random.shuffle(random_indices)

    # loop over all data
    for batch, i in enumerate(range(0, training_data.size(0) - 1, BPTT)):

        # get batch of training data
        context, target = get_batch(training_data, random_indices[batch], BPTT)

        # repackage hidden stop backprop from going to beginning each time
        hidden = repackage_hidden(hidden)
        latent = repackage_hidden(latent)

        # set gradients to zero
        model.zero_grad()

        # forward pass
        hidden, latent, log_probs, _, _ = model(context, hidden, latent)

        # get the loss
        loss = loss_function(log_probs.view(-1, ntokens), target)

        # back propagate
        loss.backward()

        # clip gradients to get rid of exploding gradients problem
        if CLIP > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), CLIP)

        # update parameters
        optimizer = optim.SGD(model.parameters(), lr=LR)
        optimizer.step()

        # update total loss
        total_loss += loss.data

        # print progress
        if batch % PRINT_EVERY == 0 and batch > 0:
            cur_loss = total_loss[0] / PRINT_EVERY
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(training_data) // BPTT, LR,
                              elapsed * 1000 / PRINT_EVERY, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


# initialize best validation loss
best_val_loss = None

# hit Ctrl + C to break out of training early
try:

    # loop over epochs
    for epoch in range(1, args.epochs):

        epoch_start_time = time.time()
        train(epoch)
        val_loss = evaluate(model, corpus, loss_function, validation_data, CUDA, EVAL_BSZ, BPTT)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        # Anneal the learning rate if no improvement has been seen in the validation data set.
        else:
            LR /= 4


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
