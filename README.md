# Simplified LSTM
Simplified LSTM recurrent neural network with adjustable number of layers

Implementation of Simplfied LSTM or Recurrent Additive Network (RAN) by Kenton Lee, Omer Levy, and Luke Zettlemoyer:

http://www.kentonl.com/pub/llz.2017.pdf

For training S-LSTM clone this repository and run:

```
python train_RAN.py
```
This will train a S-LSTM with 2 hidden layers, embedding size of 650 and hidden layer size of 200.

Possible arguments:

- --seed: type=int, default=1111, help=set seed
- --cuda: action='store_true', help='use GPU to compute'
- --log: type=int, default=200, metavar='N', help='number of iterations after which to log'
- --save: type=str,  default='model.pt', help='path to save the final model'
- --load: action='store_true', help='path to load model to continue training from'
- --data: type=str, default='../data/penn/', help='data directory'
- --embed_size: type=int, default=200, help='size of word embeddings'
- --nhid: type=int, default=200, help='number of hidden units per layer'
- --nlayers: type=int, default=2, help='number of layers'
- --lr: type=float, default=20, help='learning rate'
- --bsz: type=int, default=40, metavar='N', help='batch size'
- --bptt: type=int, default=35, help='sequence length'
- --clip: type=float, default=0.25, help='gradient clipping'
- --epochs: type=int, default=150, help='maximum number of epochs'
- --dropout: type=float, default=0.5, help='percentage of dropout applied between layers'
- --tying: action='store_true', help='use weight tying if true'

Method for training LM on Penn Tree Bank (PTB) used from:

https://github.com/pytorch/examples/tree/master/word_language_model
