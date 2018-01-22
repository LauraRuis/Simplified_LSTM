import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class RAN(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, tie_weights, nlayers, dropout, cuda):
        super(RAN, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.nlayers = nlayers

        # input and output layer weights and biases
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.h2o = nn.Linear(hidden_size, vocab_size)

        # RAN cell weights and biases
        self.weights = []
        self.biases = []
        for layer in range(self.nlayers):

            if cuda:
                w_x2c = nn.Parameter(torch.Tensor(hidden_size, embedding_size).cuda())
                w_h2i = nn.Parameter(torch.Tensor(hidden_size, hidden_size).cuda())
                w_x2i = nn.Parameter(torch.Tensor(hidden_size, embedding_size).cuda())
                w_h2f = nn.Parameter(torch.Tensor(hidden_size, hidden_size).cuda())
                w_x2f = nn.Parameter(torch.Tensor(hidden_size, embedding_size).cuda())

                b_x2c = nn.Parameter(torch.Tensor(hidden_size).cuda())
                b_h2i = nn.Parameter(torch.Tensor(hidden_size).cuda())
                b_x2i = nn.Parameter(torch.Tensor(hidden_size).cuda())
                b_h2f = nn.Parameter(torch.Tensor(hidden_size).cuda())
                b_x2f = nn.Parameter(torch.Tensor(hidden_size).cuda())
            else:
                w_x2c = nn.Parameter(torch.Tensor(hidden_size, embedding_size))
                w_h2i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
                w_x2i = nn.Parameter(torch.Tensor(hidden_size, embedding_size))
                w_h2f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
                w_x2f = nn.Parameter(torch.Tensor(hidden_size, embedding_size))

                b_x2c = nn.Parameter(torch.Tensor(hidden_size))
                b_h2i = nn.Parameter(torch.Tensor(hidden_size))
                b_x2i = nn.Parameter(torch.Tensor(hidden_size))
                b_h2f = nn.Parameter(torch.Tensor(hidden_size))
                b_x2f = nn.Parameter(torch.Tensor(hidden_size))

            # loop over weights and biases and set attributes self
            param_names = ['x2c', 'h2i', 'x2i', 'h2f', 'x2f']
            weights = (w_x2c, w_h2i, w_x2i, w_h2f, w_x2f)
            biases = (b_x2c, b_h2i, b_x2i, b_h2f, b_x2f)
            for i, w in enumerate(weights):
                setattr(self, 'w_' + param_names[i] + '_' + str(layer), w)
            for i, b in enumerate(biases):
                setattr(self, 'b_' + param_names[i] + '_' + str(layer), b)

            weights = []
            biases = []
            for i in range(len(param_names)):
                weights.append(getattr(self, 'w_' + param_names[i] + '_' + str(layer)))
            for i in range(len(param_names)):
                biases.append(getattr(self, 'b_' + param_names[i] + '_' + str(layer)))

            self.weights.append(tuple(weights))
            self.biases.append(tuple(biases))

        if tie_weights:
            if self.hidden_size != self.embedding_size:
                raise ValueError('When using the weight tying, hidden size must be equal to embedding size')
            self.h2o.weight = self.embeddings.weight

        self.init_weights()

    def init_weights(self):

        # input and output layer weights
        nn.init.xavier_uniform(self.embeddings.weight)
        self.h2o.bias.data.fill_(0)
        nn.init.xavier_uniform(self.h2o.weight)

        # init weights
        for layer_weights in self.weights:
            for w in layer_weights:
                nn.init.xavier_uniform(w)

        # init biases
        for layer_biases in self.biases:
            for b in layer_biases:
                b.data.fill_(0)

    def forward(self, word, h_s, l_s, weight_analysis=False):

        # set embeddings and dropout
        inputs = self.embeddings(word)
        inputs = self.drop(inputs)

        # initialize lists for saving input and forget gates
        input_gates = []
        forget_gates = []

        # initialize list for saving hidden states per layer
        next_hidden = []
        next_latent = []
        
        # loop over layers
        for layer in range(self.nlayers):

            # layer calculations
            next_hid, next_lat, inputs, i_gates, f_gates = self.layer_calculations(
                inputs, h_s[layer], l_s[layer], self.weights[layer], self.biases[layer], weight_analysis
            )
                    
            # save hidden state per layer
            next_hidden.append(next_hid)
            next_latent.append(next_lat)

            # dropout between layers, not at final layer
            if layer < self.nlayers - 1:
                inputs = self.drop(inputs)

            # append input gates and forget gates per layer (for weight analysis per layer)
            input_gates.append(i_gates)
            forget_gates.append(f_gates)

        # concatenate hidden states of layers to make next hidden state
        h_s = torch.cat(next_hidden, 0).view(self.nlayers, *next_hidden[0].size())
        l_s = torch.cat(next_latent, 0).view(self.nlayers, *next_latent[0].size())

        # dropout before going through output layer
        outputs = self.drop(inputs)
        outputs = self.h2o(outputs)

        return h_s, l_s, outputs, input_gates, forget_gates

    def layer_calculations(self, x_input, hidden, latent, weights, biases, weight_analysis=False):
        i_gates = []
        f_gates = []
        seq_length = x_input.size()[0]
        hiddens = []

        # loop over sequence of words and put them through RAN cell
        for i in range(seq_length):
            hidden, latent, i_gate, f_gate = self.RANcell(x_input[i], hidden, latent, weights, biases)
            hiddens.append(hidden)

            # save gates for weight analysis
            if weight_analysis:
                i_gates.append(i_gate)
                f_gates.append(f_gate)

        # concatenate all outputs to input for next layer
        outputs = torch.cat(hiddens, 0).view(x_input.size(0), *hiddens[0].size())

        return hidden, latent, outputs, i_gates, f_gates

    def RANcell(self, x_in, hidden_state, latent_state, weights, biases):

        # get weights and biases
        w_x2c, w_h2i, w_x2i, w_h2f, w_x2f = weights
        b_x2c, b_h2i, b_x2i, b_h2f, b_x2f = biases

        # input to content
        c_tilde = F.linear(x_in, w_x2c, b_x2c)

        # input gate
        input_gate = F.sigmoid(F.linear(hidden_state, w_h2i, b_h2i) + F.linear(x_in, w_x2i, b_x2i))

        # forget gate
        forget_gate = F.sigmoid(F.linear(hidden_state, w_h2f, b_h2f) + F.linear(x_in, w_x2f, b_x2f))

        # element wise multiplication
        latent_state = input_gate * c_tilde + forget_gate * latent_state

        # activation
        hidden_state = F.tanh(latent_state)

        return hidden_state, latent_state, input_gate, forget_gate

    def init_states(self, cud, bsz):
        weight = next(self.parameters()).data
        if cud:
            return Variable(weight.new(self.nlayers, bsz, self.hidden_size).zero_().cuda())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.hidden_size).zero_())
