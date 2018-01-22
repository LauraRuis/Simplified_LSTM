import torch.autograd as autograd


def batchify(data, bsz, cuda):

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz

    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)

    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()

    if cuda:
        data.cuda()

    return data


def get_batch(source, i, bptt, evaluation=False):

    seq_len = min(bptt, len(source) - 1 - i)
    data = autograd.Variable(source[i:i + seq_len], volatile=evaluation)
    target = autograd.Variable(source[i + 1:i + 1 + seq_len].view(-1))

    return data, target


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == autograd.Variable:
        return autograd.Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(model, corpus, criterion, data_source, cuda, bsz, bptt):

    # disable dropout
    model.eval()

    # Turn on evaluation mode which disables dropout.
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_states(cuda, bsz)
    latent = model.init_states(cuda, bsz)
    for i in range(0, data_source.size(0) - 1, bptt):
        context, target = get_batch(data_source, i, bptt, evaluation=True)
        if target.size(0) == bsz * bptt:
            hidden, latent, log_probs, _, _ = model(context, hidden, latent)
            output_flat = log_probs.view(-1, ntokens)
            total_loss += len(context) * criterion(output_flat, target).data
            latent = repackage_hidden(latent)
            hidden = repackage_hidden(hidden)

    return total_loss[0] / len(data_source)
