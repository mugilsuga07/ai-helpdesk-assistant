import torch
import random

class CharDataset:
    def __init__(self, text, block_size):
        text = ''.join([c for c in text if ord(c) < 128])
        self.text = text
        self.block_size = block_size

       
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])

def batch_loader(raw_text, context_length, batch_size, dataset=None):
    if dataset is None:
        dataset = CharDataset(raw_text, context_length)

    x = torch.zeros((batch_size, context_length), dtype=torch.long)
    y = torch.zeros((batch_size, context_length), dtype=torch.long)

    for i in range(batch_size):
   
        idx = random.randint(0, len(dataset.text) - context_length - 1)
        chunk = dataset.text[idx:idx + context_length + 1]

        input_chunk = dataset.encode(chunk[:-1])
        target_chunk = dataset.encode(chunk[1:])

       
        while len(input_chunk) < context_length:
            input_chunk.append(0)
            target_chunk.append(0)

        x[i] = torch.tensor(input_chunk[:context_length])
        y[i] = torch.tensor(target_chunk[:context_length])

    return x, y

