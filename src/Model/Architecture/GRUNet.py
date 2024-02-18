import torch
import torch.nn as nn


class GRUNet(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim=200, hidden_size=4096, num_layers=1):

        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.out_size = output_size 
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.outLayer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:,-1,:]
        out = self.outLayer(out)
        return out

