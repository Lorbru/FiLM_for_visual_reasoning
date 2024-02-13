import torch
import torch.nn as nn


class GRUNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, nb_film_layers, res_channels):

        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.res_channels = res_channels
        self.nb_film_layers = nb_film_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.outLayer = nn.Linear(hidden_size, 2*nb_film_layers*res_channels)

    def forward(self, x):

        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(x.device)
        out, _ = self.gru(x, h0)
        print("Size out :")
        print(out.size(0), out.size(1), out.size(2))
        out = out[:,-1,:]
        out = self.outLayer(out)
        return out

