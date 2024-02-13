import torch
import torch.nn as nn


class GRUNet(nn.Module):


    '''
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden
    '''

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        # Initialiser la classe parente, nn.Module
        super(GRUNet, self).__init__()
        # Stocker la taille de l'état caché et le nombre de couches
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Créer la couche d'embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Créer la couche GRU
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)

        # sortie des paramètres 
        self.outLayer = nn.Linear(embedding_dim, 8)

    def forward(self, x):
        # Passer l'entrée à travers la couche d'embedding
        x = self.embedding(x)
        # Initialiser l'état caché
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Passer l'embedding et l'état caché initial à travers le GRU
        out, _ = self.gru(x, h0)
        # Renvoyer la sortie du GRU
        out = self.outLayer(out)
        return out

