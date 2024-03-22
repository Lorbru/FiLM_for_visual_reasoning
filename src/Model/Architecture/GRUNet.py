import torch
import torch.nn as nn

class GRUNet(nn.Module):
    """
    ============================================================================================
    CLASS GRUNET(nn.Module) : GRU network for natural question processing

    METHODS : 
        * __init__(num_channels, output_size, vocab_size, dcr=128, dps=14): constructor
        * forward(x, z) : forward 
    ============================================================================================
    """

    def __init__(self, vocab_size, output_size, embedding_dim=200, hidden_size=4096, num_layers=2):
        """
        -- __init__(vocab_size, output_size, embedding_dim=200, hidden_size=4096, num_layers=2) : constructor

        In >> :
            * vocab_size: int  - vocabulary size
            * output_size: int - output size
            * embedding_dim: int - embedding dimension
            * hidden_size: int - number of hidden unit
            * num_layers: int - number of layers
        """
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size // num_layers
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.out_size = output_size 
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, self.hidden_size, num_layers, batch_first=True)
        self.outLayer = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        """
        -- forward(x) : forward

        In >> :
            * x: list[int] - input, encoded question
        """
        x = self.embedding(x)
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:,-1,:]
        out = self.outLayer(out)
        return out

