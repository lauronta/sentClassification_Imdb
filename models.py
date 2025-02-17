import torch.nn as nn
import torch
import math

def generate_sinusoidal_embeddings(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class SentModel(nn.Module):
    def __init__(self, emb_size, voc_size, num_layers, num_heads, hidden_size_mlp ,
                 output_size, PAD, maxlen=1000):
        super(SentModel, self).__init__()

        self.PAD = PAD
        self.emb_size = emb_size

        self.emb = nn.Embedding(voc_size, emb_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size_mlp,
            activation='relu'
        )

        # Création d'un TransformerEncoder avec plusieurs couches
        self.trans = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=num_layers
        )

        # Attention, seuls les modules sont envoyés vers les devices
        # pour envoyer automatiquement les tenseurs, il faut les "enregistrer"
        self.register_buffer("posemb", generate_sinusoidal_embeddings(maxlen, self.emb_size).unsqueeze(1))

        # du CLS vers la classif
        self.h2o = nn.Linear(emb_size, output_size)

   
    def forward(self, input, lengths=None):
        # Principales étapes
        # 1. translation of the input from int to emb
        # 2. Passage dans le trans
        # 3. Prediction sur le CLS

        # print("input", input.size())
        maxlen = input.size(0)
        batch_size = input.size(1)

        # A analyser (et à utiliser plus tard)
        padding_mask = (input[:, :] == self.PAD).T 

        # 1. translation of the input from int to emb + ajout des positional embeddings
        
        xemb = self.emb(input)
        xemb = xemb + self.posemb[:maxlen,:,:]
        
        # 2. Passage dans le transformer... Avec le masque pour le padding
        encoded_output = self.trans(xemb, src_key_padding_mask=padding_mask)
        # print("encoded_output", encoded_output.size())
        
        # 3. Appliquer la classification sur le CLS
        output = self.h2o(encoded_output[0,:,:]) # CLS is always in the first position in every batch 
        
        return output