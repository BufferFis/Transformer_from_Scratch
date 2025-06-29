import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super.__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Matric of shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        #Create a vector shape of seq_length
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (Seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        #Apply the sin to even and cos to odd 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model) [Extra dimension for Batch handling]

        self.register_buffer('pe', pe) #Should be saved when model saved but dont learn this as it stays fixed 

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        x = self.dropout(x)
        return x

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None: # the float value is to avoid devision by zero
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Param makes it learnable (this is multiplied)
        self.bias = nn.Parameter(torch.zeros(1)) # Param makes it learnable (this is added)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # -1 represents everything after the batch
        std = x.std(dim=-1, keepdim=True)
        norm = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return norm
    
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model) # Query weights
        self.w_k = nn.Linear(d_model, d_model) # Key weights
        self.w_v = nn.Linear(d_model, d_model) # Value weights
        
        self.w_o = nn.Linear(d_model, d_model) # Output weights
        self.dropout = nn.Dropout(dropout)

    
    @staticmethod
    def attention(query, key, value, mask, dropout= nn.Dropout):
        d_k = query.shape[-1]  # Get the last dimension size (d_k)

        # (batch_size, h, seq_len, d_k) --> (batch_size, h, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = torch.softmax(attention_scores, dim=-1) # (batch_size, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        x = torch.matmul(attention_scores, value) # (batch_size, h, seq_len, d_k)

        return (x, attention_scores)

    
    def forward(self, q, k, v, mask):
        # All are muls btw
        query = self.w_q(q) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)

        # Reshape (batch, Seq_len, d_model) to (batch_size, seq_len, h, d_k) and then transpose to (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # (batch_size, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # (batch_size, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self)

        # (Batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))  # Add the output of the sublayer to the input x, in paper they applied sublayer first then norm but video says something else



