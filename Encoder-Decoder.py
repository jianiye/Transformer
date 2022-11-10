import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len, causal=False):
        super().__init__()

        self.d_k = d_k
        # the dimensions of k,q,v are the same
        self.n_heads = n_heads

        self.key = nn.Linear(d_model, d_k * n_heads)
        self.query = nn.Linear(d_model, d_k * n_heads)
        self.value = nn.Linear(d_model, d_k * n_heads)

        self.fc = nn.Linear(d_k * n_heads, d_model)

        # build a diagonal causal mask so that each token can only attend to its previous tokens
        self.causal = causal
        if causal:
            cm = torch.tril(torch.ones(max_len, max_len))
            self.register_buffer(
                'causal_mask',
                cm.view(1, 1, max_len, max_len)
            )

    def forward(self, q, k, v, pad_mask=None):
        q = self.query(q) # N*T*d_model  --> N*T*(d_k*n_heads)
        k = self.key(k) # N*T*d_model  --> N*T*(d_k*n_heads)
        v = self.value(v) # N*T*d_model  --> N*T*(d_k*n_heads)

        N = q.shape[0]
        T_output = q.shape[1]
        T_input = k.shape[1]

        # N, T, (d_k*n_heads) --> N, T, n_heads, d_k --> N, n_heads, T, d_k
        q = q.view(N, T_output, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(N, T_input, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(N, T_input, self.n_heads, self.d_k).transpose(1, 2)

        # compute Q K_transpose in the attention equation
        # @ is equal to np.dot
        # N, n_heads, T, d_k  @  N, n_heads, d_k, T --> N, n_heads, T, T
        attn_scores = q @ k.transpose(-2,-1) / math.sqrt(self.d_k)
        if pad_mask is not None:
            # mask: N,T --> N,1,1,T
            attn_scores = attn_scores.masked_fill(pad_mask[:, None, None, :] == 0, float('-inf'))

        if self.causal:
            attn_scores = attn_scores.masked_fill(self.causal_mask[:, :, :T_output, :T_input] == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # N, n_heads, T, T @ N, n_heads, T, d_k --> N, n_heads, T, d_k
        A = attn_weights @ v

        # N, n_heads, T, d_k --> N, T, n_heads, d_k
        A = A.transpose(1, 2)
        # N, T, n_heads, d_k --> N, T, n_heads*d_k
        A = A.contiguous().view(N, T_output, self.n_heads*self.d_k)
        # N, T, n_heads*d_k --> N, T, d_model
        return self.fc(A)


class EncoderBlock(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
        super().__init__()

        self.k_embed = nn.Linear(d_model, d_model)
        self.q_embed = nn.Linear(d_model, d_model)
        self.v_embed = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model) # layer norm normalizes the embedding dimension
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=False)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_prob),
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x, mask = None):
        x = self.ln1(x + self.mha(self.q_embed(x), self.k_embed(x), self.v_embed(x), mask))
        x = self.ln2(x + self.ann(x))
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_k, d_model, n_heads, max_len, dropout_prob=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model) # layer norm normalizes the embedding dimension
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.mha1 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=True)
        self.mha2 = MultiHeadAttention(d_k, d_model, n_heads, max_len, causal=False)
        self.ann = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_prob),
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, enc_output, dec_input, enc_mask = None, dec_mask = None):
        x = self.ln1(dec_input + self.mha1(dec_input, dec_input, dec_input, dec_mask))
        # print(x.shape, enc_output.shape, dec_input.shape)
        x = self.ln2(x + self.mha2(x, enc_output, enc_output, enc_mask))
        x = self.ln3(x + self.ann(x))
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

        position = torch.arange(max_len).unsqueeze(1)
        exp_term = torch.arange(0, d_model, 2)
        div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0,:,0::2] = torch.sin(position*div_term)
        pe[0,:,1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe) # save and load model

    def forward(self, x):
        # x: N*T*d_model
        x = x + self.pe[:, :x.size(1),:]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_len,
                 d_k,
                 d_model,
                 n_heads,
                 n_layers,
                 # n_classes,
                 dropout_prob):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        transformer_blocks = [
            EncoderBlock(
            d_k,
            d_model,
            n_heads,
            max_len,
            dropout_prob) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        # self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x, mask)

        # when we do classification, we only need to keep the final hidden sequence, by convention, keep the first
        # N*T*d_model --> N*1*d_model
        # x = x[:,0,:]

        x = self.ln(x)
        # x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_len,
                 d_k,
                 d_model,
                 n_heads,
                 n_layers,
                 dropout_prob):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        transformer_blocks = [
            DecoderBlock(
            d_k,
            d_model,
            n_heads,
            max_len,
            dropout_prob) for _ in range(n_layers)]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, enc_output, dec_input, enc_mask = None, dec_mask = None):
        x = self.embedding(dec_input)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(enc_output, x, enc_mask, dec_mask)

        x = self.ln(x)
        x = self.fc(x)
        return x


class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_input, dec_input, enc_mask, dec_mask):
        enc_output = self.encoder(enc_input, enc_mask)
        dec_output = self.decoder(enc_output, dec_input, enc_mask, dec_mask)
        return dec_output

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    x = np.random.randint(0,2000, size=(8, 128))
    x_t = torch.tensor(x).to(device)

    mask = np.ones((8, 128))
    mask[:, 64:] = 0
    mask_t = torch.tensor(mask).to(device)

    encoder = Encoder(2000, 128, 16, 16, 4, 2, 0.1)
    encoder.to(device)
    y = encoder(x_t, mask_t)
    print('Encoder test: ', y.shape)

    decoder = Decoder(2000, 128, 16, 16, 4, 2, 0.1)
    decoder.to(device)
    y = decoder(y, x_t, mask_t, mask_t)
    print('Decoder test: ',y.shape)

    transformer = Transformer(encoder, decoder)
    transformer.to(device)
    y = transformer(x_t, x_t, mask_t, mask_t)
    print('Transformer test: ',y.shape)
