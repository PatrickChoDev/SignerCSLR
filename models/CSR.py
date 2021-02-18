import torch
import torch.nn as nn
import math


class CNN3D(nn.Module):
    def __init__(self, num_classes):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(1, 3, 3), padding=(0,1,1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(1, 3, 3), padding=(0,1,1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 3, 3))
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(1, 2, 2))
        self.conv5b = nn.Conv3d(512, num_classes, kernel_size=(1, 3, 3))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 3, 3))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        return x


class Embedder(nn.Module):
    def __init__(self,num_classes,embed_dim,layers,dropout=0.3):
        super(Embedder,self).__init__()
        self.conv = CNN3D(num_classes)
        self.emb = nn.LSTM(input_size=num_classes,hidden_size=embed_dim,num_layers=layers,batch_first=True,bidirectional=True,dropout=dropout)

    def forward(self,x):
        x = self.conv(x)
        x = x.view(-1,x.shape[2],x.shape[1])
        x = self.emb(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=400):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super(PositionalEncoder).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        x = x * torch.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super(MultiHeadAttention,self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        scores = torch.matmul(scores, v)
    
        # scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super(FeedForward,self).__init__() 
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Normalizer(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super(Normalizer,self).__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True))/ (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Normalizer(d_model)
        self.norm_2 = Normalizer(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Normalizer(d_model)
        self.norm_2 = Normalizer(d_model)
        self.norm_3 = Normalizer(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super(Encoder,self).__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model,N)
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model,heads) for i in range(N)])
        self.norm = Normalizer(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x[0])
        for i in range(N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super(Decoder,self).__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model,N)
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model,heads) for i in range(N)])
        self.norm = Normalizer(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output