import torch
import CSR
import time

num_classes = 40

emb = CSR.Embedder(40,40,2)
trans = torch.nn.Transformer(80,nhead=8,num_encoder_layers=20)

x = torch.randn(1,3,100,256,256)
with torch.no_grad():
    s = time.time()
    x = emb(x)
    x = trans(src=x[0],tgt=torch.rand(1,100,80))
    e = time.time()
print(x[0].dtype,x[0].shape)
print(e-s)
print(x[0][0])