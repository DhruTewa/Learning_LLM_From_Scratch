from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset,DataLoader

# basic attention understanding

import torch
inputs = torch.tensor(
    [[0.43,0.15,0.89], # Your,
     [0.55,0.87,0.66], # Journey,
     [0.57,0.85,0.64], # Start
     [0.22,0.58,0.33], # with
     [0.77,0.25,0.10], # single
     [0.05,0.80,0.55]] # step
     )
query = inputs[1]
attn_score_2 = torch.empty(inputs.shape[0])
for i,x_i in enumerate(inputs):
    attn_score_2[i] = torch.dot(x_i,query)
print(attn_score_2)    
#print(query)
attn_weight_2 = torch.softmax(attn_score_2,dim=0)
print("Attention wegihts:",attn_weight_2)
print("Sum:", attn_weight_2.sum())
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weight_2[i]*x_i
print(context_vec_2)