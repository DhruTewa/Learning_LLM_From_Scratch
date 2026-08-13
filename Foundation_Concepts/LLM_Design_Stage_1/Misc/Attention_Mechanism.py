import torch
import torch.nn as nn

# implementation of Simplified Self-Attention

inputs = torch.tensor(
    [[0.43,0.15,0.89], # Your,
     [0.55,0.87,0.66], # Journey,
     [0.57,0.85,0.64], # Start
     [0.22,0.58,0.33], # with
     [0.77,0.25,0.10], # single
     [0.05,0.80,0.55]] # step
     )

attn_scores = torch.empty(6,6)
attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores,dim=-1)
all_context_vecs = attn_weights @inputs
print(all_context_vecs)

# Implementation of Self attention with trainable weights

x_2  = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in,d_out,requires_grad=False))
W_key = torch.nn.Parameter(torch.rand(d_in,d_out,requires_grad=False))
W_value = torch.nn.Parameter(torch.rand(d_in,d_out,requires_grad=False))

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2)


class SelfAttention_v2(nn.Module):
    
    def __init__(self,d_in,d_out,qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        
    def forward(self,x):
        keys = self.W_key(x)
        queries =self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weight = torch.softmax(attn_scores/keys.shape[-1]**0.5,dim = -1)
        context_vec = attn_weight @ values
        
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in=d_in,d_out=d_out)
print(sa_v2(inputs))
