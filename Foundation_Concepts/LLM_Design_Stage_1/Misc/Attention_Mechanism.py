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

# Masking the attention 

queries =sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5,dim =-1)
print(attn_weights)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length,context_length))
print(mask_simple)

mask_simple = attn_weights * mask_simple
print(mask_simple)


class CausalAttention(nn.Module):
    
    def __init__(self,d_in,d_out,context_length,dropout, qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))
        
    def forward(self,x):
        b, num_tokens,d_in = x.shape
        keys = self.W_key(x)
        queries =self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5,dim = -1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values

        return context_vec

# stack two copies of the same sentence to simulate a batch of 2 sequences
batch = torch.stack((inputs, inputs), dim=0)

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)


# Adding multihead wrapper
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,dropout, num_heads, qkv_bias=False):
        super().__init__()
        self. heads = nn.ModuleList([CausalAttention(d_in, d_out, 
                                                     context_length, dropout, qkv_bias) 
                                     for _ in range(num_heads)])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0,num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
    context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer( "mask",torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view( b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec
    

    
    