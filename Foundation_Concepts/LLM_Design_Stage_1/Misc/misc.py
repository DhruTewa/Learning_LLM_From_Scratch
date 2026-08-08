from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset,DataLoader

vocab_size = 6 # 6 word vocab
output_dim = 3 # embedding size is 3
input_id = torch.tensor([2,3,5,1])

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
print(embedding_layer.weight)
print("-"* 80)
print(embedding_layer(input_id))