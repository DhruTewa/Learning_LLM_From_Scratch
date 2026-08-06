from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset,DataLoader
print("tiktoken version:",version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")
# # text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces""of someunknownPlace.")
# text2 = "Akwirw ier"
# integers = tokenizer.encode(text=text2,allowed_special={"<|endoftext|>"})
# print(integers)
# strings = tokenizer.decode(integers)
# print(strings)

# print("-" * 40)
# for token_id in integers:
#     print(token_id, "->", tokenizer.decode([token_id]))
    
#Data Sampling with sliding window

with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()

# enc_txt = tokenizer.encode(raw_text)
# print(len(enc_txt))

# # Taking the sample from the raw text:
# enc_sample = enc_txt[50:]
# context_size = 4
# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]
# print(f"x:{x}")
# print(f"y:  {y}")
# for i in range(1,context_size+1):
#     context =enc_sample[:i]
#     desired = enc_sample[i]
#     print(context,"-->",desired)
    
# for i in range(1,context_size+1):
#     context =enc_sample[:i]
#     desired = enc_sample[i]
#     print(tokenizer.decode(context),"-->",tokenizer.decode([desired]))
    
class GPTDataset_v1(Dataset):
    def __init__(self,text,tokenizer,max_length,stride):
        self.input_ids =[]
        self.target_ids=[]
        token_ids = tokenizer.encode(text)
        
        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index],self.target_ids[index]
    
    
def create_dataloader_v1(txt,batch_size =4,max_length = 256,
                         stride =128,shuffle = True,drop_last = True,num_worker  = 0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataset_v1(txt,tokenizer,max_length,stride)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_worker)
    return dataloader

dataloader = create_dataloader_v1(raw_text,batch_size=8,max_length=5,stride=5,shuffle=False)
data_iter = iter(dataloader)
inputs,targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)
       

