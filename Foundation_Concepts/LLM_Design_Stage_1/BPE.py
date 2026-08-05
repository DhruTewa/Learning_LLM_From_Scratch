from importlib.metadata import version
import tiktoken
print("tiktoken version:",version("tiktoken"))

tokenizer = tiktoken.get_encoding('gpt2')
# text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces""of someunknownPlace.")
text2 = "Akwirw ier"
integers = tokenizer.encode(text=text2,allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)

print("-" * 40)
for token_id in integers:
    print(token_id, "->", tokenizer.decode([token_id]))
    
s