import os
import urllib.request
import re

url = ("https://raw.githubusercontent.com/rasbt/"
"LLMs-from-scratch/main/ch02/01_main-chapter-code/"
"the-verdict.txt")
file_path = os.path.join(os.path.dirname(__file__), "the-verdict.txt")
urllib.request.urlretrieve(url,file_path)

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()
    print("-" * 40)
    print("Total number of character:", len(raw_text))
    print("-" * 40)
    print("First 99 characters:", raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print("-" * 40)
print("Total number of tokens:", len(preprocessed))
print("-" * 40)
print("First 30 tokens:", preprocessed[:30])

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print("-" * 40)
print("vocab size is:", vocab_size)

vocab = {token:integer for integer, token in enumerate(all_words)}
print(vocab)