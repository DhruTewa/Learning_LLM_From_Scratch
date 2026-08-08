
'''
class ClassName:

    # __init__ = the "birth certificate" step.
    # It runs automatically, once, the instant an object is created.
    # Ask: "What does this object absolutely need before it can function at all?"
    # Whatever the answer is -> becomes a parameter here.
    
    def __init__(self, required_thing):

        # self = "me, this specific object" — not any other object made from this same class.
        # Ask: "Will another method need this again later?" If yes -> store it in self.
        # If it's only used once, right here, it doesn't need self at all.
        
        self.some_attribute = required_thing

        # You can also build something derived from the input here,
        # ONCE, at birth — so later methods don't have to rebuild it every time they run.
        
        self.derived_attribute = some_transformation(required_thing)

    # A regular method = an action this object can perform, on demand, later.
    # It reaches into self to use what it already remembers —
    # it does NOT need to be told self_attribute again, it already has it.
    
    def do_something(self, new_input):
        result = self.some_attribute  # using what was remembered at birth
        return result
        
        '''
import re
import os

class SimpleTokenizer:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
        
    def encoder(self,text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])  
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
file_path = os.path.join(os.path.dirname(__file__), "the-verdict.txt")  
with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab_size = len(all_tokens)

vocab = {token:integer for integer, token in enumerate(all_tokens)}
print(vocab)
    
tokenizer = SimpleTokenizer(vocab=vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
id =tokenizer.encoder(text=text)
print(id)
print(tokenizer.decode(ids=id))
        