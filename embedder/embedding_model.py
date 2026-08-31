import torch
from transformers import AutoTokenizer, AutoModel

#Class that will be used for embedding operations
class Embedder:
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        self.model =  AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
        self.model.eval() #We turn on the evaluation mode, as this model here we are using
        #is already pretrained
        
    def tokenize(self, text):
        return self.tokenizer(
            text,
            return_tensors = "pt" #returns the tensors in pytorch format as most modals utilize it
        )
    
    def encode(self, inputs):
        with torch.no_grad():  #Saves GPU Memory storing gradients requires a lot of memory. Turning them off cuts memory usage by up to 50%
            return self.model(**inputs)
        
        



