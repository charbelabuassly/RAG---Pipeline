#This will split texts based on the tokens, it will have an overlap value of 50 tokens
#to ensure chunks retain information from where they came from
from transformers import AutoTokenizer

class Chunker:

    def __init__(self, tokenizer, chunk_size=400, overlap=50):

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text):

        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

        chunks = []
        step = self.chunk_size - self.overlap
        
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(
                chunk_tokens
            )
            chunks.append(chunk_text)
        return chunks