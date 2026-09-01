from vectordb.docs import QdrantDB
from embedder.embedding_model import Embedder


class Retriever:

    def __init__(self):
        self.embedder = Embedder() #Used to embed incoming prompts
        self.db = QdrantDB() #Used to retrieve from the db
        
    def retrieve(self, query, limit=3): 
        #This limit is the top k chunks
        inputs = self.embedder.tokenize(query) #Tokenizing the input
        output = self.embedder.encode(inputs) #Embedding it
        #The BGE model first produces contextual semantic representations for each token
        #Mean pooling then collapses those multiple token representations into one 384-dimensional representation for the entire piece of text
        vector = self.embedder.mean_pool(
            output,
            inputs["attention_mask"] #To avoid including the padding in the mean calculations, as 0 = padding & 1 = real token
        )
        
        results = self.db.search(
        vector,
        limit
    )

        return results
        