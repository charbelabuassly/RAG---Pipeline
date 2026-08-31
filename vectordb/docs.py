from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class QdrantDB:

    def __init__(self):
        #Creates the connection
        self.client = QdrantClient(
            host="localhost",
            port=6333
        )
        
    def create_collection(self):
        if not self.client.collection_exists("rag_documents"):
            self.client.create_collection(
                collection_name="rag_documents",
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
        )
        
        
    def add(self, point_id, vector, payload): #Adds data into the vector db rag_documents collection. Each data point is called point
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )

        self.client.upsert(
            collection_name="rag_documents",
            points=[point]
        )
