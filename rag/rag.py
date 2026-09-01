from retriever.retriever import Retriever
from generator.generator import QwenGenerator


class RAG:

    def __init__(self, model_name):
        self.retriever = Retriever() #Retrievers relevant vectors
        self.generator = QwenGenerator(model_name) #Talks with AI directly

    def ask(self, query, limit=3):
        # Retrieve relevant chunks
        results = self.retriever.retrieve(
            query,
            limit=limit
        )

        # Extract text from Qdrant payloads
        context_parts = []
        for result in results:
            if result.payload and "text" in result.payload:
                context_parts.append(
                    result.payload["text"]
                )

        context = "\n\n".join(context_parts)

        # Build the RAG prompt
        prompt = f"""
            Use the following context to answer the question.
            Context:
            {context}
            Question:
            {query}
            Answer:
            """

        # Give the prompt to the Generator
        return self.generator.generate(prompt)