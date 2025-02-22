from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import pandas as pd
import string
import time

class DocumentProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.loader = PyMuPDFLoader(file_path)
        self.docs = self.loader.load()
        self.page_texts = []
        self.page_numbers = []
        self._extract_pages()
    
    def _extract_pages(self):
        for doc in self.docs:
            self.page_texts.append(doc.page_content)
            self.page_numbers.append(doc.metadata["page"])

class TextChunker:
    def __init__(self, chunk_size=2500, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_text(self, page_texts, page_numbers):
        chunked_texts, chunk_page_numbers = [], []
        previous_page_tail = ""
        
        for page_text, page_number in zip(page_texts, page_numbers):
            page_text = previous_page_tail + page_text
            chunks = self.text_splitter.split_text(page_text)
            chunked_texts.extend(chunks)
            chunk_page_numbers.extend([page_number] * len(chunks))
            previous_page_tail = page_text[-self.text_splitter._chunk_overlap:]
        
        return chunked_texts, chunk_page_numbers

class EmbeddingGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding
    
    def create_dataframe(self, chunked_texts, chunk_page_numbers):
        df = pd.DataFrame({"text": chunked_texts, "page_number": chunk_page_numbers})
        df["text"] = df["text"].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)).replace("\n", " "))
        df["embeddings"] = df["text"].apply(self.get_embedding)
        return df

class PineconeManager:
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)
    
    def create_index(self, index_name):
        # Check if index already exists
        try:
            existing_index = self.pc.describe_index(index_name)
            print(f"Index '{index_name}' already exists, using existing index.")
            return self.pc.Index(index_name)
        except Exception:
            # Create new index if it doesn't exist
            self.pc.create_index(
                name=index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            
            return self.pc.Index(index_name)
    
    def insert_embeddings(self, index, df, batch_size=100):
        records = []
        for i, row in df.iterrows():
            record = {
                "id": f"vec{i}",
                "values": row["embeddings"],
                "metadata": {
                    "text": row["text"],
                    "page_number": row["page_number"]
                }
            }
            records.append(record)
            
            if len(records) >= batch_size:
                index.upsert(records)
                records = []
        
        if records:
            index.upsert(records)

class QueryEngine:
    def __init__(self, vector_store, openai_client):
        self.vector_store = vector_store
        self.client = openai_client
    
    def search(self, query, top_k=5):
        return self.vector_store.similarity_search(query, k=top_k)
    
    def get_ai_response(self, query, context, model="gpt-3.5-turbo"):
        prompt = f"""
            You are an AI assistant answering questions strictly based on the provided book context.
            If the question cannot be answered from the context, say it is not relevant.

            Context: {context}

            Question: {query}
            Answer:
        """
        message = {"role": "user", "content": prompt}
        response = self.client.chat.completions.create(
            model=model,
            messages=[message]
        )
        return response.choices[0].message.content

def main():
    # Initialize components
    doc_processor = DocumentProcessor("./machine_learning.pdf")
    openai_key_file = "./openai_key.txt"
    Pinecone_api_key_file = "./pinecone_api_key.txt"
    with open(openai_key_file, "r") as file:
        openai_key = file.read()
    with open(Pinecone_api_key_file, "r") as file:
        Pinecone_api_key = file.read()
    chunker = TextChunker(chunk_size=2500, chunk_overlap=50)
    embedding_gen = EmbeddingGenerator(openai_key)
    pinecone_manager = PineconeManager(Pinecone_api_key)
    
    # Process document
    chunked_texts, chunk_page_numbers = chunker.split_text(doc_processor.page_texts, doc_processor.page_numbers)
    df = embedding_gen.create_dataframe(chunked_texts, chunk_page_numbers)
    
    # Setup Pinecone
    index = pinecone_manager.create_index("eep596-index")
    pinecone_manager.insert_embeddings(index, df)
    
    # Setup query engine
    vector_store = PineconeVectorStore(index, OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key))
    client = OpenAI(api_key=openai_key)
    query_engine = QueryEngine(vector_store, client)
    
    # Example query
    query = "What is bias-variance tradeoff?"
    results = query_engine.search(query)
    context = "\n".join([result.page_content for result in results])
    response = query_engine.get_ai_response(query, context)
    print(response)

if __name__ == "__main__":
    main()