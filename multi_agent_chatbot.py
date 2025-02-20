import openai
import pinecone
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
import time
# Initialize Pinecone
pinecone_api_key_file = "./pinecone_api_key.txt"
openai_api_key_file = "./openai_key.txt"
with open(pinecone_api_key_file, "r") as f:
    pinecone_api_key = f.read().strip()
with open(openai_api_key_file, "r") as f:
    openai_api_key = f.read().strip()
pc = pinecone.Pinecone(api_key=pinecone_api_key)  # 替换为你的 API Key

def create_pinecone_index(index_name):
    try:
        # Check if index already exists
        existing_indexes = pc.list_indexes()
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name, 
                dimension=1536,  # text-embedding-3-small
                metric="cosine", 
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

            # Wait for index to be ready
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
    except Exception as e:
        print(f"Error creating/accessing index: {e}")
        
    # Get the index instance whether it was just created or already existed
    pinecone_index = pc.Index(index_name)
    return pinecone_index

pinecone_index = create_pinecone_index("ml-index-2500")

class Obnoxious_Agent:
    def __init__(self, mode='chatty'):
        self.mode = mode
    
    def is_obnoxious(self, query: str) -> str:
        # Simple rule-based approach to detect obnoxious content
        obnoxious_words = ["stupid", "idiot", "dumb", "hate"]
        if any(word in query.lower() for word in obnoxious_words):
            return "Yes"
        return "No"

class Relevant_Documents_Agent:
    def __init__(self, mode='chatty'):
        self.mode = mode
    
    def retrieve_documents(self, query: str) -> List[str]:
        # Simulated relevant document retrieval
        documents = {
            "machine learning": ["Intro to ML", "Supervised Learning", "Neural Networks"],
            "history": ["World War II", "French Revolution", "Cold War"]
        }
        return documents.get(query.lower(), [])

class Query_Agent:
    def __init__(self, mode='chatty'):
        self.mode = mode
    
    def query_pinecone(self, query: str) -> List[str]:
        response = pinecone_index.query(
            vector=[0.1] * 1536,  # You need to generate proper embeddings here
            top_k=3,
            include_metadata=True
        )
        return [match["metadata"]["text"] for match in response["matches"]]

class Answering_Agent:
    def __init__(self, mode='chatty'):
        self.mode = mode
        self.client = openai.OpenAI(api_key=openai_api_key)
        
    def generate_response(self, query: str, documents: List[str]) -> str:
        system_prompt = """You are a helpful AI assistant specialized in machine learning topics. 
        Follow these rules:
        1. If the documents contain relevant information, use it to answer the question.
        2. If the documents don't contain the specific information but the question is about ML, provide a brief general answer and acknowledge that the documents don't cover this specific topic.
        3. If the question is a follow-up to a previous topic, maintain context and provide a coherent response.
        4. Always be clear about what information comes from the documents and what is general knowledge.
        5. If appropriate, encourage further specific questions about the topics that are covered in the documents."""
        
        # Include conversation history for context
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # Add conversation history if available
        if hasattr(st.session_state, 'messages'):
            # Get last few messages for context (limiting to last 4 exchanges)
            recent_messages = st.session_state.messages[-8:]
            messages.extend([
                {"role": msg["role"], "content": msg["content"]}
                for msg in recent_messages
            ])
        
        # Add current query and context
        messages.append({
            "role": "user", 
            "content": f"Context documents: {documents}\nUser question: {query}"
        })
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return response.choices[0].message.content

class Head_Agent:
    def __init__(self, mode='chatty'):
        self.mode = mode
        self.query_agent = Query_Agent(mode)
        self.answering_agent = Answering_Agent(mode)
        
    def handle_query(self, user_input: str) -> str:
        # Handle greetings
        greetings = ['hello', 'hi', 'hey', 'greetings']
        if user_input.lower().strip() in greetings:
            return "Hello! How can I assist you today?"
            
        # Check for offensive content
        offensive_words = ['dumb', 'stupid', 'idiot']
        if any(word in user_input.lower() for word in offensive_words):
            return "Please do not ask obnoxious questions."
        
        # Check if it's a follow-up question
        is_followup = False
        if hasattr(st.session_state, 'messages') and len(st.session_state.messages) > 0:
            last_few_messages = st.session_state.messages[-4:]
            followup_indicators = ['that', 'it', 'this', 'these', 'those', 'the']
            is_followup = any(indicator in user_input.lower() for indicator in followup_indicators)
            
        # Query Pinecone for relevant documents
        pinecone_docs = self.query_agent.query_pinecone(user_input)
        
        # If no relevant documents found but it's a follow-up
        if not pinecone_docs and is_followup:
            # Try to query using the context from previous messages
            previous_query = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else ""
            pinecone_docs = self.query_agent.query_pinecone(previous_query + " " + user_input)
            
        # Generate response using the answering agent
        return self.answering_agent.generate_response(user_input, pinecone_docs)

# Streamlit Integration
import streamlit as st

def chatbot_interface():
    st.title("Multi-Agent Chatbot")
    
    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "head_agent" not in st.session_state:
        st.session_state.head_agent = Head_Agent(mode='chatty')

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle user input
    if user_input := st.chat_input("Ask me anything about machine learning:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Get bot response
        response = st.session_state.head_agent.handle_query(user_input)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    chatbot_interface()
