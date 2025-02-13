# # Mini Project 2 - Part 2: Building a Chatbot with OpenAI's GPT-3.5-turbo Model (50 pts)

# ## Goal

# The goal of this assignment is to design a chatbot using OpenAI's GPT-3.5-turbo model and the Streamlit library in Python. The chatbot should be able to maintain a conversation history and generate responses to user inputs.

# Import the necessary libraries
import streamlit as st
from openai import OpenAI  # TODO: Install the OpenAI library using pip install openai

st.title("Mini Project 2: Streamlit Chatbot")

# TODO: Replace with your actual OpenAI API key
openai_key_file = "openai_key.txt"
with open(openai_key_file, "r") as f:
    openai_key = f.read().strip()
client = OpenAI(api_key=openai_key)

# Define a function to get the conversation history (Not required for Part-2, will be useful in Part-3)
def get_conversation() -> str:
    # return: A formatted string representation of the conversation.
    # ... (code for getting conversation history)
    pass

# Check for existing session state variables
if "openai_model" not in st.session_state:
    # ... (initialize model)
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    # ... (initialize messages)
    st.session_state["messages"] = []

# Display existing chat messages
# ... (code for displaying messages)
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Wait for user input
if prompt := st.chat_input("What would you like to chat about?"):
    # ... (append user message to messages)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # ... (display user message)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    response = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=st.session_state["messages"],
    ).choices[0].message.content

    # ... (append AI response to messages)
    st.session_state["messages"].append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)

