# # Mini Project 2 - Part 2: Building a Chatbot with OpenAI's GPT-3.5-turbo Model (50 pts)

# ## Goal

# The goal of this assignment is to design a chatbot using OpenAI's GPT-3.5-turbo model and the Streamlit library in Python. The chatbot should be able to maintain a conversation history and generate responses to user inputs.

# ## Action Items

# 1. **Import the necessary libraries**: Import the OpenAI and Streamlit libraries, which are necessary for interacting with the OpenAI API and creating the chat interface, respectively.

# 2. **Initialize the OpenAI client**: Initialize the OpenAI client with your API key using the `OpenAI()` constructor.

# 3. **Initialize the session state**: Check if the "openai_model" and "messages" keys exist in the session state. If they don't, initialize them with the GPT-3.5-turbo model and an empty list, respectively.

# 4. **Display the conversation history**: Iterate over all the messages in the session state and create a chat message for each one with the appropriate role and content.

# 5. **Wait for user input**: Use the `st.chat_input()` function to wait for the user to input a message.

# 6. **Add the user's message to the conversation**: If a message is inputted, add it to the session state's messages list and display it in the chat interface.

# 7. **Generate the assistant's response**: Send a request to the OpenAI API to generate a response from the assistant. This request should include all the previous messages in the conversation.

# 8. **Display the assistant's response**: Add the assistant's response to the session state's messages list and display it in the chat interface.

# 9. **Define a function to get the conversation history**: Define a function `get_conversation()` that iterates over all the messages in the session state and concatenates them into a single string, each message prefixed by the role of the sender. **(Not required for Part-2, will be useful in Part-3)**

# By following these action items, you should be able to create a chat interface where the user can have a conversation with an AI assistant powered by OpenAI's GPT-3.5-turbo model.


# Import the necessary libraries
import streamlit as st
from openai import OpenAI  # TODO: Install the OpenAI library using pip install openai

st.title("Mini Project 2: Streamlit Chatbot")

# TODO: Replace with your actual OpenAI API key
client = OpenAI(api_key='sk-YOUR_API_KEY')

# Define a function to get the conversation history (Not required for Part-2, will be useful in Part-3)
def get_conversation() -> str:
    # return: A formatted string representation of the conversation.
    # ... (code for getting conversation history)

# Check for existing session state variables
if "openai_model" not in st.session_state:
    # ... (initialize model)

if "messages" not in st.session_state:
    # ... (initialize messages)

# Display existing chat messages
# ... (code for displaying messages)

# Wait for user input
if prompt := st.chat_input("What would you like to chat about?"):
    # ... (append user message to messages)

    # ... (display user message)

    # Generate AI response
    with st.chat_message("assistant"):
        # ... (send request to OpenAI API)

        # ... (get AI response and display it)

    # ... (append AI response to messages)
