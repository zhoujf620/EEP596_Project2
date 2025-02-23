# Multi-Agent Chatbot Documentation

@Contributors:
- Huapeng Zhou: Implement the multi-agent system structure and the diagram Implement the Router_Agent and Query_Agent
- Junfeng Zhang: Implement the Obnoxious_Agent, Relevant_Documents_Agent, and Answering_Agent, refine and recview the code and report

## Overview
The Multi-Agent Chatbot is a sophisticated system designed to handle machine learning-related queries through a coordinated network of specialized agents. The system uses OpenAI's GPT-4 model and Pinecone vector database for document retrieval and question answering.

## System Architecture

### Head Agent
The `Head_Agent` serves as the main coordinator, managing the interaction between various sub-agents and maintaining conversation history. It initializes all necessary connections and sub-agents.

Key responsibilities:
- Orchestrates the query handling process
- Maintains conversation history
- Manages response modes (precise/chatty)

### Sub-Agents

#### 1. Router Agent
The `Router_Agent` analyzes incoming queries and categorizes them into different types:
- GREETING: Initial greetings or casual conversation
- FOLLOW_UP: Questions referring to previous context
- PROMPT_INJECTION: Potential security threats
- STANDALONE: New, independent questions
- OTHER: Queries that don't fit other categories

#### 2. Obnoxious Agent
The `Obnoxious_Agent` screens queries for inappropriate or hostile content, ensuring safe and respectful interaction.

#### 3. Query Agent
The `Query_Agent` interfaces with the Pinecone vector store to retrieve relevant documents based on the query.

#### 4. Relevant Documents Agent
The `Relevant_Documents_Agent` evaluates whether retrieved documents are actually relevant to the query.

#### 5. Answering Agent
The `Answering_Agent` generates final responses by:
- Analyzing query context and available information
- Determining if context is sufficient
- Generating appropriate responses using GPT-4

## Query Flow
1. User submits a query
2. Obnoxious check is performed
3. Query type is determined
4. Relevant documents are retrieved (if needed)
5. Document relevance is verified
6. Final response is generated
7. Conversation history is updated

## User Interface
The system uses Streamlit for the web interface, providing:
- Chat-like interaction
- Message history display
- Clean input/output interface

## Test Results
![Test Results](./result1.png)
![Test Results 2](./result2.png)

## Diagram
![Diagram](./diagram.png)