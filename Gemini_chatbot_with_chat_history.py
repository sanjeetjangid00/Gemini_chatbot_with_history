import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, MessagesState

# Title
st.write('<h1 style="text-align: center; color: blue;">AI Chatbot</h1>', unsafe_allow_html=True)

# Load secrets
LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
LANGCHAIN_PROJECT = st.secrets["LANGCHAIN_PROJECT"]
LANGCHAIN_ENDPOINT = st.secrets["LANGCHAIN_ENDPOINT"]
LANGCHAIN_TRACING_V2 = st.secrets["LANGCHAIN_TRACING_V2"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Initialize model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define model call function
def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {"messages": state['messages'] + [response]}

# Build LangGraph workflow
workflow = StateGraph(state_schema=MessagesState)
workflow.add_node("model", call_model)
workflow.set_entry_point("model")

# Memory checkpoint
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Output parser
parser = StrOutputParser()
config = {"configurable": {"thread_id": "abc123"}}

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
chat = st.chat_input("Enter your message:")

if chat:
    # Show user message
    with st.chat_message("user"):
        st.markdown(chat)

    # Add to history
    st.session_state["chat_history"].append({"role": "user", "content": chat})

    # Build LangChain message objects (HumanMessage / AIMessage)
    langchain_messages = []
    for m in st.session_state["chat_history"]:
        if m["role"] == "user":
            langchain_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            langchain_messages.append(AIMessage(content=m["content"]))

    # Prepare state and invoke model
    state = MessagesState(messages=langchain_messages)
    result = app.invoke({"messages": state["messages"]}, config=config)

    # Parse response
    response_message = result["messages"][-1]
    response_text = parser.invoke(response_message)

    # Show and store assistant message
    with st.chat_message("assistant"):
        st.markdown(response_text)

    st.session_state["chat_history"].append({"role": "assistant", "content": response_text})

elif not st.session_state["chat_history"]:
    st.chat_message("assistant").write("Hello! How can I help you today?")
