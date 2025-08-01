import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, MessagesState
from langchain_core.messages import HumanMessage

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

# Define graph
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {"messages": state['messages'] + [response]}

workflow.add_node("model", call_model)
workflow.set_entry_point("model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

parser = StrOutputParser()
config = {"configurable": {"thread_id": "abc123"}}

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

# Display chat history
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
chat = st.chat_input("Enter your message:")

if chat:
    # Display user message
    with st.chat_message("user"):
        st.markdown(chat)

    # Add user message to history
    st.session_state["chat_history"].append({"role": "user", "content": chat})

    # Prepare message state for model
    langchain_messages = [HumanMessage(m["content"]) if m["role"] == "user" else m["content"] for m in st.session_state["chat_history"] if m["role"] == "user"]

    state = MessagesState(messages=langchain_messages)
    result = app.invoke({"messages": state["messages"]}, config=config)

    response_message = result["messages"][-1]
    response_text = parser.invoke(response_message)

    # Display and store assistant message
    with st.chat_message("assistant"):
        st.markdown(response_text)

    st.session_state["chat_history"].append({"role": "assistant", "content": response_text})

elif not st.session_state["chat_history"]:
    # Initial greeting
    st.chat_message("assistant").write("Hello! How can I help you today?")
