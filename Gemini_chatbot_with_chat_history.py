import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, MessagesState

st.write('<h1 style="text-align: center; color: blue;">AI Chatbot</h1>', unsafe_allow_html=True)

LANGCHAIN_API_KEY = st.secrets["LANGCHAIN_API_KEY"]
LANGCHAIN_PROJECT = st.secrets["LANGCHAIN_PROJECT"]
LANGCHAIN_ENDPOINT = st.secrets["LANGCHAIN_ENDPOINT"]
LANGCHAIN_TRACING_V2 = st.secrets["LANGCHAIN_TRACING_V2"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {"messages": response}

workflow.add_edge(START, 'model')
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}
parser = StrOutputParser()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    # Preload with a system message introducing the bot and mentioning its creator
    st.session_state["chat_history"] = []

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)
        
chat = st.chat_input("Enter your message:")
if chat:
    st.session_state["chat_history"].append(HumanMessage(str(chat)))

    # Prepare the message state
    state = MessagesState(messages=st.session_state["chat_history"])
    result = app.invoke({"messages": state["messages"]}, config=config)
    response_message = result["messages"][-1]
    response_text = parser.invoke(response_message)
    st.chat_message("human").write_stream(stream_data(chat))
    st.text(st.session_state['chat_history']['HumanMessage'])
    with st.spinner("Generating...."):
        st.chat_message("ai").write_stream(stream_data(response_text))
        st.session_state["chat_history"].append(response_message)
else:
    st.chat_message("ai").write("Hello! How can I help you today?")
    st.warning("Please enter a message")
