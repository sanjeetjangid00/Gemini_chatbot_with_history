import os
import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, MessagesState

st.write('<h1 style="text-align: center; color: blue;">Gemini Chatbot</h1>', unsafe_allow_html=True)

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model = "gemini-pro")

workflow = StateGraph(state_schema= MessagesState)

def call_model(state:MessagesState):
    response = model.invoke(state['messages'])
    return {"messages":response}

workflow.add_edge(START, 'model')
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer = memory)

config = {"configurable":{"thread_id":"abc123"}}
parser = StrOutputParser()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
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
    with st.spinner("Generating...."):        
        st.chat_message("ai").write_stream(stream_data(response_text))
        st.session_state["chat_history"].append(response_message)
else:
    st.warning("Please enter a message")
