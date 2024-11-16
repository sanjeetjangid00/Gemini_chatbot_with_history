import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize the model
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

# Set up the Streamlit UI
st.title(":blue[Gemini Chatbot]")

# Define the graph
workflow = StateGraph(state_schema=MessagesState)
config = {"configurable": {"thread_id": "abc123"}}

# Define the function that calls the model
def call_model(state: MessagesState):
    try:
        response = model.invoke(state["messages"])
        return {"messages": response}
    except Exception as e:
        st.error(f"Model invocation error: {str(e)}")
        return {"messages": [HumanMessage("An error occurred while processing your request.")]}

# Add node and edges to the workflow
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Initialize memory saver
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Initialize output parser
parser = StrOutputParser()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# User input using chat_input (for Streamlit 1.27+)
query = st.chat_input("Enter Your Query:")

# Process user input
if query and query.strip():
    # Add user message to chat history
    st.session_state["chat_history"].append(HumanMessage(str(query)))
    st.chat_message("user").write(query)

    # Prepare the message state
    state = MessagesState(messages=st.session_state["chat_history"])

    # Get the response from the model
    try:
        result = app.invoke({"messages": state["messages"]}, config=config)
        response_message = result["messages"][-1]
        response_text = parser.invoke(response_message)
        st.chat_message("assistant").write(response_message)
    except Exception as e:
        response_text = "An error occurred while processing your request."
        response_message = HumanMessage(response_text)

    # Add model response to chat history
    st.session_state["chat_history"].append(response_message)

    # Display the entire chat history
    """for msg in st.session_state["chat_history"]:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)
"""
