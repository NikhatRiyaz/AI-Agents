# app.py
import os
from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
import streamlit as st

# ---------------- ENV ----------------
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# ---------------- STATE ----------------
class AgentState(BaseModel):
    messages: list[str]
    reply: str | None = None

# ---------------- LLM ----------------
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.3
)

# ---------------- AGENT ----------------
def agent_reply(state: AgentState):
    prompt = f"""
    You are a normal user replying to a suspicious message.
    Message: {state.messages[-1]}
    Reply naturally.
    """
    state.reply = llm.invoke(prompt).content
    return state

# ---------------- GRAPH ----------------
graph = StateGraph(AgentState)
graph.add_node("reply", agent_reply)
graph.set_entry_point("reply")
app_graph = graph.compile()

# ---------------- STREAMLIT UI ----------------
st.title("🕵️ Agentic Scam Honeypot")
st.write("Type a suspicious message and see how the agent would reply.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "replies" not in st.session_state:
    st.session_state.replies = []

# Input box
msg = st.text_input("Incoming message")

if st.button("Send") and msg.strip():
    st.session_state.messages.append(msg)

    state = AgentState(messages=st.session_state.messages)
    result = app_graph.invoke(state)   # ✅ Use invoke(), not _
    
    st.session_state.replies.append(result.reply)
