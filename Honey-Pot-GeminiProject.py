#!/usr/bin/env python
# coding: utf-8

# ## Agentic Scam-Detection Workflow
# 
# - Input Agent – validates text & language
# - Intent Analysis Agent – checks scam / synthetic patterns
# - Decision Agent – final classification
# - Explanation Agent (Gemini) – explains reasoning
# - Orchestrator Agent – controls flow

# ## Setup Gemini

# In[4]:


# import and load environment variables
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-flash-lite-latest")


# ## Supported Languages

# In[5]:


SUPPORTED_LANGUAGES = {
    "English", "Hindi", "Tamil", "Telugu", "Malayalam"
}


# ## Import libraries

# In[6]:


# Importing necessary libraries
from typing import Optional, Dict, List
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END


# ## Define Agent State

# In[7]:


class DetectionState(BaseModel):
    language: str
    text: str

    features: Optional[Dict] = None
    classification: Optional[str] = None
    confidenceScore: Optional[float] = None
    explanation: Optional[str] = None


# ## Validation Agent

# In[8]:


def validation_agent(state: DetectionState) -> DetectionState:
    if state.language not in SUPPORTED_LANGUAGES:
        raise ValueError("Unsupported language")

    if len(state.text.strip()) < 10:
        raise ValueError("Text too short")

    return state


# ## Intent Analysis Agent Node:

# In[9]:


SCAM_KEYWORDS = [
    "urgent", "verify", "account", "blocked",
    "click", "otp", "upi", "bank",
    "limited time", "act now"
]

def analysis_agent(state: DetectionState) -> DetectionState:
    text = state.text.lower()

    keyword_hits = sum(1 for k in SCAM_KEYWORDS if k in text)

    features = {
        "keyword_hits": keyword_hits,
        "excessive_exclamations": state.text.count("!") > 3,
        "repeated_cta": text.count("click") >= 2
    }

    state.features = features
    return state


# ## Decision Agent Node:

# In[10]:


def decision_agent(state: DetectionState) -> DetectionState:
    f = state.features

    score = (
        f["keyword_hits"] * 0.2
        + (0.3 if f["excessive_exclamations"] else 0)
        + (0.3 if f["repeated_cta"] else 0)
    )

    score = min(score, 1.0)

    state.classification = (
        "AI_GENERATED_SCAM" if score >= 0.6 else "HUMAN"
    )
    state.confidenceScore = round(score, 2)

    return state


# ## Gemini Explanation Agent 

# In[11]:


llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0.2
)

def explanation_agent(state: DetectionState) -> DetectionState:
    prompt = f"""
    Analyze the following message written in {state.language}:

    "{state.text}"

    The system classified it as {state.classification}
    with confidence {state.confidenceScore}.

    Explain briefly and clearly why this classification makes sense.
    """

    response = llm.invoke(prompt)
    state.explanation = response.content.strip()

    return state


# ## Build the Graph

# In[12]:


graph = StateGraph(DetectionState)

graph.add_node("validate", validation_agent)
graph.add_node("analyze", analysis_agent)
graph.add_node("decide", decision_agent)
graph.add_node("explain", explanation_agent)

graph.set_entry_point("validate")

graph.add_edge("validate", "analyze")
graph.add_edge("analyze", "decide")
graph.add_edge("decide", "explain")
graph.add_edge("explain", END)

app = graph.compile()


# ## Run the Agent Graph

# In[13]:


input_state = DetectionState(
    language="English",
    text="Urgent! Your bank account will be blocked. Click now to verify your UPI!!!"
)

result = app.invoke(input_state)
result


# ## Final Output(API Ready)

# 

# In[14]:


final_response = {
    "status": "success",
    "language": result["language"],
    "classification": result["classification"],
    "confidenceScore": result["confidenceScore"],
    "explanation": result["explanation"]
}

final_response

