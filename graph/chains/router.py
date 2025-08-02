from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from model import llm_model

class RouteQuery(BaseModel):
    """
    Route a user query to the most relevant datasource
    """
    datasource: Literal["vectorstore", "websearch"] = Field(..., 
        description="Given a user question choose to route it to web search or a vectorstore."),

llm = llm_model

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """
    You are an expert at routing a user question to a vectorstore or websearch.
    The vectorstore contains documents related to agents, prompt engineering and adversarial attacks.
    Use the vectorstore for questions on these topics. For all other questions use websearch.
"""
route_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{question}"),
])

question_router = route_prompt | structured_llm_router


"""
The query router is the system's first decision point that determines the optimal source for answering a user's question. 
We define a RouteQuery Pydantic model that constrains the router's output to either "vectorstore" or "websearch", ensuring reliable parsing of the LLM's decision.
The system prompt clearly defines the scope of our local knowledge base, instructing the router to use the vectorstore for questions about agents, prompt engineering, and adversarial attacks, while routing everything else to web search.
This intelligent routing prevents unnecessary web searches for topics we have comprehensive local knowledge about.
"""    