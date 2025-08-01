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

structured_llm_router = llm.with_structured_prompt(RouteQuery)

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