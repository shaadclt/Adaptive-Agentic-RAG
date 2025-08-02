from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from model import llm_model

llm = llm_model

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()


"""
The generation chain is responsible for creating the actual response to the user's question. 
We leverage a proven RAG prompt from LangChain Hub that has been optimized for retrieval-augmented generation tasks. 
This prompt template knows how to effectively combine retrieved context with the user's question to generate coherent, informative responses.
The chain uses StrOutputParser ensures we get clean string output that can be easily processed by subsequent components.    
"""