from typing import Any, Dict
from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    
    documents = retriever.invoke(question)
    return {"documents": documents,"question" :question}


"""
The retrieve node is our first workflow step that fetches relevant documents from our local vector store. 
It takes the user's question from the state and uses our pre-configured retriever to find the most semantically similar documents.
The function returns both the retrieved documents and the original question, updating the state for the next nodes in the workflow. 
This node represents the traditional RAG retrieval step but is enhanced by our subsequent grading and decision-making processes.
"""