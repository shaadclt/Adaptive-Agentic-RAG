from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    generation = generation_chain.invoke({"context":documents,"question":question})
    
    return {"documents": documents,"question" :question, "generation": generation }

"""
The generation node is where our system creates the actual response to the user's question. 
It takes the user's question and all gathered documents (both from local retrieval and web search) and uses our generation chain to create a comprehensive answer. 
The node maintains all the context information in the state while adding the generated response, enabling subsequent nodes to evaluate the quality and accuracy of the generation.
"""