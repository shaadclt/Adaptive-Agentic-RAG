from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represent the state of our graph
    
    Attributes:
        question:question
        generation: LLM generation
        web_search:whether to add search
        documents:list of documents
    """
    
    question:str
    generation:str
    web_search:bool
    documents:List[str]
    
    
"""
This GraphState class acts as the central data structure that flows through every node in our graph workflow. 
The question field holds the user's input query, generation stores the LLM's response, web_search is a boolean flag that determines whether we need to search the web for additional information, and documents contains all the retrieved documents from both local and web sources.
By using TypedDict, we ensure type safety while maintaining the flexibility needed for our dynamic workflow.
"""