from typing import Any, Dict
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    
    documents = state.get("documents", [])
    
    tavily_results = web_search_tool.invoke({"query":question})["results"]
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    
    # Add web results to existing documents 
    if documents:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    return {"documents": documents,"question" :question}

if __name__ == "__main__":
    web_search(state={"question":"agent memory","documents":None})
    
"""
The web search node extends the system's knowledge beyond the local vector store by querying the internet for current and comprehensive information. 
It uses Tavily, a search API optimized for AI applications, to find the most relevant web results.
The node retrieves up to 3 results, combines their content into a single document, and adds it to our document collection. 
This hybrid approach ensures that our system can handle both domain-specific queries using local knowledge and general queries requiring current information from the web.
"""