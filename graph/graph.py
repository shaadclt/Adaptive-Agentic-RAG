from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    
    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"   
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE
    
def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("--CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    score = hallucination_grader.invoke(
        {"documents":documents, "generation": generation}
    )
    
    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
    
def route_question(state: GraphState) -> str:
    """
    Route question to web search or RAG
    
    Args:
        state (dict): The current graph state
        
    Returns:
        str: Next node to call
    """
    
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE
    
workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE
    },
)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    }
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "useful": END,
        "not useful": WEBSEARCH,
        "not supported": GENERATE,
    }
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.to_fastapi_app()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")


"""
This is the heart of the Agentic RAG system, where we orchestrate all components into a cohesive workflow.
The graph starts with a conditional entry point that routes questions to either web search or local retrieval based on the router's decision.
The workflow includes three key decision functions that implement our adaptive logic. 
The decide_to_generate function determines whether to proceed with generation or search the web based on document relevance scores.
The grade_generation_grounded_in_documents_and_question function implements our self-correction mechanism by checking for hallucinations and answer relevance, potentially triggering regeneration or additional search.
The route_question function handles initial query routing.
The compiled workflow creates a robust system that can adapt its behavior based on the quality of information at each step, ensuring reliable and comprehensive responses.
"""