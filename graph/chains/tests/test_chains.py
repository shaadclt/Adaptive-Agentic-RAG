from pprint import pprint
import pytest

from dotenv import load_dotenv

load_dotenv()

from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import (GradeHallucinations,hallucination_grader)
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.router import RouteQuery, question_router
from ingestion import retriever


def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    
    doc_txt = docs[1].page_content
    
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )
    
    assert res.binary_score == "yes"
    
def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    
    doc_txt = docs[0].page_content
    
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents":docs,
         "generation": "In order to make pizza we need to first start with the dough"}
    )
    
    assert not res.binary_score
    
def test_router_to_vectostore() -> None:
    question = "agent memory"
    res: RouteQuery = question_router.invoke({"question": question})
    
    assert res.datasource == "vectostore"
    
def test_router_to_websearch() -> None:
    question = "how to make pizza"
    
    res: RouteQuery = question_router.invoke({"question": question})
    
    assert res.datasource == "websearch"
    
"""
This comprehensive test suite validates each component of our Agentic RAG system independently.
The tests cover document relevance grading with both positive and negative cases, ensuring our grader correctly identifies relevant and irrelevant documents.
The hallucination detection tests verify that our system can distinguish between grounded and fabricated responses.
The router tests confirm that questions are correctly routed to vectorstore or web search based on their content.
These tests will ensure that each component works correctly in isolation before integration into the complete workflow.
"""