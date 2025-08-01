from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from model import llm_model

llm = llm_model

class GradeDocuments(BaseModel):
    """
    Binary score for relevance check on retrieved documents.
    """
    
    binary_score:str = Field(
        description = "Documents are relevant to the question, 'yes' or 'no'"
    )
    
structured_llm_grader = llm.with_structured_prompt(GradeDocuments)