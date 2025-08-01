from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from model import llm_model

llm = llm_model

class GradeHallucinations(BaseModel):
    """
    Binary score for hallucination present in the generated answer.
    """
    
    binary_score:bool = Field(
        description = "Answer is grounded in the facts, 'yes' or 'no'"
    )