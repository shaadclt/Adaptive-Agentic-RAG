from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from model import llm_model

llm = llm_model

class GradeAnswer(BaseModel):
    
    binary_score: bool = Field(
        description = "Answer addresses the question, 'yes' or 'no'"
    )
    
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """
You are a grader assessing whether an answer addresses / resolves a question \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer addresses / resolves the question.
"""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system),
        ("human", "User question: \n\n {generation} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader

