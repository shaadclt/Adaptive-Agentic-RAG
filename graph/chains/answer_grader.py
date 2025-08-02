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

"""
The answer grader evaluates whether the generated response actually addresses the user's question. 
Even if a response is factually grounded, it might not directly answer what the user asked.
This component ensures that our system provides responses that are both accurate and relevant to the specific query. 
The grader checks if the generation resolves the question, and if not, the system can trigger additional retrieval or web search to find more appropriate information.
"""