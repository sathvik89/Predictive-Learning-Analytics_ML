from typing import TypedDict, List, Annotated
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    student_data: dict
    ml_results: dict
    retrieved_docs: List[str]
    last_retrieval_query: str
    quiz_questions: List[dict]
    current_q_idx: int
    quiz_score: int
    quiz_active: bool
    awaiting_answer: bool
    study_plan: str
    plan: List[str] #initial master node steps prep 
    task_plan: List[dict] #reason for each step 
    response_parts: List[dict] #append teh respnses of each node so that u can compile them at the end
    response_mode: str #(academic , greeting, off-topic , cheatingCase)
    direct_response: str #(ex - greeting could be direct msg)
    current_step_index: int
    next_node: str
