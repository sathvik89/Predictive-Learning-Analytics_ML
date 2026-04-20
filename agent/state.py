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
    plan: List[str]
    task_plan: List[dict]
    response_parts: List[dict]
    response_mode: str
    direct_response: str
    current_step_index: int
    next_node: str
