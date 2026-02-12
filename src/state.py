"""
RLM State Definition
====================
Defines the graph state schema for the Recursive Language Model.
The state flows through every node and edge in the LangGraph.
"""

from typing import TypedDict, Annotated, List, Dict, Any
import operator


class RLMState(TypedDict):
    """
    Core state object that tracks data across recursion levels.

    Attributes:
        query:     The question/task for this recursion level.
        context:   The text chunk assigned to this level for processing.
        depth:     Current recursion depth (root agent starts at 0).
        max_depth: Hard ceiling to prevent runaway recursion (e.g., 3).
        answer:    Result dict — {"content": "...", "ready": bool}.
                   When ready=True the graph terminates.
        messages:  Accumulated chat history (append-only via operator.add).
    """
    query: str
    context: str
    depth: int
    max_depth: int
    answer: Dict[str, Any]
    messages: Annotated[List[Any], operator.add]
