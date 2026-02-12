"""
RLM Graph Definition
====================
Defines the main LangGraph StateGraph for the Recursive Language Model.

Nodes:
  - reason_node : Calls the LLM to analyze context and decide next action.
  - tool_node   : Executes tool calls (REPL / delegate_subtask).

Edges:
  - Conditional: answer["ready"] == True  →  END
                 otherwise                →  tools
"""

from src.state import RLMState


def compile_graph():
    """
    Build and compile the RLM StateGraph.

    Returns:
        A compiled LangGraph application ready for .invoke() or .astream_events().
    """
    # TODO (Phase 2): Full implementation
    # 1. Instantiate StateGraph(RLMState)
    # 2. Add reason_node and tool_node
    # 3. Wire conditional edges based on answer["ready"]
    # 4. Compile and return
    raise NotImplementedError("Graph compilation is implemented in Phase 2.")
