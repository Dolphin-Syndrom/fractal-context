"""
RLM Tools
=========
Core tools available to the LLM agent:
  1. Python REPL — for inspecting/slicing context strings.
  2. Delegate Subtask — spawns a child agent at depth+1.
"""

from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool


# ---------------------------------------------------------------------------
# Graph accessor (breaks circular import between tools ↔ graph)
# ---------------------------------------------------------------------------

_graph_factory = None  # Will be set by graph.py at import time


def set_graph_factory(factory_fn):
    """Register the graph compile function (called from graph.py)."""
    global _graph_factory
    _graph_factory = factory_fn


def get_app():
    """Return a freshly compiled graph instance, or None if not yet set."""
    if _graph_factory is None:
        return None
    return _graph_factory()


# ---------------------------------------------------------------------------
# Tool: Python REPL
# ---------------------------------------------------------------------------

python_repl = PythonREPLTool()


# ---------------------------------------------------------------------------
# Tool: delegate_subtask
# ---------------------------------------------------------------------------

@tool
def delegate_subtask(
    sub_query: str,
    sub_context: str,
    current_depth: int = 0,
    max_depth: int = 3,
) -> str:
    """
    Delegate a sub-query to a child agent at depth + 1.

    Use this when the context is too large to reason about directly.
    The child agent receives a smaller chunk and returns its answer.

    Args:
        sub_query:     The refined question for the child agent.
        sub_context:   The text chunk the child agent should analyze.
        current_depth: The current recursion depth (passed from parent state).
        max_depth:     Hard ceiling for recursion depth.

    Returns:
        The child agent's answer as a string.
    """
    # --- Guard: prevent runaway recursion ---
    if current_depth >= max_depth:
        return (
            f"[delegate_subtask] Max recursion depth ({max_depth}) reached. "
            f"Returning sub_context summary at depth {current_depth}:\n"
            f"{sub_context[:500]}"
        )

    # --- Compile a fresh sub-graph ---
    app = get_app()
    if app is None:
        return "[delegate_subtask] Graph factory not registered — cannot recurse."

    # --- Invoke child agent with incremented depth ---
    child_state = {
        "query": sub_query,
        "context": sub_context,
        "depth": current_depth + 1,
        "max_depth": max_depth,
        "answer": {"content": "", "ready": False},
        "messages": [],
    }

    result = app.invoke(child_state)
    return result.get("answer", {}).get("content", "[No answer from child agent]")


# ---------------------------------------------------------------------------
# Exported tool list (used by graph.py to bind to the LLM)
# ---------------------------------------------------------------------------

tools = [python_repl, delegate_subtask]
