"""
RLM Tools
=========
Core tools available to the LLM agent:
  1. Python REPL — for inspecting/slicing context strings.
  2. Delegate Subtask — spawns a child agent at depth+1.
"""

from langchain_core.tools import tool


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
# Tool: delegate_subtask
# ---------------------------------------------------------------------------

@tool
def delegate_subtask(sub_query: str, sub_context: str) -> str:
    """
    Delegate a sub-query to a child agent at depth + 1.

    Use this when the context is too large to reason about directly.
    The child agent receives a smaller chunk and returns its answer.

    Args:
        sub_query:   The refined question for the child agent.
        sub_context: The text chunk the child agent should analyze.

    Returns:
        The child agent's answer as a string.
    """
    # TODO (Phase 2): Implement full recursion logic
    # - Read current depth from parent state
    # - Check depth < max_depth
    # - Compile a new graph instance via get_app()
    # - Invoke with incremented depth
    return "[delegate_subtask] Not yet implemented — see Phase 2."
