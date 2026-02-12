"""
RLM Tools
=========
Core tools available to the LLM agent:
  1. python_repl — run Python code to inspect/slice context strings.
  2. delegate_subtask — spawn a child agent at depth+1.

Tool schemas are kept deliberately simple (required string args only)
so Groq's Llama function-calling works reliably.
"""

import io
import contextlib

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Graph accessor (breaks circular import between tools ↔ graph)
# ---------------------------------------------------------------------------

_graph_factory = None  # Will be set by graph.py at import time
_current_depth = 0     # Set by graph.py reason_node before tool execution
_current_max_depth = 3


def set_graph_factory(factory_fn):
    """Register the graph compile function (called from graph.py)."""
    global _graph_factory
    _graph_factory = factory_fn


def get_app():
    """Return a freshly compiled graph instance, or None if not yet set."""
    if _graph_factory is None:
        return None
    return _graph_factory()


def set_depth_context(depth: int, max_depth: int):
    """Called by reason_node to pass current depth to tools."""
    global _current_depth, _current_max_depth
    _current_depth = depth
    _current_max_depth = max_depth


# ---------------------------------------------------------------------------
# Tool: Python REPL  (simple custom wrapper — Groq-compatible)
# ---------------------------------------------------------------------------

@tool
def python_repl(code: str) -> str:
    """
    Execute Python code and return the printed output.
    Use this to inspect, slice, or compute things about the context.

    Args:
        code: Python source code to execute.

    Returns:
        The stdout output from the code execution, or an error message.
    """
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": __builtins__})
        output = stdout.getvalue()
        return output if output else "(No output)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Tool: delegate_subtask
# ---------------------------------------------------------------------------

@tool
def delegate_subtask(sub_query: str, sub_context: str) -> str:
    """
    Delegate a sub-query to a child agent at the next recursion depth.
    Use this ONLY when the context is too large to process directly.

    Args:
        sub_query:   The refined question for the child agent.
        sub_context: The text chunk the child agent should analyze.

    Returns:
        The child agent's answer as a string.
    """
    depth = _current_depth
    max_depth = _current_max_depth

    # --- Guard: prevent runaway recursion ---
    if depth >= max_depth:
        return (
            f"Max recursion depth ({max_depth}) reached. "
            f"Context preview: {sub_context[:500]}"
        )

    # --- Compile a fresh sub-graph ---
    app = get_app()
    if app is None:
        return "Graph factory not registered — cannot recurse."

    # --- Invoke child agent with incremented depth ---
    child_state = {
        "query": sub_query,
        "context": sub_context,
        "depth": depth + 1,
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
