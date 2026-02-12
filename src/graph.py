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

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.state import RLMState
from src.tools import tools, set_graph_factory, set_depth_context

# ---------------------------------------------------------------------------
# Load environment & initialise LLM
# ---------------------------------------------------------------------------

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# Bind the tool definitions so the model can emit tool-call messages
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------------------------
# System prompt — conservative: only use tools for genuinely large context
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Recursive Language Model (RLM) agent.
Your job is to answer the user's QUERY using only the provided CONTEXT.

The user has provided:
- QUERY: The question you must answer
- CONTEXT: The information you should analyze

IMPORTANT RULES:
1. Read the QUERY carefully - this is what you need to answer.
2. Use ONLY the information in the CONTEXT to answer. Never make up information.
3. If the context is small (under 3,000 tokens), read it directly and answer.
4. Use Python REPL tool ONLY for calculations (counting, math, etc).
5. Use delegate_subtask ONLY when context is massive (many thousands of tokens).
6. After using any tool, immediately provide your final answer to the QUERY.
7. DO NOT ask for more information. DO NOT say "please provide the query."
   You already have the query and context - just answer it!

Current depth: {depth} / {max_depth}

FORMAT YOUR FINAL ANSWER CLEARLY AND DIRECTLY.
"""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def reason_node(state: RLMState) -> dict:
    """
    Call the LLM with the current context and query.

    The model may either:
      - Produce a final answer  →  sets answer["ready"] = True
      - Emit tool calls         →  graph routes to the tool_node

    CRITICAL: The full context is only sent on the FIRST call.
    Subsequent calls (after tool execution) only see the accumulated
    message history, avoiding re-sending the context every loop.
    """
    sys_msg = SystemMessage(
        content=SYSTEM_PROMPT.format(
            depth=state["depth"],
            max_depth=state["max_depth"],
        )
    )

    # Pass current depth to tools module so delegate_subtask knows
    set_depth_context(state["depth"], state["max_depth"])

    existing_messages = state.get("messages", [])

    if not existing_messages:
        # ── First call: include context in a human message ──────────
        human_msg = HumanMessage(
            content=f"QUERY: {state['query']}\n\nCONTEXT:\n{state['context']}"
        )
        messages = [sys_msg, human_msg]
    else:
        # ── Subsequent calls: context is already in history ─────────
        # Just prepend the system prompt; history already has the
        # human msg + AI response + tool results.
        messages = [sys_msg] + existing_messages

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        # Groq can fail on malformed function calls — fall back to
        # a direct answer without tools
        fallback = llm.invoke(messages)
        return {
            "answer": {"content": fallback.content, "ready": True},
            "messages": [fallback],
        }

    # If the model did NOT emit tool calls, treat response as the final answer
    if not response.tool_calls:
        return {
            "answer": {"content": response.content, "ready": True},
            "messages": [response],
        }

    # Otherwise, keep the answer pending and let the tool node execute calls
    return {
        "answer": {"content": "", "ready": False},
        "messages": [response],
    }


# Tool node — automatically executes any tool calls from the LLM response
tool_node = ToolNode(tools)


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_continue(state: RLMState) -> str:
    """Route to END when the answer is ready, otherwise to tools."""
    if state.get("answer", {}).get("ready", False):
        return "end"
    return "tools"


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def compile_graph():
    """
    Build and compile the RLM StateGraph.

    Returns:
        A compiled LangGraph application ready for .invoke() or .astream_events().
    """
    graph = StateGraph(RLMState)

    # -- Add nodes --
    graph.add_node("reason", reason_node)
    graph.add_node("tools", tool_node)

    # -- Set entry point --
    graph.set_entry_point("reason")

    # -- Conditional edge from reason --
    graph.add_conditional_edges(
        "reason",
        should_continue,
        {
            "end": END,
            "tools": "tools",
        },
    )

    # -- After tools, always go back to reason --
    graph.add_edge("tools", "reason")

    return graph.compile()


# ---------------------------------------------------------------------------
# Register the factory so tools.py can compile fresh sub-graphs
# ---------------------------------------------------------------------------

set_graph_factory(compile_graph)
