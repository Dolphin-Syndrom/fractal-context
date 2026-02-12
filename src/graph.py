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
from src.tools import tools, set_graph_factory

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

IMPORTANT RULES — read carefully:
1. ALWAYS try to answer DIRECTLY first. If the context fits in your
   current message (roughly under 3,000 tokens), just read it and
   produce your final answer. DO NOT use any tools.
2. ONLY use the Python REPL tool if you truly need to compute something
   (e.g., count words, do math). Never use it just to "inspect" short text.
3. ONLY call delegate_subtask when the context is genuinely massive
   (many thousands of tokens) and you cannot process it in one pass.
4. NEVER hallucinate information that is not in the CONTEXT.
5. When you have a final answer, return it clearly and stop.

Current depth: {depth} / {max_depth}
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

    response = llm_with_tools.invoke(messages)

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
