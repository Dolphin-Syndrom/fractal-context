"""
RLM Chainlit UI — "Glass Box" Visualization
=============================================
Entry point for the recursive agent visualization.
Run with:  chainlit run ui/app.py -w

Renders LangGraph events as nested Chainlit Steps so the user
can *see* sub-agents spawning at each recursion depth.
"""

import sys
import os

# Ensure project root is on sys.path so `from src.*` imports work
# regardless of where Chainlit launches from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chainlit as cl
from src.graph import compile_graph
from src.state import RLMState


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    """Compile the RLM graph once per chat session."""
    app = compile_graph()
    cl.user_session.set("app", app)
    await cl.Message(
        content=(
            "🧠 **Recursive Language Model** ready.\n\n"
            "Send me a question along with some context, and I'll "
            "recursively break it down if needed."
        ),
    ).send()


# ---------------------------------------------------------------------------
# Message handler — streams graph events into nested Steps
# ---------------------------------------------------------------------------

@cl.on_message
async def on_message(message: cl.Message):
    """
    Run the RLM graph on the user's message and visualise every
    internal event as a nested Chainlit Step.
    """
    app = cl.user_session.get("app")
    if app is None:
        await cl.Message(content="❌ Graph not initialised. Please refresh.").send()
        return

    # Build the initial state
    # Priority: file attachment > "Context:" in text > raw message
    raw = message.content
    context = None
    query = None

    # Check for file attachments
    if message.elements:
        for elem in message.elements:
            if hasattr(elem, "path") and elem.path:
                try:
                    with open(elem.path, "r", encoding="utf-8") as f:
                        context = f.read()
                    await cl.Message(
                        content=f"📎 Loaded file: **{elem.name}** ({len(context):,} chars)"
                    ).send()
                except Exception as e:
                    await cl.Message(content=f"⚠️ Could not read file: {e}").send()
                break

    # Determine query and context based on what was provided
    if context is not None:
        # File was uploaded - use message as query
        query = raw if raw.strip() else "Analyze this file."
    elif "Context:" in raw:
        # "Context:" format - split it
        parts = raw.split("Context:", 1)
        query = parts[0].strip()
        context = parts[1].strip()
    else:
        # Plain message - treat as both query and context
        query = raw
        context = raw

    initial_state: RLMState = {
        "query": query,
        "context": context,
        "depth": 0,
        "max_depth": 3,
        "answer": {"content": "", "ready": False},
        "messages": [],
    }

    # Tracking objects for nested step rendering
    final_answer = ""
    active_steps: dict[str, cl.Step] = {}  # keyed by run_id or event tag

    # ----- Stream events from the graph -----
    async for event in app.astream_events(initial_state, version="v2"):
        kind = event.get("event", "")
        name = event.get("name", "")
        data = event.get("data", {})
        run_id = event.get("run_id", "")
        tags = event.get("tags", [])

        # ── 🧠 LLM token streaming (Thinking…) ──────────────────────
        if kind == "on_chat_model_stream":
            chunk = data.get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                step_key = f"thinking_{run_id}"
                if step_key not in active_steps:
                    step = cl.Step(name="🧠 Thinking…", type="llm")
                    active_steps[step_key] = step
                    await step.send()
                active_steps[step_key].output = (
                    (active_steps[step_key].output or "") + chunk.content
                )
                await active_steps[step_key].update()

        # ── LLM done ────────────────────────────────────────────────
        elif kind == "on_chat_model_end":
            step_key = f"thinking_{run_id}"
            if step_key in active_steps:
                await active_steps[step_key].update()

        # ── 🔧 Tool start ──────────────────────────────────────────
        elif kind == "on_tool_start":
            tool_input = data.get("input", {})

            if name == "delegate_subtask":
                depth = tool_input.get("current_depth", 0) + 1
                step = cl.Step(
                    name=f"🔀 Sub-Agent (Depth {depth})",
                    type="tool",
                )
                step.input = (
                    f"**Query:** {tool_input.get('sub_query', '')}\n\n"
                    f"**Context chunk:** {str(tool_input.get('sub_context', ''))[:200]}…"
                )
                active_steps[f"tool_{run_id}"] = step
                await step.send()

            elif "python" in name.lower() or "repl" in name.lower():
                step = cl.Step(name="💻 Coding…", type="tool")
                step.input = str(tool_input)
                active_steps[f"tool_{run_id}"] = step
                await step.send()

            else:
                step = cl.Step(name=f"🔧 {name}", type="tool")
                step.input = str(tool_input)
                active_steps[f"tool_{run_id}"] = step
                await step.send()

        # ── 🔧 Tool end ────────────────────────────────────────────
        elif kind == "on_tool_end":
            step_key = f"tool_{run_id}"
            if step_key in active_steps:
                output = data.get("output", "")
                # ToolMessage objects have a .content attribute
                if hasattr(output, "content"):
                    output = output.content
                active_steps[step_key].output = str(output)[:1000]
                await active_steps[step_key].update()

        # ── Graph chain end — capture final answer ──────────────────
        elif kind == "on_chain_end" and name == "reason":
            output = data.get("output", {})
            if isinstance(output, dict):
                answer = output.get("answer", {})
                if isinstance(answer, dict) and answer.get("ready"):
                    final_answer = answer.get("content", "")

    # ----- Send the final answer as a top-level message -----
    if final_answer:
        await cl.Message(content=final_answer).send()
    else:
        await cl.Message(
            content="⚠️ The agent did not produce a final answer. Check the steps above for details."
        ).send()
