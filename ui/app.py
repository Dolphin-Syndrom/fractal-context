"""
RLM Chainlit UI
===============
Entry point for the "Glass Box" visualization of the RLM agent.
Run with:  chainlit run ui/app.py
"""

import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the RLM graph when a new chat session begins.
    """
    # TODO (Phase 3): Compile graph and store in user session
    # from src.graph import compile_graph
    # app = compile_graph()
    # cl.user_session.set("app", app)
    await cl.Message(content="🧠 **RLM Agent** initialized. Send a message to begin.").send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming user messages.

    Triggers the RLM graph and streams events back to the UI,
    rendering sub-agent steps as nested Chainlit Steps.
    """
    # TODO (Phase 3): Full implementation
    # 1. Retrieve compiled app from session
    # 2. Build initial RLMState from message
    # 3. Stream events via app.astream_events()
    # 4. Render sub-agent steps as cl.Step(name=f"Sub-Agent (Depth {depth})")
    await cl.Message(content="⏳ Graph execution not yet wired — see Phase 3.").send()
