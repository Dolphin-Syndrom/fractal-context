"""
Needle-in-a-Haystack Test — Standalone
=======================================
Runs the RLM graph directly (no Chainlit) against tests/dummy_data.txt
and checks whether SECRET_KEY_99 is found.

Usage:
    python tests/test_needle.py
"""

import os
import sys
import time

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph import compile_graph
from src.utils import load_file


def run_test():
    # --- Locate test data ---
    data_path = os.path.join(os.path.dirname(__file__), "dummy_data.txt")
    if not os.path.isfile(data_path):
        print("❌ dummy_data.txt not found. Run generate_test_data.py first:")
        print("   python tests/generate_test_data.py")
        sys.exit(1)

    context = load_file(data_path)
    print(f"📄 Loaded {len(context):,} characters from {data_path}")

    # --- Compile graph ---
    app = compile_graph()
    print("✅ Graph compiled\n")

    # --- Run ---
    query = "Find the secret key hidden in this text. Return the exact key value."
    state = {
        "query": query,
        "context": context,
        "depth": 0,
        "max_depth": 3,
        "answer": {"content": "", "ready": False},
        "messages": [],
    }

    print(f"Query: {query}")
    print("Running agent...\n")

    start = time.time()
    result = app.invoke(state)
    elapsed = time.time() - start

    answer = result.get("answer", {}).get("content", "")
    print("=" * 60)
    print(f"Answer:\n{answer}")
    print("=" * 60)
    print(f"⏱Elapsed: {elapsed:.1f}s")

    # --- Validate ---
    if "SECRET_KEY_99" in answer:
        print("\nTEST PASSED — Secret key found!")
    else:
        print("\nTEST FAILED — Secret key not in answer.")
        print("   Expected 'SECRET_KEY_99' in the response.")


if __name__ == "__main__":
    run_test()
