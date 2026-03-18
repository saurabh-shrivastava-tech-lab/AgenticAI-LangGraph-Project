"""
Sequential Workflow with LangGraph + OpenAI
============================================
This example builds a 3-step pipeline:
  Step 1 → Generate a topic idea
  Step 2 → Write a short paragraph about it
  Step 3 → Summarize the paragraph in one sentence

Each step is a "node" in the graph, and they run in sequence.
"""

import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# 1. STATE  — shared data passed between nodes
# ─────────────────────────────────────────────
class WorkflowState(TypedDict):
    topic: str        # filled by Node 1
    paragraph: str    # filled by Node 2
    summary: str      # filled by Node 3


# ─────────────────────────────────────────────
# 2. LLM  — one shared model instance
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",          # cheap & fast for learning
    api_key=os.environ["OPENAI_API_KEY"],
)


# ─────────────────────────────────────────────
# 3. NODES  — each receives state, returns update
# ─────────────────────────────────────────────

def generate_topic(state: WorkflowState) -> WorkflowState:
    """Node 1: Ask the LLM to pick an interesting science topic."""
    print("\n[Node 1] Generating topic...")
    response = llm.invoke("Suggest one short, interesting science topic (just the topic name, no explanation).")
    topic = response.content.strip()
    print(f"  → Topic: {topic}")
    return {"topic": topic}


def write_paragraph(state: WorkflowState) -> WorkflowState:
    """Node 2: Write a short paragraph about the topic from Node 1."""
    print("\n[Node 2] Writing paragraph...")
    prompt = f"Write a short 3-sentence paragraph about: {state['topic']}"
    response = llm.invoke(prompt)
    paragraph = response.content.strip()
    print(f"  → Paragraph: {paragraph}")
    return {"paragraph": paragraph}


def summarize(state: WorkflowState) -> WorkflowState:
    """Node 3: Summarize the paragraph from Node 2 in one sentence."""
    print("\n[Node 3] Summarizing...")
    prompt = f"Summarize this in exactly one sentence:\n\n{state['paragraph']}"
    response = llm.invoke(prompt)
    summary = response.content.strip()
    print(f"  → Summary: {summary}")
    return {"summary": summary}


# ─────────────────────────────────────────────
# 4. GRAPH  — wire the nodes together
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(WorkflowState)

    # Register nodes
    graph.add_node("generate_topic", generate_topic)
    graph.add_node("write_paragraph", write_paragraph)
    graph.add_node("summarize", summarize)

    # Set the entry point (first node to run)
    graph.set_entry_point("generate_topic")

    # Sequential edges: A → B → C → END
    graph.add_edge("generate_topic", "write_paragraph")
    graph.add_edge("write_paragraph", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 5. RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = build_graph()

    # Initial state — empty because Node 1 fills it
    initial_state: WorkflowState = {
        "topic": "",
        "paragraph": "",
        "summary": "",
    }

    print("=" * 50)
    print("  LangGraph Sequential Workflow Demo")
    print("=" * 50)

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 50)
    print("  FINAL RESULTS")
    print("=" * 50)
    print(f"Topic     : {final_state['topic']}")
    print(f"Paragraph : {final_state['paragraph']}")
    print(f"Summary   : {final_state['summary']}")