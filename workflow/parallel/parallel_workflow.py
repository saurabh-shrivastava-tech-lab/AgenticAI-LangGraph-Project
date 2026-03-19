"""
Parallel Workflow with LangGraph + OpenAI
==========================================
This example builds a pipeline where 3 nodes run IN PARALLEL,
then their results are merged by a final node.
 
Pipeline:
                ┌──► summarize_node   ──┐
                │                       │
  topic_node ───┼──► keywords_node  ────┼──► merge_node → END
                │                       │
                └──► fun_fact_node  ────┘
 
Step 1 (topic_node)   : Pick a science topic
Step 2 (PARALLEL)     : Summarize it / Extract keywords / Find a fun fact
Step 3 (merge_node)   : Combine all results into a final report
"""

import os
from langgraph.graph import StateGraph, START,END
from langchain_openai import ChatOpenAI

from typing import TypedDict

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# 1. STATE
# ─────────────────────────────────────────────

class WorkflowState(TypedDict):
    topic: str          # set by topic_node
    summary : str       # set by summarize_node  (parallel)
    keywords:str        # set by keywords_node   (parallel)
    fun_fact:str        # set by fun_fact_node   (parallel)
    report : str        # set by merge_node

# ─────────────────────────────────────────────
# 2. LLM
# ─────────────────────────────────────────────

llm = ChatOpenAI(
    model ="gpt-4o-mini",
    api_key = os.environ["OPENAI_API_KEY"]
)

# ─────────────────────────────────────────────
# 3. NODES
# ─────────────────────────────────────────────

def topic_node(state:WorkflowState)->WorkflowState:
    """Step 1: Picking Topic . Run First before the parallel nodes. """
    print("\n [topic_node] Picking a topic.....")
    response = llm.invoke("Name one facinating science topic in 3-5 words.Just name, nothing else.")
    topic = response.content.strip()
    print(f"    ->{topic}")
    return {"topic":topic}

# ── The three parallel nodes below all receive the same state ──

def summarize_node(state:WorkflowState)->WorkflowState:
    """Parallel Branch A: Write 2 -3 sentences sumary. """
    print(f"\n [summarize_node] sumarizing '{state['topic']}'....")
    prompt = f"Write a 2-3 sentences summary of '{state['topic']}'"
    response = llm.invoke(prompt)
    summary = response.content.strip()
    print(f"    ->{summary}")
    return {"summary":summary}

def keywords_node(state: WorkflowState) -> WorkflowState:
    """Parallel branch B: Extract 5 keywords."""
    print(f"\n[keywords_node] Extracting keywords for '{state['topic']}'...")
    response = llm.invoke(
        f"List exactly 5 keywords related to '{state['topic']}'. "
        "Return them comma-separated on one line."
    )
    keywords = response.content.strip()
    print(f"  → {keywords}")
    return {"keywords": keywords}
 
 
def fun_fact_node(state: WorkflowState) -> WorkflowState:
    """Parallel branch C: Find a fun fact."""
    print(f"\n[fun_fact_node] Finding a fun fact about '{state['topic']}'...")
    response = llm.invoke(
        f"Give me one surprising fun fact about '{state['topic']}' in one sentence."
    )
    fun_fact = response.content.strip()
    print(f"  → {fun_fact}")
    return {"fun_fact": fun_fact}
 
 
def merge_node(state: WorkflowState) -> WorkflowState:
    """Step 3: Merge all parallel results into one report."""
    print("\n[merge_node] Merging results...")
    report = (
        f"📌 TOPIC\n{state['topic']}\n\n"
        f"📝 SUMMARY\n{state['summary']}\n\n"
        f"🔑 KEYWORDS\n{state['keywords']}\n\n"
        f"💡 FUN FACT\n{state['fun_fact']}"
    )
    return {"report": report}

# ─────────────────────────────────────────────
# 4. GRAPH
# ─────────────────────────────────────────────

def build_graph()->StateGraph:
    graph = StateGraph(WorkflowState)

     # Register all nodes 

         # Register all nodes
    graph.add_node("topic_node",     topic_node)
    graph.add_node("summarize_node", summarize_node)
    graph.add_node("keywords_node",  keywords_node)
    graph.add_node("fun_fact_node",  fun_fact_node)
    graph.add_node("merge_node",     merge_node)
 
    # Entry point
    graph.set_entry_point("topic_node")

    # topic_node fans OUT to 3 parallel nodes

    graph.add_edge("topic_node","summarize_node")
    graph.add_edge("topic_node","keywords_node")
    graph.add_edge("topic_node","fun_fact_node")


    # All 3 parallel nodes fan IN to merge_node

    graph.add_edge("summarize_node","merge_node")
    graph.add_edge("keywords_node","merge_node")
    graph.add_edge("fun_fact_node","merge_node")

    graph.add_edge("merge_node", END)

    return graph.compile()

# ─────────────────────────────────────────────
# 5. RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = build_graph()

    initial_state: WorkflowState = {
        "topic": "", "summary": "", "keywords": "", "fun_fact": "", "report": ""
    }

    print("=" * 55)
    print("  LangGraph Parallel Workflow Demo")
    print("=" * 55)
 
    final_state = app.invoke(initial_state)
 
    print("\n" + "=" * 55)
    print("  FINAL REPORT")
    print("=" * 55)
    print(final_state["report"])




