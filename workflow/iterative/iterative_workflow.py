"""
Iterative Workflow with LangGraph + OpenAI
==========================================
A writer node drafts content. A critic node scores it 1-10.
If the score is below the threshold, the loop sends it BACK
to the writer for improvement. This repeats until the content
is good enough OR the max iteration limit is hit.

Pipeline:

  write_node ──► critic_node
      ▲               │
      │    score < 7  │  score >= 7
      └───────────────┘         │
                                ▼
                          format_node → END

The loop is created with a conditional edge on critic_node.
"""

import os
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# 1. STATE
# ─────────────────────────────────────────────
class WorkflowState(TypedDict):
    topic: str          # the writing topic (set up front)
    draft: str          # current draft (updated every iteration)
    feedback: str       # critic's feedback (updated every iteration)
    score: int          # critic's score 1-10
    iteration: int      # how many times we've written so far
    final: str          # polished final output


# ─────────────────────────────────────────────
# 2. SETTINGS
# ─────────────────────────────────────────────
SCORE_THRESHOLD = 7     # stop looping when score reaches this
MAX_ITERATIONS  = 4     # hard cap to prevent infinite loops

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])


# ─────────────────────────────────────────────
# 3. NODES
# ─────────────────────────────────────────────

def write_node(state: WorkflowState) -> WorkflowState:
    """
    Writes (or rewrites) a short paragraph on the topic.
    On iteration > 1 it receives the critic's feedback and improves.
    """
    iteration = state["iteration"] + 1
    print(f"\n[write_node]  Iteration {iteration} — writing draft...")

    if state["feedback"]:
        # Has been through the loop before — use feedback to improve
        prompt = (
            f"Topic: {state['topic']}\n\n"
            f"Previous draft:\n{state['draft']}\n\n"
            f"Critic feedback: {state['feedback']}\n\n"
            f"Rewrite the paragraph addressing the feedback. "
            f"Keep it to 3-4 sentences."
        )
    else:
        # First attempt
        prompt = (
            f"Write a short, engaging 3-4 sentence paragraph about: {state['topic']}"
        )

    response = llm.invoke(prompt)
    draft = response.content.strip()
    print(f"  → Draft: {draft[:90]}...")
    return {"draft": draft, "iteration": iteration}


def critic_node(state: WorkflowState) -> WorkflowState:
    """
    Scores the draft 1-10 and gives one sentence of feedback.
    The score determines whether the loop continues or exits.
    """
    print(f"\n[critic_node] Scoring draft (iteration {state['iteration']})...")

    response = llm.invoke(
        f"Rate this paragraph on clarity, engagement, and accuracy (1-10).\n\n"
        f"Paragraph:\n{state['draft']}\n\n"
        f"Reply in this exact format:\n"
        f"SCORE: <number>\n"
        f"FEEDBACK: <one sentence of the most important improvement>"
    )

    text = response.content.strip()

    # Parse score (default to 5 if parsing fails)
    score_match = re.search(r"SCORE:\s*(\d+)", text)
    score = int(score_match.group(1)) if score_match else 5
    score = max(1, min(10, score))   # clamp to 1-10

    # Parse feedback
    feedback_match = re.search(r"FEEDBACK:\s*(.+)", text)
    feedback = feedback_match.group(1).strip() if feedback_match else "Improve clarity."

    print(f"  → Score: {score}/10")
    print(f"  → Feedback: {feedback}")
    return {"score": score, "feedback": feedback}


def format_node(state: WorkflowState) -> WorkflowState:
    """Produces the final output once quality is approved."""
    reason = (
        f"score {state['score']}/10 ≥ threshold {SCORE_THRESHOLD}"
        if state["score"] >= SCORE_THRESHOLD
        else f"max iterations ({MAX_ITERATIONS}) reached"
    )
    print(f"\n[format_node] Finalizing — {reason}.")
    final = (
        f"Topic      : {state['topic']}\n"
        f"Iterations : {state['iteration']}\n"
        f"Final score: {state['score']}/10\n\n"
        f"{state['draft']}"
    )
    return {"final": final}


# ─────────────────────────────────────────────
# 4. ROUTING FUNCTION  ← the loop lives here
# ─────────────────────────────────────────────

def should_continue(state: WorkflowState) -> str:
    """
    Called after critic_node every iteration.
    Returns "improve" to loop back, or "done" to exit.
    """
    if state["score"] >= SCORE_THRESHOLD:
        print(f"\n  ✓ Score {state['score']} reached threshold — exiting loop.")
        return "done"
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"\n  ⚠ Max iterations reached — exiting loop.")
        return "done"
    print(f"\n  ↺ Score {state['score']} below threshold — looping back to improve.")
    return "improve"


# ─────────────────────────────────────────────
# 5. GRAPH
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(WorkflowState)

    graph.add_node("write_node",  write_node)
    graph.add_node("critic_node", critic_node)
    graph.add_node("format_node", format_node)

    graph.set_entry_point("write_node")
    graph.add_edge("write_node", "critic_node")

    # ★ THE KEY PART: conditional edge that creates the loop
    graph.add_conditional_edges(
        "critic_node",      # after the critic scores the draft…
        should_continue,    # …call this to decide what's next
        {
            "improve": "write_node",   # loop back
            "done":    "format_node",  # exit the loop
        }
    )

    graph.add_edge("format_node", END)
    return graph.compile()


# ─────────────────────────────────────────────
# 6. RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = build_graph()

    initial_state: WorkflowState = {
        "topic":     "The surprising benefits of cold showers",
        "draft":     "",
        "feedback":  "",
        "score":     0,
        "iteration": 0,
        "final":     "",
    }

    print("=" * 55)
    print("  LangGraph Iterative Workflow Demo")
    print(f"  Topic: {initial_state['topic']}")
    print(f"  Target score: {SCORE_THRESHOLD}/10  |  Max loops: {MAX_ITERATIONS}")
    print("=" * 55)

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 55)
    print("  FINAL OUTPUT")
    print("=" * 55)
    print(final_state["final"])