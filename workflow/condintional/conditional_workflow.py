"""
Conditional Workflow with LangGraph + OpenAI
=============================================
A user submits a question. The router decides which specialist
node should handle it, then a final node formats the answer.
 
Pipeline:
 
  input_node
      │
      ▼
  router_node  ── "science"  ──► science_node ──┐
               ── "history"  ──► history_node ──┤
               ── "general"  ──► general_node ──┤
                                                 ▼
                                           format_node → END
 
The conditional branching is done with add_conditional_edges().
"""
import os
from langgraph.graph import StateGraph, START,END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# 1. STATE
# ─────────────────────────────────────────────

class WorkflowState(TypedDict):
    question :str   # user's question (provided up front)
    category :str   # filled by router_node: "science" | "history" | "general"
    answer :str     # filled by whichever specialist runs
    final : str     # filled by format_node


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

def input_node(state:WorkflowState)->WorkflowState:
    """Step 1: Just echo the question so we can see what's coming in."""
    print(f"\n[input_node]  Question received: '{state['question']}'")
    return {}   # nothing new to add — question is already in state

def router_node(state:WorkflowState)->WorkflowState:
    """
    Step 2: Classify the question into a category.
    This sets the 'category' field that the conditional edge will read.
    """
    print("\n[router_node] Classifying question...")

    response = llm.invoke(f"Classify this question into exactly one work : science , history , or general.\n"
                          f"Question :{state['question']}\n"
                          f"Reply with only singleword , lowercase."
                          )
    
    category = response.content.strip().lower()

    # Safety fallback — if LLM returns something unexpected
    if category not in ("science", "history", "general"):
        category ="general"

      
    print(f"  → Category: {category}")
    return {"category": category}


# ── Specialist nodes (only ONE runs per question) ──

def science_node(state: WorkflowState) -> WorkflowState:
    """Handles science questions."""
    print("\n[science_node] Answering science question...")
    response = llm.invoke(
        f"You are a science expert. Answer clearly in 2-3 sentences:\n{state['question']}"
    )
    answer = response.content.strip()
    print(f"  → {answer[:80]}...")
    return {"answer": answer}
 
 
def history_node(state: WorkflowState) -> WorkflowState:
    """Handles history questions."""
    print("\n[history_node] Answering history question...")
    response = llm.invoke(
        f"You are a history expert. Answer clearly in 2-3 sentences:\n{state['question']}"
    )
    answer = response.content.strip()
    print(f"  → {answer[:80]}...")
    return {"answer": answer}
 
 
def general_node(state: WorkflowState) -> WorkflowState:
    """Handles everything else."""
    print("\n[general_node] Answering general question...")
    response = llm.invoke(
        f"Answer helpfully in 2-3 sentences:\n{state['question']}"
    )
    answer = response.content.strip()
    print(f"  → {answer[:80]}...")
    return {"answer": answer}

def format_node(state: WorkflowState) -> WorkflowState:
    """Step 4: Wrap the answer with the category label."""
    print("\n[format_node] Formatting final response...")
    final = (
        f"[{state['category'].upper()} ANSWER]\n"
        f"{state['answer']}"
    )
    return {"final": final} 


# ─────────────────────────────────────────────
# 4. ROUTING FUNCTION
# ─────────────────────────────────────────────

def route_question(state: WorkflowState) -> Literal["science", "history", "general"]:
    """
    This function is called by add_conditional_edges().
    It reads the state and returns the NAME of the next node to run.
    """
    return f"{state['category']}"



# ─────────────────────────────────────────────
# 5. GRAPH
# ─────────────────────────────────────────────

def build_graph() ->StateGraph:
    graph = StateGraph(WorkflowState)

    # Register all nodes
    graph.add_node("input_node",   input_node)
    graph.add_node("router_node",   router_node)
    graph.add_node("science_node",   science_node)
    graph.add_node("history_node",   history_node)
    graph.add_node("general_node",   general_node)
    graph.add_node("format_node",   format_node)

    # Fixed edges
    graph.set_entry_point("input_node")

    graph.add_edge("input_node", "router_node")

    # ★ THE KEY PART: conditional edge from router_node
    #   - route_question() decides which path to take
    #   - The dict maps possible return values → node names
    graph.add_conditional_edges(
        "router_node",          # source node
        route_question,         # function that returns the next node name or value
        {
            "science": "science_node", # value : next node (science_node) 
            "history": "history_node",
            "general": "general_node",
        }
    )

    # All specialist paths converge here
    graph.add_edge("science_node", "format_node")
    graph.add_edge("history_node", "format_node")
    graph.add_edge("general_node", "format_node")
    graph.add_edge("format_node",  END)

    return graph.compile()
    
QUESTIONS = [
    "Why is the sky blue?",
    "Who started World War I?",
    "What is the best way to learn a new language?",
]
 
if __name__ == "__main__":
    app = build_graph()
 
    for question in QUESTIONS:
        print("\n" + "=" * 55)
        print(f"  QUESTION: {question}")
        print("=" * 55)
 
        final_state = app.invoke({
            "question": question,
            "category": "",
            "answer": "",
            "final": "",
        })
 
    print(f"\n  ✓ RESULT\n  {final_state['final']}")


"""



Value (from function)	Node (next step)
"red"	                "stop_vehicle"
"yellow"	            "slow_down"
"green"	                "move_forward"


graph.add_conditional_edges(
    "check_signal",
    check_signal,
    {
        "red": "stop_vehicle",
        "yellow": "slow_down",
        "green": "move_forward"
    }
)

What is Literal?
from typing import Literal
It restricts the return value to specific allowed values.

def check_signal(state) -> Literal["red", "yellow", "green"]:

def check_signal(state: SignalState) -> Literal["red", "yellow", "green"]:
    if state.signal == "red":
        return "rede"
    elif state.signal == "yellow":
        return "yellow"
    elif state.signal == "green":
        return "green"
    else:
        raise ValueError("Invalid signal")
"""
