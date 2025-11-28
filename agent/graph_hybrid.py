import dspy
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from .dspy_signatures import Planner, TextToSQL, Router, Synthesizer
from .tools.sqlite_tool import get_schema, run_query
from .rag.retrieval import LocalRetriever

# CONFIG
lm = dspy.LM(
    "ollama_chat/phi3.5:3.8b-mini-instruct-q4_K_M",
    api_base="http://localhost:11434",
    api_key="",
    max_tokens=1000,
)
dspy.settings.configure(lm=lm)


# STATE
class AgentState(TypedDict):
    question: str
    format_hint: str
    route: str
    docs: List[dict]
    constraints: str
    sql: str
    sql_result: Any
    sql_error: str
    final_answer: Any
    citations: List[str]
    retries: int


# NODES
def router_node(state: AgentState):
    loaded_router = Router()
    loaded_router.load("agent/optimized_router.json")
    pred = loaded_router(question=state["question"])
    route = pred.classification.lower().strip()
    return {"route": route}


def retrieval_node(state: AgentState):
    retriever = LocalRetriever()
    results = retriever.search(state["question"])
    return {"docs": results}


def planner_node(state: AgentState):
    context_str = "\n".join([d["content"] for d in state.get("docs", [])])
    # planner_module = dspy.Predict(Planner)
    # pred = planner_module(question=state["question"], retrieved_context=context_str)
    # return {"constraints": pred.constraints}
    return {"constraints": context_str}


def sql_generation_node(state: AgentState):
    print(
        f"--- [SQL Gen] Generating SQL (Attempt {state.get('retries', 0) + 1})... ---"
    )
    schema = get_schema()
    sql_module = dspy.ChainOfThought(TextToSQL)
    previous_error = state.get("sql_error", "")
    pred = sql_module(
        question=state["question"],
        db_schema=schema,
        constraints=state.get("constraints", ""),
        error_feedback=previous_error,
    )

    clean_sql = pred.sql_query.replace("```sql", "").replace("```", "").strip()
    return {"sql": clean_sql}


def execution_node(state: AgentState):
    df, error = run_query(state["sql"])
    if error:
        return {"sql_error": error, "retries": state.get("retries", 0) + 1}
    return {"sql_result": df.to_dict(orient="records"), "sql_error": None}


def synthesis_node(state: AgentState):
    context_str = "\n".join(
        [f"[{d['id']}] {d['content']}" for d in state.get("docs", [])]
    )

    pred = dspy.Predict(Synthesizer)(
        question=state["question"],
        sql_query=state.get("sql", ""),
        sql_result=str(state.get("sql_result", "")),
        retrieved_context=context_str,
        format_hint=state["format_hint"],
    )
    return {
        "final_answer": pred.final_answer,
        "citations": pred.citations,
        "explanation": pred.explanation,
    }


# ROUTERS
def route_decision(state):
    if state["route"] == "sql":
        return "sql_gen"
    return "retriever"


def check_planner_route(state):
    if state["route"] == "rag":
        return "synthesizer"
    return "sql_gen"


def check_execution(state):
    if state.get("sql_error") and state.get("retries", 0) < 2:
        return "sql_gen"
    return "synthesizer"


# WORKFLOW
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("retriever", retrieval_node)
workflow.add_node("planner", planner_node)
workflow.add_node("sql_gen", sql_generation_node)
workflow.add_node("executor", execution_node)
workflow.add_node("synthesizer", synthesis_node)
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router", route_decision, {"sql_gen": "sql_gen", "retriever": "retriever"}
)
workflow.add_edge("retriever", "planner")
workflow.add_conditional_edges(
    "planner", check_planner_route, {"sql_gen": "sql_gen", "synthesizer": "synthesizer"}
)
workflow.add_edge("sql_gen", "executor")
workflow.add_conditional_edges(
    "executor", check_execution, {"sql_gen": "sql_gen", "synthesizer": "synthesizer"}
)
workflow.add_edge("synthesizer", END)
app = workflow.compile()
