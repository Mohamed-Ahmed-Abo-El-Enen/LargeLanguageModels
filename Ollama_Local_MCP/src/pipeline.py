from typing import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from src.vector import retrieve 

from src.config import DEFAULT_MODEL, OLLAMA_BASE
from src.logger import get_logger, log_call
from src.registry import list_agents

log = get_logger(__name__)


class PipelineState(TypedDict):
    input:   str
    context: str
    outputs: list[dict]
    final:   str


def _format_prior(outputs: list[dict]) -> str:
    return "\n\n".join(f"[{o['agent']}]\n{o['output']}" for o in outputs)


def _build_user_prompt(state: PipelineState) -> str:
    parts = [f"User input:\n{state['input']}"]
    if state.get("context"):
        parts.append(f"Retrieved context:\n{state['context']}")
    prior = _format_prior(state["outputs"])
    if prior:
        parts.append(f"Prior agent outputs:\n{prior}")
    return "\n\n".join(parts)


def _make_agent_node(agent: dict):
    """Build a LangGraph node for a single registered agent."""

    def node(state: PipelineState) -> dict:
        log.info("pipeline agent start: %s (prior=%d)", agent["name"], len(state["outputs"]))
        llm = ChatOllama(
            model=agent.get("model", DEFAULT_MODEL),
            base_url=OLLAMA_BASE,
        )
        response = llm.invoke([
            SystemMessage(content=agent["system_prompt"]),
            HumanMessage(content=_build_user_prompt(state)),
        ])
        output = response.content if hasattr(response, "content") else str(response)
        log.info("pipeline agent done: %s (chars=%d)", agent["name"], len(output))

        return {
            "outputs": state["outputs"] + [{"agent": agent["name"], "output": output}],
            "final":   output,
        }

    node.__name__ = f"agent_{agent['name']}"
    return node


def _retrieve_node(state: PipelineState) -> dict:
    """Optional FAISS retrieval step. Tolerant of a missing/unavailable index."""
    try:
        docs = retrieve(state["input"])
        if not docs:
            return {"context": ""}
        return {
            "context": "\n---\n".join(
                f"[{d.get('source')} p.{d.get('page')}]\n{d['content']}"
                for d in docs
            )
        }
    except Exception as e:
        return {"context": f"(retrieval unavailable: {e})"}


def build_graph(use_retrieval: bool = True):
    """Compile a LangGraph whose node order matches the registry."""
    agents = list_agents()
    graph = StateGraph(PipelineState)

    prev = START
    if use_retrieval:
        graph.add_node("retrieve", _retrieve_node)
        graph.add_edge(START, "retrieve")
        prev = "retrieve"

    if not agents:
        graph.add_node("passthrough", lambda s: {"final": s["input"]})
        graph.add_edge(prev, "passthrough")
        graph.add_edge("passthrough", END)
        return graph.compile()

    for agent in agents:
        name = f"agent_{agent['name']}"
        graph.add_node(name, _make_agent_node(agent))
        graph.add_edge(prev, name)
        prev = name

    graph.add_edge(prev, END)
    return graph.compile()


@log_call
def run_pipeline(user_input: str, use_retrieval: bool = True) -> dict:
    """Execute the full sequential pipeline and return a structured result."""
    graph = build_graph(use_retrieval=use_retrieval)
    final_state = graph.invoke({
        "input":   user_input,
        "context": "",
        "outputs": [],
        "final":   "",
    })
    return {
        "input":   user_input,
        "context": final_state.get("context", ""),
        "outputs": final_state["outputs"],
        "final":   final_state["final"],
        "agent_count": len(final_state["outputs"]),
    }


def list_pipeline() -> list[dict]:
    """Return the agents in pipeline execution order (name + description only)."""
    return [{"name": a["name"], "description": a["description"]} for a in list_agents()]
