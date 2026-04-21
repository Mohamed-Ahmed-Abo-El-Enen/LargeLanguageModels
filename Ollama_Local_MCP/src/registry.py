import json
from typing import Optional

from src.config import REGISTRY_PATH


def _load() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    
    return {"skills": [], "agents": []}


def _save(reg: dict):
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


# ── Skills ─────────────────────────────────────────────────────────────────────
def register_skill(
    name: str,
    description: str,
    system_prompt: str,
    user_prompt_template: str,
    model: Optional[str] = None,
) -> dict:
    """Register a custom prompt-based skill."""
    reg = _load()
    reg["skills"] = [s for s in reg["skills"] if s["name"] != name]

    skill = {
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        "user_prompt_template": user_prompt_template,
    }
    if model:
        skill["model"] = model

    reg["skills"].append(skill)
    _save(reg)

    return skill


def get_skill(name: str) -> Optional[dict]:
    reg = _load()
    return next((s for s in reg["skills"] if s["name"] == name), None)


def list_skills() -> list[dict]:
    return _load()["skills"]


def delete_skill(name: str) -> bool:
    reg = _load()
    before = len(reg["skills"])
    reg["skills"] = [s for s in reg["skills"] if s["name"] != name]
    _save(reg)
    return len(reg["skills"]) < before


# ── Agents ─────────────────────────────────────────────────────────────────────
def register_agent(
    name: str,
    description: str,
    system_prompt: str,
    tools: list[str],
    model: Optional[str] = None,
) -> dict:
    """Register a named agent that chains multiple tools."""
    reg = _load()
    reg["agents"] = [a for a in reg["agents"] if a["name"] != name]

    agent = {
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        "tools": tools,
    }
    if model:
        agent["model"] = model

    reg["agents"].append(agent)
    _save(reg)

    return agent


def get_agent(name: str) -> Optional[dict]:
    reg = _load()
    return next((a for a in reg["agents"] if a["name"] == name), None)


def list_agents() -> list[dict]:
    return _load()["agents"]


def delete_agent(name: str) -> bool:
    reg = _load()
    before = len(reg["agents"])
    reg["agents"] = [a for a in reg["agents"] if a["name"] != name]
    _save(reg)
    return len(reg["agents"]) < before


# ── Full listing ───────────────────────────────────────────────────────────────
def list_all() -> dict:
    reg = _load()
    return {
        "skills": [{"name": s["name"], "description": s["description"]} for s in reg["skills"]],
        "agents": [{"name": a["name"], "description": a["description"]} for a in reg["agents"]],
    }


# ── Seed defaults  ───────────────────────────────────
def seed_defaults():
    """Populate the registry with starter skills if it is empty."""
    if REGISTRY_PATH.exists():
        return

    defaults = [
        dict(
            name="grammar_fix",
            description="Fix grammar and spelling. Returns corrected text only.",
            system_prompt="Fix all grammar, spelling, and punctuation. Return ONLY the corrected text.",
            user_prompt_template="Fix this: {input}",
        ),
        dict(
            name="sentiment",
            description="Detect sentiment: Positive, Negative, or Neutral.",
            system_prompt="Return ONLY one word: Positive, Negative, or Neutral.",
            user_prompt_template="Sentiment of: {input}",
        ),
        dict(
            name="bullet_points",
            description="Rewrite text as a clean bullet-point list. No intro or outro.",
            system_prompt="Rewrite the input as a concise bullet-point list. Return ONLY the bullets, no preamble or summary sentence.",
            user_prompt_template="Convert to bullet points:\n\n{input}",
        ),
    ]

    for skill in defaults:
        register_skill(**skill)

    register_agent(
        name="news_analyst",
        description="Search for news on a topic and produce an executive summary with sources.",
        system_prompt="You are a senior analyst. Summarize clearly. Cite sources by [number].",
        tools=["web_search", "summarize"],
    )
