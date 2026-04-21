import json
from pathlib import Path

import streamlit as st

import src.tools as T
from src.config import (
    APP_LAYOUT,
    APP_TITLE,
    DEFAULT_MODEL,
    UPLOADS_DIR,
    VISION_MODEL,
    is_tool_capable,
)
from src.db import (
    add_file_ref,
    add_message,
    create_session,
    delete_session,
    export_session,
    get_file_refs,
    get_messages,
    import_session,
    init_db,
    list_sessions,
    memory_list,
    rename_session,
    session_summary,
)
from src.registry import (
    delete_agent,
    delete_skill,
    list_agents,
    list_skills,
    register_agent,
    register_skill,
    seed_defaults,
)
from src.logger import get_logger

log = get_logger(__name__)

init_db()
seed_defaults()
st.set_page_config(page_title=APP_TITLE, layout=APP_LAYOUT)


def _init_state():
    defaults = {
        "session_id": None,
        "messages":   [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _switch_session(sid: str):
    """Load a session from DB into the UI state."""
    st.session_state.session_id = sid
    st.session_state.messages   = [
        {"role": m["role"], "content": m["content"]}
        for m in get_messages(sid)
    ]


def _save_upload(uploaded_file) -> Path:
    """Save a Streamlit uploaded file to UPLOADS_DIR and return its path."""
    dest = UPLOADS_DIR / uploaded_file.name
    dest.write_bytes(uploaded_file.read())
    return dest


def _add_and_display(role: str, content: str):
    """Append to in-memory messages and persist to DB if session is active."""
    st.session_state.messages.append({"role": role, "content": content})
    if st.session_state.session_id:
        add_message(st.session_state.session_id, role, content)


def _get_models() -> list[str]:
    try:
        return T.list_models()
    except Exception:
        return [DEFAULT_MODEL]


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def _sidebar_model() -> str:
    """Model picker + tool-capability banner. Returns the selected model name."""
    st.subheader("Model")
    models = _get_models()
    default_index = models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0
    selected = st.selectbox("Ollama model", models, index=default_index, label_visibility="collapsed")

    if not is_tool_capable(selected):
        st.error(
            f"**`{selected}` does NOT support tool calling.**\n\n"
            "The agent will not be able to run web_search, fetch_page, "
            "run_skill, run_agent, or any other tool. Replies will be plain chat.\n\n"
            "**Fix:** on your Ollama host run one of:\n"
            "```\nollama pull llama3.2\nollama pull qwen2.5\n```\n"
            "Then pick that model above."
        )
    st.caption(f"Vision model for image tools: `{VISION_MODEL}` (set `VISION_MODEL` env var to change)")
    return selected


def _sidebar_sessions(selected_model: str) -> None:
    """Session list + create + active-session actions (rename/delete/export/import/memory/files)."""
    st.subheader("Sessions")

    with st.form("new_session", clear_on_submit=True):
        new_name = st.text_input("Session name", placeholder="My research session")
        if st.form_submit_button("Create", use_container_width=True) and new_name:
            sid = create_session(new_name, model=selected_model)
            _switch_session(sid)
            st.rerun()

    all_sessions = list_sessions()
    if all_sessions:
        st.caption(f"{len(all_sessions)} session(s)")
        for s in all_sessions:
            is_active = s["id"] == st.session_state.session_id
            label = f"{'> ' if is_active else ''}{s['name']}"
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"sess_{s['id']}", use_container_width=True, type=btn_type):
                _switch_session(s["id"])
                st.rerun()

    if st.session_state.session_id:
        _sidebar_active_session(st.session_state.session_id)


def _sidebar_active_session(sid: str) -> None:
    """Actions available only when a session is selected."""
    st.divider()
    st.caption("Session actions")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Info", use_container_width=True):
            st.info(session_summary(sid))
    with col_b:
        if st.button("Delete", use_container_width=True, type="secondary"):
            delete_session(sid)
            st.session_state.session_id = None
            st.session_state.messages = []
            st.rerun()

    new_n = st.text_input("Rename", placeholder="New name…", key="rename_input")
    if st.button("Rename", use_container_width=True) and new_n:
        rename_session(sid, new_n)
        st.rerun()

    if st.button("Export JSON", use_container_width=True):
        data = export_session(sid)
        st.download_button(
            "Download",
            data=json.dumps(data, indent=2),
            file_name=f"session_{sid[:8]}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()
    uploaded_json = st.file_uploader("Import session JSON", type="json", key="import_upload")
    if uploaded_json and st.button("Import", use_container_width=True):
        data = json.loads(uploaded_json.read())
        new_sid = import_session(data)
        _switch_session(new_sid)
        st.rerun()

    mem = memory_list(sid)
    if mem:
        st.divider()
        st.caption("Session memory")
        for k, v in mem.items():
            st.text(f"{k}: {str(v)[:60]}")

    files = get_file_refs(sid)
    if files:
        st.divider()
        st.caption("Files in session")
        for fi in files:
            st.text(f"{fi['file_type']} · {fi.get('label') or Path(fi['file_path']).name}")


def _sidebar_skills() -> None:
    """Custom-skills register form + existing-skills list with delete buttons."""
    st.subheader("Custom Skills")
    with st.expander("Register new skill"):
        with st.form("reg_skill", clear_on_submit=True):
            sk_name  = st.text_input("Name (no spaces)")
            sk_desc  = st.text_input("Description")
            sk_sys   = st.text_area("System prompt")
            sk_tmpl  = st.text_area("User prompt template (use {input})")
            sk_model = st.text_input(
                "Model override (blank = use .env DEFAULT_MODEL)",
                placeholder="e.g. llama3.2:latest",
            )
            if st.form_submit_button("Register", use_container_width=True) and sk_name:
                register_skill(sk_name, sk_desc, sk_sys, sk_tmpl, model=sk_model or None)
                st.success(f"Skill '{sk_name}' registered!")

    existing = list_skills()
    if existing:
        with st.expander(f"{len(existing)} registered skill(s)"):
            for sk in existing:
                col1, col2 = st.columns([3, 1])
                col1.markdown(f"**{sk['name']}** — {sk['description']}")
                if col2.button("Delete", key=f"del_sk_{sk['name']}"):
                    delete_skill(sk["name"])
                    st.rerun()


def _sidebar_agents() -> None:
    """Custom-agents register form + existing-agents list with delete buttons."""
    st.subheader("Custom Agents")
    with st.expander("Register new agent"):
        # Imported lazily so an agent.py import error doesn't blow up the sidebar
        try:
            from src.agent import TOOLS as _AGENT_TOOLS
            available_tools = sorted(t.name for t in _AGENT_TOOLS if t.name != "run_agent")
        except Exception:
            available_tools = []

        with st.form("reg_agent", clear_on_submit=True):
            ag_name  = st.text_input("Name (no spaces)", key="ag_name")
            ag_desc  = st.text_input("Description", key="ag_desc")
            ag_sys   = st.text_area(
                "System prompt (how the sub-agent should behave)",
                key="ag_sys",
                height=120,
            )
            ag_tools = st.multiselect(
                "Tools this agent is expected to use (hint to the planner)",
                options=available_tools,
                default=[],
                help="Descriptive hint only — run_agent gives the sub-agent access to all basic tools.",
            )
            ag_model = st.text_input(
                "Model override (blank = use .env DEFAULT_MODEL)",
                placeholder="e.g. llama3.2:latest",
                key="ag_model",
            )
            if st.form_submit_button("Register agent", use_container_width=True) and ag_name:
                register_agent(ag_name, ag_desc, ag_sys, tools=ag_tools, model=ag_model or None)
                st.success(f"Agent '{ag_name}' registered!")

    existing = list_agents()
    if existing:
        with st.expander(f"{len(existing)} registered agent(s)"):
            for ag in existing:
                col1, col2 = st.columns([3, 1])
                tool_list = ", ".join(ag.get("tools") or []) or "_(no tools pinned)_"
                col1.markdown(f"**{ag['name']}** — {ag['description']}  \nTools: {tool_list}")
                if col2.button("Delete", key=f"del_ag_{ag['name']}"):
                    delete_agent(ag["name"])
                    st.rerun()


def _render_sidebar(model: str) -> str:
    """Render the full sidebar. Returns the selected model."""
    with st.sidebar:
        st.title("Ollama MCP")
        selected_model = _sidebar_model()
        _sidebar_sessions(selected_model)
        st.divider()
        _sidebar_skills()
        st.divider()
        _sidebar_agents()
    return selected_model


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT AREA
# ═══════════════════════════════════════════════════════════════════════════════

def _render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — LLM-driven tool routing (see src/agent.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_attachment_slot():
    counter = st.session_state.get("attach_counter", 0)
    uploaded = st.file_uploader(
        "Attach a PDF or image (optional)",
        type=["pdf", "png", "jpg", "jpeg", "webp", "gif", "bmp"],
        key=f"attach_{counter}",
    )
    if uploaded:
        st.caption(f"Attached: **{uploaded.name}**")
        if st.button("Clear attachment"):
            st.session_state.attach_counter = counter + 1
            st.rerun()
    return uploaded


def main():
    from src.agent import run_router

    model = _render_sidebar(DEFAULT_MODEL)

    session_label = ""
    if st.session_state.session_id:
        sessions = list_sessions()
        s = next((s for s in sessions if s["id"] == st.session_state.session_id), None)
        if s:
            session_label = f"  ·  Session: **{s['name']}**"
    st.caption(
        f"Model: **{model}**{session_label}  ·  "
        "Just chat — the model will call the right tool itself (search, PDF, image, custom skill, …). "
        "Attach a file below if your question is about one."
    )

    if not st.session_state.session_id:
        st.info("Create or select a session from the sidebar to save your conversation.")

    _render_chat_history()

    with st.expander("Attach a file", expanded=False):
        uploaded = _render_attachment_slot()

    user_input = st.chat_input("Ask anything…")
    if not user_input:
        return

    file_path = None
    if uploaded:
        file_path = _save_upload(uploaded)
        if st.session_state.session_id:
            kind = "pdf" if file_path.suffix.lower() == ".pdf" else "image"
            add_file_ref(st.session_state.session_id, str(file_path), kind, label=uploaded.name)

    log.info(
        "turn start: session=%s model=%s file=%s input=%r",
        st.session_state.session_id, model, file_path, user_input[:120],
    )

    displayed = f"[{uploaded.name}]\n\n{user_input}" if uploaded else user_input
    with st.chat_message("user"):
        st.markdown(displayed)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = run_router(
                    user_input=user_input,
                    file_path=file_path,
                    session_id=st.session_state.session_id,
                    model=model,
                )
                log.info("turn done: session=%s reply_chars=%d", st.session_state.session_id, len(result))
            except Exception as exc:
                log.exception("turn failed: session=%s", st.session_state.session_id)
                result = f"**Agent error:** {exc}"
        st.markdown(result)

    _add_and_display("user", displayed)
    _add_and_display("assistant", result)

    if uploaded:
        st.session_state.attach_counter = st.session_state.get("attach_counter", 0) + 1
        st.rerun()


if __name__ == "__main__":
    main()
