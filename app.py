"""Streamlit chat frontend for the research agent."""

import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

from agent_core import build_agent

st.set_page_config(page_title="Research Agent", page_icon="*", layout="centered")


@st.cache_resource(show_spinner="Starting agent...")
def load_agent(model: str, temperature: float):
    """Built once per (model, temperature) and reused across reruns."""
    return build_agent(model=model, temperature=temperature)


with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"],
        help="Flash is faster and cheaper; Pro reasons better.",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    show_work = st.toggle("Show tool calls", value=True)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption("Tools: Tavily web search, Weatherstack")

st.title("Research Agent")
st.caption("Ask about current events or the weather anywhere.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            executor = load_agent(model, temperature)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        # The prompt's chat_history placeholder wants LangChain message
        # objects, not the dicts we keep for rendering.
        history = [
            HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
            for m in st.session_state.messages[:-1]
        ]

        callbacks = []
        if show_work:
            callbacks.append(StreamlitCallbackHandler(st.container(), expand_new_thoughts=False))

        try:
            result = executor.invoke(
                {"input": user_input, "chat_history": history},
                {"callbacks": callbacks},
            )
            answer = result["output"]
        except Exception as exc:
            answer = f"Something went wrong: {exc}"
            st.error(answer)
        else:
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
