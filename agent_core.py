"""Agent construction, kept separate from the Streamlit UI.

Secrets resolve from st.secrets first (Streamlit Cloud) then the
environment (local .env), so the same code runs in both places.
"""

import os

import requests
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful research assistant. Use your search tool for anything "
    "current or factual you are unsure of, and your weather tool for weather "
    "questions. Answer concisely and cite sources when you searched."
)


def get_secret(name: str, required: bool = True) -> str | None:
    """Read a secret from st.secrets if available, else the environment."""
    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # Not running under Streamlit, or no secrets.toml present.
        pass

    value = os.getenv(name)
    if required and not value:
        raise RuntimeError(
            f"Missing {name}. Add it to .env locally, or to the app's "
            f"Secrets in Streamlit Cloud."
        )
    return value


@tool
def get_weather_data(city: str) -> str:
    """Get the current weather for a city.

    Use this for any question about current weather, temperature,
    or conditions in a specific place.

    Args:
        city: The city to retrieve weather for, e.g. "Hyderabad".
    """
    api_key = get_secret("WEATHERSTACK_API_KEY", required=False)
    if not api_key:
        return "The weather tool is not configured (WEATHERSTACK_API_KEY is unset)."

    # Free Weatherstack plans are HTTP-only; https 403s on them.
    try:
        resp = requests.get(
            "http://api.weatherstack.com/current",
            params={"access_key": api_key, "query": city},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        return f"Weather service unreachable: {exc}"

    # Weatherstack reports failures in the body with a 200 status.
    if not data.get("success", True):
        info = data.get("error", {}).get("info", "unknown error")
        return f"Weather lookup failed: {info}"

    if "current" not in data:
        return f"No weather data found for {city!r}. Check the city name and try again."

    current = data["current"]
    return (
        f"City: {data.get('location', {}).get('name', city)}\n"
        f"Temperature: {current['temperature']}\u00b0C\n"
        f"Weather: {current['weather_descriptions'][0]}\n"
        f"Humidity: {current['humidity']}%\n"
    )


def build_agent(model: str = "gemini-2.5-flash", temperature: float = 0.0) -> AgentExecutor:
    """Wire up the LLM, tools, prompt and executor."""
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        api_key=get_secret("GEMINI_API_KEY"),
    )

    tools = [
        TavilySearchResults(max_results=3, tavily_api_key=get_secret("TAVILY_API_KEY")),
        get_weather_data,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )


if __name__ == "__main__":
    executor = build_agent()
    result = executor.invoke({"input": "What is India's capital, and its weather?"})
    print(result["output"])
