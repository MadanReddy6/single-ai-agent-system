import os

import requests
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0,api_key=GEMINI_API_KEY)
response = llm.invoke("what is the current year")

response
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant. Use your search tool for anything current, then answer concisely."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

search_tool = TavilySearchResults(max_results=3, tavily_api_key=TAVILY_API_KEY)
@tool
def get_weather_data(city: str) -> str:
    """Get the current weather for a city.

    Use this for any question about current weather, temperature,
    or conditions in a specific place.

    Args:
        city: The city to retrieve weather for, e.g. "Hyderabad".
    """
    # Free Weatherstack plans are HTTP-only; https 403s on them.
    resp = requests.get(
        "http://api.weatherstack.com/current",
        params={"access_key": WEATHERSTACK_API_KEY, "query": city},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    # Weatherstack signals failure in the body with a 200 status.
    if not data.get("success", True):
        return f"Weather lookup failed: {data.get('error', {}).get('info', 'unknown error')}"

    if "current" not in data:
        return f"No weather data found for {city!r}. Check the city name and try again."

    current = data["current"]
    return (
        f"City: {data.get('location', {}).get('name', city)}"
        f"Temperature: {current['temperature']}°C"
        f"Weather: {current['weather_descriptions'][0]}"
        f"Humidity: {current['humidity']}%"
    )
tools = [search_tool, get_weather_data]
agent = create_tool_calling_agent(llm, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = agent_executor.invoke({"input": "what is india's capital? and its wheather and humidity?"})

print(response["output"])